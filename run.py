import re
import sys
import threading
from pathlib import Path

import numpy as np
import qdarktheme
from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QStringListModel,
    Qt,
    QThread,
    QUrl,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from conservation_graph import ConservationDialog, locate_mafft
from coverage_graph import CoverageDialog, RangeStore
from download_gene_seq import (
    GeneRow,
    configure_entrez,
    download_selected,
    group_into_files,
    is_rdrp,
    is_unnamed,
    list_genes,
    resolve_taxon,
)


def _format_regions(regions: list[tuple[int, int]]) -> str:
    return ", ".join(f"{b}-{e}" for b, e in regions)


class _TokenCompleter(QCompleter):
    """
    Autocomplete the last comma-separated token
    """

    def pathFromIndex(self, index) -> str:
        completion = super().pathFromIndex(index)
        text = self.widget().text()
        cut = text.rfind(",")
        head = text[: cut + 1] + " " if cut >= 0 else ""
        return head + completion

    def splitPath(self, path: str) -> list[str]:
        return [path.split(",")[-1].strip()]


class GeneTableModel(QAbstractTableModel):
    HEADERS = ["", "Gene name", "Genome accession", "Isolate", "Region"]

    def __init__(self, rows: list[GeneRow] | None = None) -> None:
        super().__init__()
        self._rows: list[GeneRow] = list(rows or [])
        self._checked: list[bool] = [False] * len(self._rows)
        self._checked_count: int = 0

    def set_rows(self, rows: list[GeneRow]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self._checked = [False] * len(self._rows)
        self._checked_count = 0
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            return self.HEADERS[section]
        return None

    def flags(self, index: QModelIndex):
        base = super().flags(index)
        if index.column() == 0:
            return base | Qt.ItemFlag.ItemIsUserCheckable
        return base

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()
        if col == 0 and role == Qt.ItemDataRole.CheckStateRole:
            return (
                Qt.CheckState.Checked
                if self._checked[index.row()]
                else Qt.CheckState.Unchecked
            )
        if role == Qt.ItemDataRole.DisplayRole:
            return [
                "",
                row.gene_name,
                row.genome_accession,
                row.isolate_name,
                _format_regions(row.regions),
            ][col]
        return None

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole):
        if index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            new = value == Qt.CheckState.Checked.value
            row = index.row()
            if new != self._checked[row]:
                self._checked[row] = new
                self._checked_count += 1 if new else -1
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def set_all_checked(self, checked: bool) -> None:
        if not self._rows:
            return
        self._checked = [checked] * len(self._rows)
        self._checked_count = len(self._rows) if checked else 0
        top = self.index(0, 0)
        bottom = self.index(len(self._rows) - 1, 0)
        self.dataChanged.emit(top, bottom, [Qt.ItemDataRole.CheckStateRole])

    def has_checked(self) -> bool:
        return self._checked_count > 0

    def selected_rows(self) -> list[GeneRow]:
        return [r for r, c in zip(self._rows, self._checked) if c]


class DownloadWorker(QObject):
    taxon_resolved = pyqtSignal(int, str)
    search_progress = pyqtSignal(int)
    genes_loaded = pyqtSignal(list)
    file_done = pyqtSignal(str, int)
    error = pyqtSignal(str, str)
    finished = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.cancel_event = threading.Event()

    @pyqtSlot(str, bool, str, bool, bool, str, str, str)
    def run_search(
        self,
        query: str,
        refseq_only: bool,
        geo_location: str,
        annotated_only: bool,
        complete_only: bool,
        released_since: str,
        email: str,
        api_key: str,
    ) -> None:
        try:
            configure_entrez(email, api_key)
            if not query.strip():
                self.error.emit("input", "Enter a taxid or name")
                self.finished.emit(False)
                return
            try:
                taxon = resolve_taxon(query)
            except LookupError as e:
                self.error.emit("taxonomy", str(e))
                self.finished.emit(False)
                return
            except ValueError as e:
                self.error.emit("input", str(e))
                self.finished.emit(False)
                return
            except Exception as e:
                self.error.emit("network", str(e))
                self.finished.emit(False)
                return
            self.taxon_resolved.emit(taxon.taxid, taxon.scientific_name)
            try:
                rows = list_genes(
                    taxon.taxid,
                    refseq_only,
                    api_key,
                    geo_location=geo_location,
                    annotated_only=annotated_only,
                    complete_only=complete_only,
                    released_since=released_since,
                    progress=lambda n: self.search_progress.emit(n),
                )
            except RuntimeError as e:
                self.error.emit("datasets", str(e))
                self.finished.emit(False)
                return
            except Exception as e:
                self.error.emit("network", str(e))
                self.finished.emit(False)
                return
            self.genes_loaded.emit(rows)
            self.finished.emit(True)
        except Exception as e:
            self.error.emit("network", str(e))
            self.finished.emit(False)

    @pyqtSlot(int, list, str, str)
    def run_download(
        self,
        taxid: int,
        rows: list[GeneRow],
        email: str,
        api_key: str,
    ) -> None:
        self.cancel_event.clear()
        try:
            configure_entrez(email, api_key)
            download_selected(
                taxid,
                rows,
                self.cancel_event,
                lambda path, count: self.file_done.emit(str(path), count),
            )
            self.finished.emit(True)
        except InterruptedError:
            self.finished.emit(False)
        except OSError as e:
            self.error.emit("io", str(e))
            self.finished.emit(False)
        except Exception as e:
            self.error.emit("entrez", str(e))
            self.finished.emit(False)


class MainWindow(QMainWindow):
    request_search = pyqtSignal(str, bool, str, bool, bool, str, str, str)
    request_download = pyqtSignal(int, list, str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Virus Primer Designer")
        self.resize(960, 640)
        self._taxid: int | None = None
        self._all_rows: list[GeneRow] = []
        self._total_files: int = 0
        self._completed_files: int = 0
        self._last_out_dir: Path | None = None
        self._coverage_dialog: CoverageDialog | None = None
        self._conservation_dialog: ConservationDialog | None = None
        # Genome ranges picked on either graph, read by the next pipeline stage
        self.saved_ranges = RangeStore()
        self._build_ui()
        self._build_worker()
        self._load_env_credentials()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        creds_box = QGroupBox("NCBI credentials")
        creds_layout = QHBoxLayout(creds_box)
        creds_layout.addWidget(QLabel("Email:"))
        self.email_edit = QLineEdit()
        creds_layout.addWidget(self.email_edit, 1)
        creds_layout.addWidget(QLabel("API key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        creds_layout.addWidget(self.api_key_edit, 1)
        layout.addWidget(creds_box)

        self.anon_banner = QLabel(
            "No API key"
        )
        self.anon_banner.setStyleSheet(
            "background: #4a3a00; color: #ffd97a; padding: 6px;"
            " border-radius: 4px;"
        )
        self.anon_banner.setVisible(False)
        layout.addWidget(self.anon_banner)

        # --- Search NCBI ---
        search_box = QGroupBox("Search NCBI")
        search_v = QVBoxLayout(search_box)

        taxon_row = QHBoxLayout()
        taxon_row.addWidget(QLabel("Taxon:"))
        self.taxon_edit = QLineEdit()
        self.taxon_edit.setPlaceholderText(
            "taxid or name (e.g. 2697049 or 'SARS-CoV-2')"
        )
        self.taxon_edit.setMinimumWidth(280)
        self.taxon_edit.setMaximumWidth(320)
        taxon_row.addWidget(self.taxon_edit)
        self.search_btn = QPushButton("Search genes")
        taxon_row.addWidget(self.search_btn)
        self.resolved_label = QLabel("")
        self.resolved_label.setWordWrap(True)
        taxon_row.addWidget(self.resolved_label, 1)
        search_v.addLayout(taxon_row)

        opts_row = QHBoxLayout()
        self.refseq_only_cb = QCheckBox("RefSeq only")
        self.refseq_only_cb.setChecked(True)
        self.refseq_only_cb.setToolTip(
            "only the RefSeq reference genome's genes"
        )
        opts_row.addWidget(self.refseq_only_cb)
        opts_row.addSpacing(20)
        self.annotated_only_cb = QCheckBox("Annotated only")
        self.annotated_only_cb.setChecked(True)
        opts_row.addWidget(self.annotated_only_cb)
        opts_row.addSpacing(20)
        self.complete_only_cb = QCheckBox("Complete only")
        self.complete_only_cb.setChecked(True)
        self.complete_only_cb.setToolTip(
            "Drops gene-only / partial deposits. "
            "Uncheck if a very large taxon makes NCBI return a server error."
        )
        opts_row.addWidget(self.complete_only_cb)
        opts_row.addStretch(1)
        search_v.addLayout(opts_row)

        opts_row2 = QHBoxLayout()
        opts_row2.addWidget(QLabel("Location:"))
        self.geo_combo = QComboBox()
        self.geo_combo.setEditable(True)
        self.geo_combo.addItems([
            "", "USA", "United Kingdom", "Germany", "France", "China", "India",
            "Brazil", "Nigeria", "South Africa", "Japan", "Australia",
            "Africa", "Asia", "Europe", "North America", "South America",
        ])
        opts_row2.addWidget(self.geo_combo)
        opts_row2.addSpacing(20)
        opts_row2.addWidget(QLabel("Released since:"))
        self.released_since_edit = QLineEdit()
        self.released_since_edit.setPlaceholderText("YYYY-MM-DD")
        self.released_since_edit.setClearButtonEnabled(True)
        self.released_since_edit.setFixedWidth(180)
        self.released_since_edit.setToolTip(
            "only genomes released on/after this date, blank = any"
        )
        opts_row2.addWidget(self.released_since_edit)
        opts_row2.addStretch(1)
        search_v.addLayout(opts_row2)

        layout.addWidget(search_box)
        self._on_refseq_toggled()

        # --- Filter loaded genes ---
        filter_box = QGroupBox("Filter loaded genes")
        filter_v = QVBoxLayout(filter_box)

        fields_row = QHBoxLayout()
        self.no_gene_name_cb = QCheckBox("Show entries with no gene name")
        fields_row.addWidget(self.no_gene_name_cb)
        fields_row.addSpacing(20)
        fields_row.addWidget(QLabel("Gene name(s):"))
        self._gene_model = QStringListModel([])
        self._gene_completer = _TokenCompleter(self)
        self._gene_completer.setModel(self._gene_model)
        self._gene_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._gene_completer.setCompletionMode(
            QCompleter.CompletionMode.PopupCompletion
        )
        self.gene_name_edit = QLineEdit()
        self.gene_name_edit.setPlaceholderText(
            "comma-separated gene names"
        )
        self.gene_name_edit.setClearButtonEnabled(True)
        self.gene_name_edit.setCompleter(self._gene_completer)
        fields_row.addWidget(self.gene_name_edit, 1)
        fields_row.addSpacing(20)
        fields_row.addWidget(QLabel("Region:"))
        self.region_edit = QLineEdit()
        self.region_edit.setPlaceholderText("e.g. 21563 or 21563-25384")
        self.region_edit.setClearButtonEnabled(True)
        self.region_edit.setFixedWidth(180)
        fields_row.addWidget(self.region_edit)
        filter_v.addLayout(fields_row)
        layout.addWidget(filter_box)

        # --- Table toolbar ---
        toolbar_row = QHBoxLayout()
        self.select_all = QCheckBox("Select all")
        toolbar_row.addWidget(self.select_all)
        self.coverage_btn = QPushButton("Coverage graph")
        self.coverage_btn.setEnabled(False)
        toolbar_row.addWidget(self.coverage_btn)
        self.conservation_btn = QPushButton("Conservation graph")
        self.conservation_btn.setEnabled(False)
        toolbar_row.addWidget(self.conservation_btn)
        toolbar_row.addStretch(1)
        self.count_label = QLabel("")
        toolbar_row.addWidget(self.count_label)
        layout.addLayout(toolbar_row)

        self.table = QTableView()
        self.model = GeneTableModel()
        self.table.setModel(self.model)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 34)
        self.table.setColumnWidth(1, 160)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnWidth(3, 220)
        layout.addWidget(self.table, 1)

        action_row = QHBoxLayout()
        self.download_btn = QPushButton("Download selected")
        self.download_btn.setEnabled(False)
        action_row.addWidget(self.download_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        action_row.addWidget(self.cancel_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        action_row.addWidget(self.progress, 1)
        self.status_label = QLabel("")
        action_row.addWidget(self.status_label, 2)
        self.open_dir_btn = QPushButton("Open output folder")
        self.open_dir_btn.setVisible(False)
        action_row.addWidget(self.open_dir_btn)
        self.next_btn = QPushButton("Next")
        action_row.addWidget(self.next_btn)
        layout.addLayout(action_row)

        self.setCentralWidget(central)

        self.email_edit.textChanged.connect(self._on_creds_edited)
        self.api_key_edit.textChanged.connect(self._on_creds_edited)
        self.taxon_edit.textChanged.connect(self._refresh_button_states)
        self.refseq_only_cb.toggled.connect(self._on_refseq_toggled)
        self.no_gene_name_cb.toggled.connect(self._apply_row_filter)
        self.gene_name_edit.textChanged.connect(self._apply_row_filter)
        self.region_edit.textChanged.connect(self._apply_row_filter)
        self.search_btn.clicked.connect(self._on_search)
        self.download_btn.clicked.connect(self._on_download)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.select_all.stateChanged.connect(self._on_select_all)
        self.model.dataChanged.connect(self._on_model_changed)
        self.open_dir_btn.clicked.connect(self._open_output_dir)
        self.coverage_btn.clicked.connect(self._on_coverage)
        self.conservation_btn.clicked.connect(self._on_conservation)

    def _build_worker(self) -> None:
        self.thread = QThread(self)
        self.worker = DownloadWorker()
        self.worker.moveToThread(self.thread)
        self.thread.start()

        self.request_search.connect(self.worker.run_search)
        self.request_download.connect(self.worker.run_download)
        self.worker.taxon_resolved.connect(self._on_taxon_resolved)
        self.worker.search_progress.connect(self._on_search_progress)
        self.worker.genes_loaded.connect(self._on_genes_loaded)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)

    def _load_env_credentials(self) -> None:
        email, api_key = configure_entrez()
        self.email_edit.setText(email)
        self.api_key_edit.setText(api_key)
        self.anon_banner.setVisible(not api_key)
        self._refresh_button_states()

    def _creds(self) -> tuple[str, str]:
        return self.email_edit.text().strip(), self.api_key_edit.text().strip()

    def _on_creds_edited(self) -> None:
        self.anon_banner.setVisible(not self.api_key_edit.text().strip())
        self._refresh_button_states()

    def _on_refseq_toggled(self) -> None:
        genbank = not self.refseq_only_cb.isChecked()
        self.geo_combo.setEnabled(genbank)
        self.annotated_only_cb.setEnabled(genbank)
        self.complete_only_cb.setEnabled(genbank)
        self.released_since_edit.setEnabled(genbank)

    def _on_model_changed(self, *_args) -> None:
        self._refresh_button_states()

    def _refresh_button_states(self) -> None:
        has_email = bool(self.email_edit.text().strip())
        has_taxon = bool(self.taxon_edit.text().strip())
        downloading = self.cancel_btn.isVisible()
        self.search_btn.setEnabled(has_email and has_taxon and not downloading)
        self.search_btn.setToolTip(
            "" if has_email else "Email required by NCBI"
        )
        any_checked = self.model.has_checked()
        self.download_btn.setEnabled(
            has_email
            and any_checked
            and self._taxid is not None
            and not downloading
        )
        self.coverage_btn.setEnabled(any_checked)
        self.coverage_btn.setToolTip(
            "selected sequences only" if any_checked
            else "Select gene(s) to plot genome coverage"
        )
        self.conservation_btn.setEnabled(
            any_checked and self._taxid is not None and not downloading
        )
        self.conservation_btn.setToolTip(
            "downloaded AND selected genes only"
            if any_checked
            else "Select gene(s) to plot conservation"
        )

    def _on_search(self) -> None:
        released = self.released_since_edit.text().strip()
        if (
            not self.refseq_only_cb.isChecked()
            and released
            and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", released)
        ):
            self._on_error("input", "Released since must be YYYY-MM-DD (or blank)")
            return
        self._taxid = None
        self.saved_ranges.clear()
        self.resolved_label.setText("")
        self.resolved_label.setStyleSheet("")
        self.count_label.setText("Searching…")
        self.status_label.setText("")
        self.open_dir_btn.setVisible(False)
        self._all_rows = []
        self._gene_model.setStringList([])
        self.gene_name_edit.clear()
        self.region_edit.clear()
        self.model.set_rows([])
        self._refresh_button_states()
        email, api_key = self._creds()
        self.request_search.emit(
            self.taxon_edit.text(),
            self.refseq_only_cb.isChecked(),
            self.geo_combo.currentText().strip(),
            self.annotated_only_cb.isChecked(),
            self.complete_only_cb.isChecked(),
            released,
            email,
            api_key,
        )

    def _on_taxon_resolved(self, taxid: int, name: str) -> None:
        self._taxid = taxid
        self.resolved_label.setText(f"→ {name} (taxid {taxid})")
        self.resolved_label.setStyleSheet("")

    def _on_search_progress(self, total: int) -> None:
        self.count_label.setText(f"Searching… {total:,} genes fetched")

    def _on_genes_loaded(self, rows: list) -> None:
        self._all_rows = [r for r in rows if not is_rdrp(r.gene_name)]
        self._build_filter_index()
        self._gene_model.setStringList(
            sorted(
                {
                    r.gene_name
                    for r in self._all_rows
                    if not is_unnamed(r.gene_name)
                }
            )
        )
        self._apply_row_filter()

    def _build_filter_index(self) -> None:
        """
        Precompute numpy arrays so row filtering is vectorised, faster, no stutter for large taxa.
        Region overlap is answered from a flattened (begin, end, row) table.
        """
        name_to_code: dict[str, int] = {}
        codes: list[int] = []
        unnamed: list[bool] = []
        reg_begin: list[int] = []
        reg_end: list[int] = []
        reg_row: list[int] = []
        for i, r in enumerate(self._all_rows):
            name = r.gene_name.lower()
            codes.append(name_to_code.setdefault(name, len(name_to_code)))
            unnamed.append(is_unnamed(r.gene_name))
            for begin, end in r.regions:
                lo, hi = (begin, end) if begin <= end else (end, begin)
                reg_begin.append(lo)
                reg_end.append(hi)
                reg_row.append(i)
        self._name_to_code = name_to_code
        self._name_code = np.asarray(codes, dtype=np.int32)
        self._unnamed = np.asarray(unnamed, dtype=bool)
        self._reg_begin = np.asarray(reg_begin, dtype=np.int64)
        self._reg_end = np.asarray(reg_end, dtype=np.int64)
        self._reg_row = np.asarray(reg_row, dtype=np.int64)

    def _gene_name_filter(self) -> set[str]:
        raw = self.gene_name_edit.text()
        return {t.strip().lower() for t in raw.split(",") if t.strip()}

    def _region_span(self) -> tuple[int, int] | None:
        text = self.region_edit.text().strip()
        if not text:
            return None
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return (min(a, b), max(a, b))
        if text.isdigit():
            n = int(text)
            return (n, n)
        return None

    def _apply_row_filter(self) -> None:
        n = len(self._all_rows)
        if n == 0:
            self.model.set_rows([])
            self._refresh_button_states()
            return
        mask = np.ones(n, dtype=bool)
        if not self.no_gene_name_cb.isChecked():
            mask &= ~self._unnamed
        names = self._gene_name_filter()
        if names:
            wanted = [c for nm, c in self._name_to_code.items() if nm in names]
            mask &= np.isin(self._name_code, wanted)
        span = self._region_span()
        if span is not None:
            lo, hi = span
            hit = (self._reg_begin <= hi) & (self._reg_end >= lo)
            row_mask = np.zeros(n, dtype=bool)
            row_mask[self._reg_row[hit]] = True
            mask &= row_mask
        if mask.all():
            rows = self._all_rows
        else:
            rows = [self._all_rows[i] for i in np.flatnonzero(mask)]
        self.model.set_rows(rows)
        self._refresh_button_states()
        total, shown = n, len(rows)
        self.count_label.setText(
            f"Loaded {total} genes" if shown == total
            else f"Showing {shown} of {total} genes"
        )

    def _on_download(self) -> None:
        if self._taxid is None:
            return
        selected = self.model.selected_rows()
        if not selected:
            return
        groups = group_into_files(self._taxid, selected)
        self._total_files = len(groups)
        self._completed_files = 0
        self._last_out_dir = next(iter(groups)).parent
        self.progress.setRange(0, self._total_files)
        self.progress.setValue(0)
        self.status_label.setText(f"0 / {self._total_files} files")
        self.cancel_btn.setVisible(True)
        self.open_dir_btn.setVisible(False)
        self._refresh_button_states()
        email, api_key = self._creds()
        self.request_download.emit(self._taxid, selected, email, api_key)

    def _on_file_done(self, path: str, count: int) -> None:
        self._completed_files += 1
        self.progress.setValue(self._completed_files)
        suffix = "s" if count != 1 else ""
        self.status_label.setText(
            f"{self._completed_files} / {self._total_files} files — "
            f"wrote {Path(path).name} ({count} record{suffix})"
        )

    def _on_error(self, stage: str, message: str) -> None:
        self.resolved_label.setText(f"[{stage}] {message}")
        self.resolved_label.setStyleSheet("color: #ff6b6b;")
        if not self._all_rows:
            self.count_label.setText("")

    def _on_finished(self, success: bool) -> None:
        self.cancel_btn.setVisible(False)
        if (
            success
            and self._last_out_dir is not None
            and self._completed_files > 0
        ):
            self.open_dir_btn.setVisible(True)
        self._refresh_button_states()

    def _on_cancel(self) -> None:
        self.worker.cancel_event.set()
        self.status_label.setText("Cancelling…")

    def _on_select_all(self, state: int) -> None:
        self.model.set_all_checked(state == Qt.CheckState.Checked.value)

    def _on_coverage(self) -> None:
        selected = self.model.selected_rows()
        if not selected:
            return
        if self._coverage_dialog is not None:
            self._coverage_dialog.close()
        self._coverage_dialog = CoverageDialog(selected, self.saved_ranges, self)
        self._coverage_dialog.show()
        self._coverage_dialog.raise_()
        self._coverage_dialog.activateWindow()

    def _on_conservation(self) -> None:
        if self._taxid is None:
            return
        selected = self.model.selected_rows()
        if not selected:
            return
        mafft = locate_mafft()
        if mafft is None:
            QMessageBox.warning(
                self,
                "MAFFT not found",
                "Could not find MAFFT",
            )
            return
        jobs = [
            (path.stem, path)
            for path in group_into_files(self._taxid, selected)
            if path.exists() and not path.stem.startswith("no_gene_name_")
        ]
        if not jobs:
            QMessageBox.information(
                self,
                "Download the genes first",
                "Conservation is computed from the downloaded gene sequences. "
                "Click “Download selected” for these genes first, then open the conservation graph.",
            )
            return
        spans = [
            (b, e)
            for r in selected
            if not is_unnamed(r.gene_name)
            for b, e in r.regions
        ]
        x_bounds = (
            min(min(b, e) for b, e in spans),
            max(max(b, e) for b, e in spans),
        )
        if self._conservation_dialog is not None:
            self._conservation_dialog.close()
        self._conservation_dialog = ConservationDialog(jobs, mafft, x_bounds, self.saved_ranges, self)
        self._conservation_dialog.show()
        self._conservation_dialog.raise_()
        self._conservation_dialog.activateWindow()

    def _open_output_dir(self) -> None:
        if self._last_out_dir is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self._last_out_dir))
            )

    def closeEvent(self, event) -> None:
        self.worker.cancel_event.set()
        self.thread.quit()
        self.thread.wait(2000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
