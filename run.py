"""
Virus Primer Designer

Stage 1 (_1_download): search NCBI, filter, tick genes and/or whole genomes, download.
Stage 2 (_2_graphs): coverage and conservation graphs of what was downloaded.
Stage 3 ...
"""

import re
import shutil
import sys
import threading

import numpy as np
from PyQt6.QtCore import (
    QAbstractItemModel,
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QSize,
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
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from _1_download import (
    GeneRow,
    Reference,
    configure_entrez,
    download_selected,
    gene_files,
    genomes_path,
    is_unnamed,
    list_genes,
    list_references,
    resolve_taxon,
    seqs_dir,
)
from _2_graphs import GraphsPage, RangeStore

SUGGESTIONS = 10       # entries in a filter field's dropdown
AUTO_EXPAND = 30       # expand every group when the filtered tree has at most this many


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


class FlowLayout(QLayout):
    """
    lays widgets out left to right
    """

    def __init__(self, parent=None, margin: int = 2, spacing: int = 4) -> None:
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items = []

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        lines: list[list] = [[]]   # per line: (item, x, width)
        x = area.x()
        for i, item in enumerate(self._items):
            width = item.sizeHint().width()
            if i == len(self._items) - 1:
                width = max(item.minimumSize().width(), area.right() - x + 1)
            if x + width > area.right() + 1 and lines[-1]:
                lines.append([])
                x = area.x()
                if i == len(self._items) - 1:
                    width = area.width()
            lines[-1].append((item, x, width))
            x += width + self.spacing()
        heights = [max(item.sizeHint().height() for item, _, _ in line) for line in lines if line]
        total = sum(heights) + self.spacing() * (len(heights) - 1)
        if apply:
            y = area.y() + max(0, (area.height() - total) // 2)
            for line, height in zip(lines, heights):
                for item, x, width in line:
                    h = item.sizeHint().height()
                    item.setGeometry(QRect(QPoint(x, y + (height - h) // 2), QSize(width, h)))
                y += height + self.spacing()
        return total + m.top() + m.bottom()


CHIP_QSS = (
    "QFrame#chip { background: rgba(74, 163, 255, 0.22); border: 1px solid rgba(74, 163, 255, 0.70);"
    " border-radius: 7px; }"
    "QFrame#chip QLabel { background: transparent; border: none; padding: 0; margin: 0; }"
    "QToolButton#chipx { background: transparent; border: none; padding: 0; margin: 0; }"
    "QToolButton#chipx:hover { color: #ff6b6b; }"
)


class _Chip(QFrame):
    removed = pyqtSignal()

    def __init__(self, text: str, height: int) -> None:
        super().__init__()
        self.setObjectName("chip")
        self.setStyleSheet(CHIP_QSS)
        self.setFixedHeight(height)
        h = QHBoxLayout(self)
        h.setContentsMargins(7, 0, 3, 0)
        h.setSpacing(3)
        h.addWidget(QLabel(text))
        x = QToolButton()
        x.setObjectName("chipx")
        x.setText("×")
        x.setToolTip("Remove this filter")
        x.setCursor(Qt.CursorShape.PointingHandCursor)
        x.setFixedSize(height - 4, height - 4)
        x.clicked.connect(self.removed)
        h.addWidget(x)


class _Suggestions(QListWidget):
    """
    suggestion dropdown menu
    """
    picked = pyqtSignal(str)

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.itemClicked.connect(lambda item: self.picked.emit(item.text()))
        self.hide()

    def place_under(self, field: QWidget) -> None:
        pos = field.mapTo(self.parentWidget(), QPoint(0, field.height() + 1))
        height = self.sizeHintForRow(0) * self.count() + 2 * self.frameWidth()
        self.setGeometry(pos.x(), pos.y(), field.width(), height)


class _ListWatcher(QObject):

    def __init__(self, field: QWidget, inside, on_outside, on_moved) -> None:
        super().__init__(field)
        self._field = field
        self._inside, self._on_outside, self._on_moved = inside, on_outside, on_moved
        self.installed = False

    def eventFilter(self, obj, event) -> bool:
        if isinstance(obj, QWidget):
            kind = event.type()
            if kind == QEvent.Type.MouseButtonPress and not self._inside(obj):
                self._on_outside()
            elif kind in (QEvent.Type.Move, QEvent.Type.Resize) and (obj is self._field or obj.isAncestorOf(self._field)):
                self._on_moved()
        return False


class ChipFilter(QLineEdit):
    """
    chip filter field: 
    enter turns the text into a chip with a delete (x);
    same as clicking on a name in suggestions menu;
    backspace in an empty field removes the last chip;
    same as clicking on the delete (x);
    """
    changed = pyqtSignal()

    def __init__(self, placeholder: str) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self._flow = FlowLayout(self)
        self.edit = QLineEdit()
        self.edit.setObjectName("chipedit")
        self.edit.setStyleSheet("QLineEdit#chipedit { border: none; background: transparent; padding: 0 2px;"
                                " min-height: 0; }")
        self.edit.setPlaceholderText(placeholder)
        self.edit.setMinimumWidth(140)
        self._flow.addWidget(self.edit)
        self.setFocusProxy(self.edit)
        self._values: list[str] = []
        self.candidates: list[str] = []      # what the suggestions are drawn from
        self._list: _Suggestions | None = None
        self._watcher = _ListWatcher(
            self,
            lambda w: w is self or self.isAncestorOf(w)
            or (self._list is not None and (w is self._list or self._list.isAncestorOf(w))),
            self._hide_list,
            lambda: self._list is not None and self._list.isVisible() and self._list.place_under(self))
        self.edit.textEdited.connect(self._suggest)
        self.edit.installEventFilter(self)

    def values(self) -> list[str]:
        return list(self._values)

    def clear(self) -> None:
        for i in reversed(range(self._flow.count() - 1)):
            chip = self._flow.takeAt(i).widget()
            chip.hide()
            chip.deleteLater()
        self._values = []
        self.edit.clear()
        self.updateGeometry()
        self.changed.emit()

    def _add(self, text: str) -> None:
        text = text.strip()
        if text and text.casefold() not in {v.casefold() for v in self._values}:
            chip = _Chip(text, self.edit.sizeHint().height())
            chip.removed.connect(lambda c=chip, v=text: self._remove(c, v))
            edit_item = self._flow.takeAt(self._flow.count() - 1)
            self._flow.addWidget(chip)
            self._flow.addItem(edit_item)
            self._values.append(text)
            self.updateGeometry()
            self.changed.emit()
        self.edit.clear()
        self._hide_list()

    def _remove(self, chip: QWidget, value: str) -> None:
        if value in self._values:
            self._values.remove(value)
        self._flow.removeWidget(chip)
        chip.hide()
        chip.deleteLater()
        self.updateGeometry()
        self.changed.emit()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(super().sizeHint().height(), self._flow.heightForWidth(width) + 2)

    def sizeHint(self) -> QSize:
        base = super().sizeHint()
        return QSize(base.width(), self.heightForWidth(self.width() or base.width()))

    def minimumSizeHint(self) -> QSize:
        return QSize(super().minimumSizeHint().width(), self.sizeHint().height())

    def _suggest(self, text: str) -> None:
        chosen = {v.casefold() for v in self._values}
        t = text.strip().casefold()
        if t:
            prefix = (c for c in self.candidates if c.casefold().startswith(t))
            inner = (c for c in self.candidates if t in c.casefold() and not c.casefold().startswith(t))
            pool = (c for gen in (prefix, inner) for c in gen)
        else:
            pool = iter(self.candidates)
        top = []
        for c in pool:
            if c.casefold() not in chosen:
                top.append(c)
                if len(top) == SUGGESTIONS:
                    break
        if self._list is None or self._list.parentWidget() is not self.window():
            self._list = _Suggestions(self.window())
            self._list.picked.connect(self._add)
        self._list.clear()
        if not top:
            self._list.hide()
            return
        self._list.addItems(top)
        self._list.setCurrentRow(-1)
        self._list.place_under(self)
        self._list.raise_()
        self._list.show()
        if not self._watcher.installed:
            QApplication.instance().installEventFilter(self._watcher)
            self._watcher.installed = True

    def _hide_list(self) -> None:
        if self._watcher.installed:
            QApplication.instance().removeEventFilter(self._watcher)
            self._watcher.installed = False
        if self._list is not None:
            self._list.hide()

    def eventFilter(self, obj, event) -> bool:
        if obj is not self.edit:
            return super().eventFilter(obj, event)
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            self._suggest(self.edit.text())
        elif kind == QEvent.Type.FocusIn:
            self.update()
            if event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason):
                self._suggest(self.edit.text())
        elif kind == QEvent.Type.FocusOut:
            self.update()
            self._hide_list()
        elif kind == QEvent.Type.KeyPress:
            key = event.key()
            shown = self._list is not None and self._list.isVisible()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and not shown:
                self._suggest(self.edit.text())
                return True
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                # step through the list, wrapping
                n, row = self._list.count(), self._list.currentRow()
                delta = 1 if key == Qt.Key.Key_Down else -1
                self._list.setCurrentRow((row + delta) % n if row >= 0 else (0 if delta > 0 else n - 1))
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                item = self._list.currentItem() if shown else None
                self._add(item.text() if item is not None else self.edit.text())
                return True
            if key == Qt.Key.Key_Escape and shown:
                self._hide_list()
                return True
            if key == Qt.Key.Key_Backspace and not self.edit.text() and self._values:
                self._remove(self._flow.itemAt(self._flow.count() - 2).widget(), self._values[-1])
                return True
        return False

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self.edit.setFocus(Qt.FocusReason.MouseFocusReason)
        self._suggest(self.edit.text())

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._hide_list()


class _Group:
    """
    one top-level row: genome or gene name, with the visible rows under it
    """
    __slots__ = ("row", "key", "label", "children")

    def __init__(self, row: int, key: int, label: str, children: np.ndarray) -> None:
        self.row, self.key, self.label, self.children = row, key, label, children


class GeneTreeModel(QAbstractItemModel):
    """
    two-level tree over the loaded rows: genomes → their genes, or genes → the genomes that have them
    """
    COL_GENOME, COL_NAME, COL_REGION = 0, 1, 2
    selection_changed = pyqtSignal()
    repaint_needed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[GeneRow] = []
        self._acc_code = np.zeros(0, dtype=np.int64)
        self._gene_code = np.zeros(0, dtype=np.int64)
        self._acc_labels: list[str] = []
        self._accessions: list[str] = []
        self._gene_labels: list[str] = []
        self._gene_checked = np.zeros(0, dtype=bool)
        self._genome_checked = np.zeros(0, dtype=bool)
        self._visible = np.zeros(0, dtype=np.int64)
        self._groups: list[_Group] = []
        self.by_gene = False

    def set_rows(self, rows, acc_code, gene_code, accessions, acc_labels, gene_labels) -> None:
        self.beginResetModel()
        self._rows = rows
        self._acc_code, self._gene_code = acc_code, gene_code
        self._accessions, self._acc_labels, self._gene_labels = accessions, acc_labels, gene_labels
        self._gene_checked = np.zeros(len(rows), dtype=bool)
        self._genome_checked = np.zeros(len(accessions), dtype=bool)
        self._visible = np.zeros(0, dtype=np.int64)
        self._shown_rows = np.zeros(len(rows), dtype=bool)
        self._visible_accs = np.zeros(0, dtype=np.int64)
        self._shown_accs = np.zeros(len(accessions), dtype=bool)
        self._groups = []
        self.endResetModel()
        self.selection_changed.emit()

    def set_view(self, visible: np.ndarray, by_gene: bool) -> None:
        self.beginResetModel()
        self.by_gene = by_gene
        self._visible = visible
        codes = (self._gene_code if by_gene else self._acc_code)[visible]
        order = np.argsort(codes, kind="stable")
        rows, codes = visible[order], codes[order]
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]) if codes.size else np.zeros(0, int)
        ends = np.r_[starts[1:], codes.size]
        labels = self._gene_labels if by_gene else self._acc_labels
        self._groups = [_Group(i, int(codes[s]), labels[codes[s]], rows[s:e])
                        for i, (s, e) in enumerate(zip(starts, ends))]
        self._shown_rows = np.zeros(len(self._rows), dtype=bool)
        self._shown_rows[visible] = True
        self._visible_accs = np.unique(self._acc_code[visible])
        self._shown_accs = np.zeros(len(self._accessions), dtype=bool)
        self._shown_accs[self._visible_accs] = True
        self.endResetModel()
        self.selection_changed.emit()

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, column, None)
        return self.createIndex(row, column, self._groups[parent.row()])

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid() or index.internalPointer() is None:
            return QModelIndex()
        return self.createIndex(index.internalPointer().row, 0, None)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if not parent.isValid():
            return len(self._groups)
        if parent.internalPointer() is None and parent.column() == 0:
            return len(self._groups[parent.row()].children)
        return 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 3

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            name = "Gene / genome · isolate" if self.by_gene else "Genome · isolate / gene"
            return ["Genome", name, "Region"][section]
        return None

    def _genome_cell(self, index: QModelIndex) -> int | None:
        """
        in the genome -> genes view - parent rows
        in the gene -> genomes view - child rows
        """
        g = index.internalPointer()
        if not self.by_gene and g is None:
            return self._groups[index.row()].key
        if self.by_gene and g is not None:
            return int(self._acc_code[g.children[index.row()]])
        return None

    def flags(self, index: QModelIndex):
        base = super().flags(index)
        if index.column() == self.COL_NAME or (index.column() == self.COL_GENOME and self._genome_cell(index) is not None):
            return base | Qt.ItemFlag.ItemIsUserCheckable
        return base

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        g = index.internalPointer()
        col = index.column()
        if role == Qt.ItemDataRole.CheckStateRole:
            if col == self.COL_GENOME:
                code = self._genome_cell(index)
                if code is None:
                    return None
                return Qt.CheckState.Checked if self._genome_checked[code] else Qt.CheckState.Unchecked
            if col == self.COL_NAME:
                if g is None:
                    ticked = self._gene_checked[self._groups[index.row()].children]
                    if ticked.all():
                        return Qt.CheckState.Checked
                    return Qt.CheckState.PartiallyChecked if ticked.any() else Qt.CheckState.Unchecked
                return Qt.CheckState.Checked if self._gene_checked[g.children[index.row()]] else Qt.CheckState.Unchecked
            return None
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if g is None:
            group = self._groups[index.row()]
            if col == self.COL_NAME:
                n = len(group.children)
                return f"{group.label}   ({_plural(n, 'genome' if self.by_gene else 'gene')})"
            return ""
        row = self._rows[g.children[index.row()]]
        if col == self.COL_NAME:
            return self._acc_labels[self._acc_code[g.children[index.row()]]] if self.by_gene else row.gene_name
        if col == self.COL_REGION:
            return ", ".join(f"{b}-{e}" for b, e in row.regions)
        return ""

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.CheckStateRole or not index.isValid():
            return False
        checked = Qt.CheckState(value) == Qt.CheckState.Checked
        g = index.internalPointer()
        if index.column() == self.COL_GENOME:
            code = self._genome_cell(index)
            if code is None:
                return False
            self._genome_checked[code] = checked
            self.dataChanged.emit(index, index, [role])
            self.repaint_needed.emit()
        elif index.column() == self.COL_NAME:
            if g is None:
                group = self._groups[index.row()]
                self._gene_checked[group.children] = checked
                self.dataChanged.emit(index, index, [role])
                parent = self.index(index.row(), 0)
                if group.children.size:
                    self.dataChanged.emit(self.index(0, self.COL_NAME, parent),
                                          self.index(group.children.size - 1, self.COL_NAME, parent), [role])
            else:
                self._gene_checked[g.children[index.row()]] = checked
                self.dataChanged.emit(index, index, [role])
                top = self.index(g.row, self.COL_NAME)
                self.dataChanged.emit(top, top, [role])
        else:
            return False
        self.selection_changed.emit()
        return True

    def set_visible_genes(self, checked: bool) -> None:
        self._gene_checked[self._visible] = checked
        self.repaint_needed.emit()
        self.selection_changed.emit()

    def set_visible_genomes(self, checked: bool) -> None:
        self._genome_checked[self._visible_accs] = checked
        self.repaint_needed.emit()
        self.selection_changed.emit()

    def all_visible_genes_checked(self) -> bool:
        return bool(self._visible.size) and bool(self._gene_checked[self._visible].all())

    def all_visible_genomes_checked(self) -> bool:
        return bool(self._visible.size) and bool(self._genome_checked[self._visible_accs].all())

    def counts(self) -> tuple[int, int, int, int]:
        if not self._rows:
            return 0, 0, 0, 0
        return (int(self._gene_checked.sum()), int((self._gene_checked & ~self._shown_rows).sum()),
                int(self._genome_checked.sum()), int((self._genome_checked & ~self._shown_accs).sum()))

    def ticked_gene_rows(self) -> list[GeneRow]:
        return [self._rows[i] for i in np.flatnonzero(self._gene_checked)]

    def ticked_genomes(self) -> list[str]:
        return [self._accessions[i] for i in np.flatnonzero(self._genome_checked)]


class DownloadWorker(QObject):
    taxon_resolved = pyqtSignal(int, str)
    search_progress = pyqtSignal(int)
    genes_loaded = pyqtSignal(list)
    references_loaded = pyqtSignal(list)
    download_progress = pyqtSignal(int, int, str)
    download_notes = pyqtSignal(list)
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
        if not query.strip():
            self.error.emit("input", "Enter a taxid or name")
            self.finished.emit(False)
            return
        try:
            configure_entrez(email, api_key)
            taxid, name = resolve_taxon(query)
            self.taxon_resolved.emit(taxid, name)
            rows = list_genes(taxid, refseq_only, api_key, geo_location=geo_location, annotated_only=annotated_only,
                              complete_only=complete_only, released_since=released_since,
                              progress=lambda n: self.search_progress.emit(n))
        except Exception as e:
            # LookupError: no such taxon; RuntimeError: NCBI Datasets refused
            stage = "taxonomy" if isinstance(e, LookupError) else "datasets" if isinstance(e, RuntimeError) else "network"
            self.error.emit(stage, str(e))
            self.finished.emit(False)
            return
        self.genes_loaded.emit(rows)
        try:
            refs = list_references(taxid, api_key)
        except Exception as e:
            refs = []
            self.error.emit("references", str(e))
        self.references_loaded.emit(refs)
        self.finished.emit(True)

    @pyqtSlot(int, list, list, list, list, str, str)
    def run_download(
        self,
        taxid: int,
        gene_rows: list[GeneRow],
        genome_accessions: list[str],
        annotations: list[GeneRow],
        references: list[Reference],
        email: str,
        api_key: str,
    ) -> None:
        self.cancel_event.clear()
        try:
            configure_entrez(email, api_key)
            notes = download_selected(taxid, gene_rows, genome_accessions, annotations, references, api_key,
                                      self.cancel_event, lambda done, total, msg: self.download_progress.emit(done, total, msg))
            self.download_notes.emit(notes)
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
    request_download = pyqtSignal(int, list, list, list, list, str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Virus Primer Designer")
        self.resize(1000, 700)
        self._taxid: int | None = None
        self._taxon_name = ""
        self._all_rows: list[GeneRow] = []
        self._references: list[Reference] = []
        self._downloading = False
        self.saved_ranges = RangeStore()
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_download_page())
        self.graphs = GraphsPage(self.saved_ranges)
        self.graphs.back_requested.connect(self._on_back)
        self.stack.addWidget(self.graphs)
        self.setCentralWidget(self.stack)

        self.thread = QThread(self)
        self.worker = DownloadWorker()
        self.worker.moveToThread(self.thread)
        self.thread.start()
        self.request_search.connect(self.worker.run_search)
        self.request_download.connect(self.worker.run_download)
        self.worker.taxon_resolved.connect(self._on_taxon_resolved)
        self.worker.search_progress.connect(lambda n: self.count_label.setText(f"Searching… {n:,} genes fetched"))
        self.worker.genes_loaded.connect(self._on_genes_loaded)
        self.worker.references_loaded.connect(self._on_references_loaded)
        self.worker.download_progress.connect(self._on_download_progress)
        self.worker.download_notes.connect(self._on_download_notes)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)

        email, api_key = configure_entrez()
        self.email_edit.setText(email)
        self.api_key_edit.setText(api_key)
        self._on_creds_edited()

    def _build_download_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

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

        self.anon_banner = QLabel("No API key")
        self.anon_banner.setStyleSheet("background: #4a3a00; color: #ffd97a; padding: 6px; border-radius: 4px;")
        self.anon_banner.setVisible(False)
        layout.addWidget(self.anon_banner)

        search_box = QGroupBox("Search NCBI")
        search_v = QVBoxLayout(search_box)
        taxon_row = QHBoxLayout()
        taxon_row.addWidget(QLabel("Taxon:"))
        self.taxon_edit = QLineEdit()
        self.taxon_edit.setPlaceholderText("taxid or name (e.g. 11320 or 'Influenza A virus')")
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
        self.refseq_only_cb.setToolTip("only the RefSeq reference genome's genes")
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
        self.released_since_edit.setToolTip("only genomes released on/after this date, blank = any")
        opts_row2.addWidget(self.released_since_edit)
        opts_row2.addStretch(1)
        search_v.addLayout(opts_row2)
        layout.addWidget(search_box)
        self._on_refseq_toggled()

        filter_box = QGroupBox("Filter loaded genes")
        filter_v = QVBoxLayout(filter_box)
        fields_row = QHBoxLayout()
        fields_row.addWidget(QLabel("Gene(s):"))
        self.gene_filter = ChipFilter("type a gene, Enter to add")
        fields_row.addWidget(self.gene_filter, 1)
        fields_row.addSpacing(12)
        fields_row.addWidget(QLabel("Isolate / accession:"))
        self.isolate_filter = ChipFilter("type an isolate or accession, Enter to add")
        fields_row.addWidget(self.isolate_filter, 1)
        filter_v.addLayout(fields_row)
        fields_row2 = QHBoxLayout()
        fields_row2.addWidget(QLabel("Region:"))
        self.region_edit = QLineEdit()
        self.region_edit.setPlaceholderText("e.g. 21563 or 21563-25384")
        self.region_edit.setClearButtonEnabled(True)
        self.region_edit.setFixedWidth(180)
        fields_row2.addWidget(self.region_edit)
        fields_row2.addSpacing(20)
        self.no_gene_name_cb = QCheckBox("Show entries with no gene name")
        fields_row2.addWidget(self.no_gene_name_cb)
        fields_row2.addStretch(1)
        filter_v.addLayout(fields_row2)
        layout.addWidget(filter_box)

        toolbar_row = QHBoxLayout()
        toolbar_row.addWidget(QLabel("Group by:"))
        self.group_combo = QComboBox()
        self.group_combo.addItems(["Genome → genes", "Gene → genomes"])
        toolbar_row.addWidget(self.group_combo)
        toolbar_row.addSpacing(12)
        self.select_all = QCheckBox("Select all genes")
        self.select_all.setToolTip("Tick every gene shown by the current filters")
        toolbar_row.addWidget(self.select_all)
        self.select_genomes = QCheckBox("Select all whole genomes")
        self.select_genomes.setToolTip("Tick the whole genome of every genome shown by the current filters")
        toolbar_row.addWidget(self.select_genomes)
        toolbar_row.addStretch(1)
        self.count_label = QLabel("")
        toolbar_row.addWidget(self.count_label)
        layout.addLayout(toolbar_row)

        self.tree = QTreeView()
        self.model = GeneTreeModel()
        self.tree.setModel(self.model)
        self.tree.setUniformRowHeights(True)
        self.tree.setTreePosition(1)
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.tree.setColumnWidth(0, 62)
        self.tree.setColumnWidth(1, 430)
        layout.addWidget(self.tree, 1)

        self.selection_label = QLabel("")
        layout.addWidget(self.selection_label)

        action_row = QHBoxLayout()
        self.open_dir_btn = QPushButton("Open output folder")
        action_row.addWidget(self.open_dir_btn)
        self.clear_btn = QPushButton("Clear downloads")
        self.clear_btn.setToolTip("Delete everything downloaded for this taxon")
        self.clear_btn.setStyleSheet("QPushButton { background: #c62828; color: white; border: 1px solid #c62828; }"
                                     "QPushButton:hover { background: #e53935; }"
                                     "QPushButton:disabled { background: #6d3434; color: #c9a9a9; border-color: #6d3434; }")
        action_row.addWidget(self.clear_btn)
        action_row.addSpacing(24)
        self.download_btn = QPushButton("Download")
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
        self.next_btn = QPushButton("Next: graphs →")
        self.next_btn.setEnabled(False)
        action_row.addWidget(self.next_btn)
        layout.addLayout(action_row)

        self.email_edit.textChanged.connect(self._on_creds_edited)
        self.api_key_edit.textChanged.connect(self._on_creds_edited)
        self.taxon_edit.textChanged.connect(self._refresh_button_states)
        self.refseq_only_cb.toggled.connect(self._on_refseq_toggled)
        self.no_gene_name_cb.toggled.connect(self._apply_row_filter)
        self.gene_filter.changed.connect(self._apply_row_filter)
        self.isolate_filter.changed.connect(self._apply_row_filter)
        self.region_edit.textChanged.connect(self._apply_row_filter)
        self.group_combo.currentIndexChanged.connect(self._apply_row_filter)
        self.search_btn.clicked.connect(self._on_search)
        self.download_btn.clicked.connect(self._on_download)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.select_all.clicked.connect(lambda on: self.model.set_visible_genes(on))
        self.select_genomes.clicked.connect(lambda on: self.model.set_visible_genomes(on))
        self.model.selection_changed.connect(self._on_selection_changed)
        self.model.repaint_needed.connect(self.tree.viewport().update)
        self.open_dir_btn.clicked.connect(self._open_output_dir)
        self.clear_btn.clicked.connect(self._on_clear)
        self.next_btn.clicked.connect(self._on_next)
        return page

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

    def _on_selection_changed(self) -> None:
        genes, hidden_genes, genomes, hidden_genomes = self.model.counts()
        parts = [_plural(genes, "gene"), _plural(genomes, "whole genome")]
        self.download_btn.setText("Download " + " + ".join(p for p, n in zip(parts, (genes, genomes)) if n)
                                  if genes or genomes else "Download")
        hidden = [f"{_plural(n, w)}" for n, w in ((hidden_genes, "ticked gene"), (hidden_genomes, "ticked genome")) if n]
        self.selection_label.setText(("Hidden by the current filters but still ticked: " + ", ".join(hidden))
                                     if hidden else "")
        for box, full in ((self.select_all, self.model.all_visible_genes_checked()),
                          (self.select_genomes, self.model.all_visible_genomes_checked())):
            box.blockSignals(True)
            box.setChecked(full)
            box.blockSignals(False)
        self._refresh_button_states()

    def _refresh_button_states(self) -> None:
        has_email = bool(self.email_edit.text().strip())
        has_taxon = bool(self.taxon_edit.text().strip())
        self.search_btn.setEnabled(has_email and has_taxon and not self._downloading)
        self.search_btn.setToolTip("" if has_email else "Email required by NCBI")
        genes, _, genomes, _ = self.model.counts()
        self.download_btn.setEnabled(has_email and bool(genes or genomes) and self._taxid is not None
                                     and not self._downloading)
        on_disk = self._taxid is not None and (bool(gene_files(self._taxid)) or genomes_path(self._taxid).exists())
        self.next_btn.setEnabled(on_disk and not self._downloading)
        self.next_btn.setToolTip("" if on_disk else "Download something first")
        self.clear_btn.setEnabled(on_disk and not self._downloading)
        self.open_dir_btn.setEnabled(on_disk)

    def _on_search(self) -> None:
        released = self.released_since_edit.text().strip()
        if not self.refseq_only_cb.isChecked() and released and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", released):
            self._on_error("input", "Released since must be YYYY-MM-DD (or blank)")
            return
        self._taxid = None
        self.saved_ranges.clear()
        self.resolved_label.setText("")
        self.resolved_label.setStyleSheet("")
        self.count_label.setText("Searching…")
        self.status_label.setText("")
        self._all_rows = []
        self._references = []
        self.gene_filter.clear()
        self.isolate_filter.clear()
        self.region_edit.clear()
        self.model.set_rows([], np.zeros(0, np.int64), np.zeros(0, np.int64), [], [], [])
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
        self._taxon_name = name
        self.resolved_label.setText(f"→ {name} (taxid {taxid})")
        self.resolved_label.setStyleSheet("")
        self._refresh_button_states()

    def _on_genes_loaded(self, rows: list) -> None:
        # RdRp rows are left out of the list
        self._all_rows = [r for r in rows if "rdrp" not in r.gene_name.lower()
                          and "rna-dependent rna polymerase" not in r.gene_name.lower()]
        self._build_filter_index()
        self.gene_filter.candidates = sorted({r.gene_name for r in self._all_rows if not is_unnamed(r.gene_name)},
                                             key=str.casefold)
        self.isolate_filter.candidates = sorted(self._acc_labels, key=str.casefold)
        self.model.set_rows(self._all_rows, self._acc_code, self._name_code,
                            self._accessions, self._acc_labels, self._gene_labels)
        self._apply_row_filter()

    def _build_filter_index(self) -> None:
        """Precompute numpy arrays so row filtering is faster"""
        name_to_code: dict[str, int] = {}
        spellings: list[dict[str, None]] = []
        acc_to_code: dict[str, int] = {}
        accessions: list[str] = []
        acc_labels: list[str] = []
        isolate_keys: dict[str, set[int]] = {}
        codes: list[int] = []
        acc_codes: list[int] = []
        unnamed: list[bool] = []
        reg_begin: list[int] = []
        reg_end: list[int] = []
        reg_row: list[int] = []
        for i, r in enumerate(self._all_rows):
            no_name = is_unnamed(r.gene_name)
            key = "(no gene name)" if no_name else r.gene_name.lower()
            code = name_to_code.setdefault(key, len(name_to_code))
            if code == len(spellings):
                spellings.append({})
            spellings[code]["(no gene name)" if no_name else r.gene_name] = None
            codes.append(code)
            acc = r.genome_accession
            a = acc_to_code.get(acc)
            if a is None:
                a = acc_to_code[acc] = len(accessions)
                accessions.append(acc)
                label = f"{acc} · {r.isolate_name}" if r.isolate_name else acc
                acc_labels.append(label)
                for k in {acc, acc.split(".", 1)[0], label, r.isolate_name}:
                    if k:
                        isolate_keys.setdefault(k.casefold(), set()).add(a)
            acc_codes.append(a)
            unnamed.append(no_name)
            for begin, end in r.regions:
                lo, hi = (begin, end) if begin <= end else (end, begin)
                reg_begin.append(lo)
                reg_end.append(hi)
                reg_row.append(i)
        self._name_to_code = name_to_code
        self._gene_labels = [next(iter(s)) + (f"   (also: {', '.join(list(s)[1:])})" if len(s) > 1 else "")
                             for s in spellings]
        self._accessions = accessions
        self._acc_labels = acc_labels
        self._isolate_keys = isolate_keys
        self._name_code = np.asarray(codes, dtype=np.int64)
        self._acc_code = np.asarray(acc_codes, dtype=np.int64)
        self._unnamed = np.asarray(unnamed, dtype=bool)
        self._reg_begin = np.asarray(reg_begin, dtype=np.int64)
        self._reg_end = np.asarray(reg_end, dtype=np.int64)
        self._reg_row = np.asarray(reg_row, dtype=np.int64)

    def _apply_row_filter(self) -> None:
        n = len(self._all_rows)
        by_gene = self.group_combo.currentIndex() == 1
        if n == 0:
            self.model.set_view(np.zeros(0, dtype=np.int64), by_gene)
            self.count_label.setText("")
            return
        mask = np.ones(n, dtype=bool)
        if not self.no_gene_name_cb.isChecked():
            mask &= ~self._unnamed
        genes = {v.lower() for v in self.gene_filter.values()}
        if genes:
            wanted = [c for nm, c in self._name_to_code.items() if nm in genes]
            mask &= np.isin(self._name_code, wanted)
        isolates = self.isolate_filter.values()
        if isolates:
            wanted_accs = set().union(*(self._isolate_keys.get(v.casefold(), set()) for v in isolates))
            mask &= np.isin(self._acc_code, list(wanted_accs))
        # region "123" or "100-200" (either order); anything else is ignored
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", self.region_edit.text().strip())
        if m:
            lo, hi = sorted((int(m.group(1)), int(m.group(2) or m.group(1))))
            hit = (self._reg_begin <= hi) & (self._reg_end >= lo)
            row_mask = np.zeros(n, dtype=bool)
            row_mask[self._reg_row[hit]] = True
            mask &= row_mask
        visible = np.flatnonzero(mask)
        self.model.set_view(visible, by_gene)
        if self.model.rowCount() <= AUTO_EXPAND:
            self.tree.expandAll()
        genomes = np.unique(self._acc_code[visible]).size
        self.count_label.setText(f"Showing {_plural(visible.size, 'gene')} in {_plural(genomes, 'genome')}"
                                 + ("" if visible.size == n else f" (of {n:,})"))

    def _on_references_loaded(self, refs: list) -> None:
        self._references = refs
        self._refresh_button_states()

    def _on_download(self) -> None:
        if self._taxid is None:
            return
        gene_rows = self.model.ticked_gene_rows()
        genomes = self.model.ticked_genomes()
        if not gene_rows and not genomes:
            return
        self._downloading = True
        self.progress.setRange(0, 0)
        self.status_label.setText("Starting…")
        self.cancel_btn.setVisible(True)
        self._refresh_button_states()
        email, api_key = self._creds()
        wanted = set(genomes)
        annotations = [r for r in self._all_rows if r.genome_accession in wanted]
        self.request_download.emit(self._taxid, gene_rows, genomes, annotations, self._references, email, api_key)

    def _on_download_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.status_label.setText(message)

    def _on_download_notes(self, notes: list) -> None:
        if notes:
            QMessageBox.information(self, "Download finished", "\n\n".join(notes))

    def _on_error(self, stage: str, message: str) -> None:
        self.resolved_label.setText(f"[{stage}] {message}")
        self.resolved_label.setStyleSheet("color: #ff6b6b;")
        if not self._all_rows:
            self.count_label.setText("")

    def _on_finished(self, success: bool) -> None:
        if self._downloading:
            self._downloading = False
            self.cancel_btn.setVisible(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(1 if success else 0)
            if success:
                self.status_label.setText("Downloaded. Next: graphs →")
        self._refresh_button_states()

    def _on_cancel(self) -> None:
        self.worker.cancel_event.set()
        self.status_label.setText("Cancelling…")

    def _on_clear(self) -> None:
        if self._taxid is None:
            return
        answer = QMessageBox.question(
            self, "Clear downloads",
            f"Delete everything downloaded for {self._taxon_name} (taxid {self._taxid})?\n\n{seqs_dir(self._taxid)}",
        )
        if answer == QMessageBox.StandardButton.Yes:
            shutil.rmtree(seqs_dir(self._taxid), ignore_errors=True)
            self.status_label.setText("Downloads cleared")
            self._refresh_button_states()

    def _open_output_dir(self) -> None:
        if self._taxid is not None and seqs_dir(self._taxid).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(seqs_dir(self._taxid))))

    def _on_next(self) -> None:
        if self._taxid is None:
            return
        title = f"{self._taxon_name} (taxid {self._taxid})"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.graphs.load(self._taxid, self._references, title)
        finally:
            QApplication.restoreOverrideCursor()
        self.stack.setCurrentWidget(self.graphs)

    def _on_back(self) -> None:
        self.graphs.close_down()
        self.stack.setCurrentIndex(0)

    def closeEvent(self, event) -> None:
        self.graphs.close_down()
        self.worker.cancel_event.set()
        self.thread.quit()
        self.thread.wait(2000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
