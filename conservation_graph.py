"""
Per-gene sequence-conservation graph, plotted with genome coordinates

For each selected and already downloaded gene, one FASTA file per gene, holding gene's CDS across many isolates
Each such file is aligned with MAFFT and a conservation score is computed as

score = OCCUPANCY * (1 - H / 2)

H is the Shannon entropy (bits) over {A,C,G,T} in the column (position)
OCCUPANCY is the fraction of non-gap sequences
The score is in range [0, 1] where 1 = every isolate agrees on a base and 0 = maximum variability or gaps

Columns are projected onto a reference sequence and placed at that reference's genome positions

Handles negative-strand genes.
"""

import os
import re
import subprocess
import shutil
import sys
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from Bio import SeqIO
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from coverage_graph import _smooth

MAX_SEQS = 100          # seqs per gene cap for faster MSA, if exceded pick out of all seqs n = MAX_SEQS randomly
MIN_SEQS = 3            # fewer than this can't show conservation
RANDOM_SEED = 67        # subsample seed for random seq picking
DEFAULT_WINDOW = 100
MIN_WINDOW = 10
MAX_WINDOW = 500

# Colours cycled across gene tracks
PALETTE = ["#4aa3ff", "#ff6b6b", "#5fd35f", "#ffd93d", "#a06cd5", "#ff8c42", "#2ec4b6", "#e84393", "#5c7cfa", "#f6c453", "#20c997", "#d6336c", "#845ef7", "#74c0fc", "#ff922b", "#38d9a9"]


def locate_mafft() -> list[str] | None:
    """
    Return a command prefix for MAFFT, or None if it can't be found.
    Only checks PATH and the folder containing this script.

    TODO: test for mac and linux
    """
    found = shutil.which("mafft")
    if found:
        return ["cmd", "/c", found] if found.lower().endswith(".bat") else [found]

    here = Path(__file__).resolve().parent

    candidates = [
        here / "mafft.bat",
        here / "mafft" / "mafft.bat",
        here / "mafft-win" / "mafft.bat",
        here / "mafft" / "mafft-win" / "mafft.bat",
        here / "mafft" / "mafft-mac" / "mafft.bat",
    ]
    for cand in candidates:
        if cand.is_file():
            if cand.suffix.lower() == ".bat":
                return ["cmd", "/c", str(cand)]
            return [str(cand)]

    return None

def _parse_regions(rec_id: str) -> list[tuple[int, int]]:
    """
    Handles records like 'XX_0123456.1:111-11111,22222-33333'
    """
    if ":" not in rec_id:
        return []
    _, ranges = rec_id.split(":", 1)
    out: list[tuple[int, int]] = []
    for part in ranges.split(","):
        m = re.fullmatch(r"(\d+)-(\d+)", part.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.append((min(a, b), max(a, b)))
    return out


def _genome_positions(regions: list[tuple[int, int]]) -> np.ndarray:
    pos: list[int] = []
    for b, e in sorted(regions):
        pos.extend(range(b, e + 1))
    return np.asarray(pos, dtype=np.int64)


def _subsample(records: list, n: int) -> list:
    """
    Cap to n records, always keeping the longest for the mapping reference
    """
    if len(records) <= n:
        return records
    ref = max(range(len(records)), key=lambda i: len(records[i].seq))
    rng = np.random.default_rng(RANDOM_SEED)
    others = [i for i in range(len(records)) if i != ref]
    keep = set(rng.choice(others, size=n - 1, replace=False).tolist())
    keep.add(ref)
    return [records[i] for i in sorted(keep)]


def _reference_index(aln: list) -> int:
    for i, r in enumerate(aln):
        acc = r.id.split(":", 1)[0]
        if acc[:3] in ("NC_", "AC_", "NG_"):
            return i
    return max(
        range(len(aln)),
        key=lambda i: sum(1 for ch in str(aln[i].seq) if ch != "-"),
    )


def _column_scores(aln: list) -> np.ndarray:
    """
    Per-column occupancy * (1 - H/2) over {A,C,G,T}
    """
    mat = np.array([list(str(r.seq).upper()) for r in aln], dtype="U1")
    n = mat.shape[0]
    counts = np.stack([(mat == b).sum(axis=0) for b in ("A", "C", "G", "T")])
    n_acgt = counts.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / n_acgt
        plogp = np.where(p > 0, p * np.log2(p), 0.0)
        entropy = -plogp.sum(axis=0)
    occupancy = n_acgt / n
    score = occupancy * (1.0 - entropy / 2.0)
    score[n_acgt == 0] = 0.0
    return score


class ConservationWorker(QObject):
    gene_done = pyqtSignal(str, object, object, int, int)
    gene_error = pyqtSignal(str, str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()

    def __init__(self, mafft_cmd: list[str]) -> None:
        super().__init__()
        self._mafft = mafft_cmd
        self._cancel = False
        self._proc: subprocess.Popen | None = None

    def cancel(self) -> None:
        self._cancel = True
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    @pyqtSlot(list)
    def run(self, jobs: list) -> None:
        total = len(jobs)
        for i, (name, path) in enumerate(jobs):
            if self._cancel:
                break
            try:
                result = self._process(Path(path))
                if result is None:
                    self.gene_error.emit(
                        name,
                        "needs ≥3 isolate sequences - search in GenBank mode",
                    )
                else:
                    x, raw, sampled, n_total = result
                    self.gene_done.emit(name, x, raw, sampled, n_total)
            except InterruptedError:
                break
            except Exception as e:
                self.gene_error.emit(name, str(e)[:200])
            self.progress.emit(i + 1, total)
        self.finished.emit()

    def _run_mafft(self, records: list) -> list:
        """
        Align records with MAFFT subprocess
        """
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False, encoding="ascii"
        )
        try:
            SeqIO.write(records, tmp, "fasta")
            tmp.close()
            flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
            self._proc = subprocess.Popen(
                self._mafft + ["--auto", "--quiet", tmp.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=flags,
            )
            out, err = self._proc.communicate(timeout=300)
            rc = self._proc.returncode
            if self._cancel:
                raise InterruptedError("cancelled")
            if rc != 0:
                raise RuntimeError(f"MAFFT failed: {err[:200]}")
            return list(SeqIO.parse(StringIO(out), "fasta"))
        finally:
            self._proc = None
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _process(self, path: Path):
        records = list(SeqIO.parse(str(path), "fasta"))
        n_total = len(records)
        if n_total < MIN_SEQS:
            return None
        records = _subsample(records, MAX_SEQS)
        sampled = len(records)
        aln = self._run_mafft(records)
        if len(aln) < MIN_SEQS:
            return None
        score = _column_scores(aln)
        ref_idx = _reference_index(aln)
        ref_seq = str(aln[ref_idx].seq).upper()
        ungapped = np.array([ch != "-" for ch in ref_seq])
        ref_score = score[ungapped]
        regions = _parse_regions(aln[ref_idx].id)
        positions = _genome_positions(regions) if regions else np.array([])
        if positions.size == ref_score.size and positions.size > 0:
            x = positions
        elif regions:
            g_lo = min(b for b, _ in regions)
            g_hi = max(e for _, e in regions)
            x = np.linspace(g_lo, g_hi, ref_score.size)
        else:
            x = np.arange(ref_score.size)
        return x.astype(float), ref_score, sampled, n_total


class ConservationDialog(QDialog):
    request_run = pyqtSignal(list)

    def __init__(
        self,
        jobs: list,
        mafft_cmd: list[str],
        x_bounds: tuple[int, int],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sequence conservation")
        self.resize(960, 560)
        self._jobs = list(jobs)
        # name -> (x, raw_score, sampled, total)
        self._raw: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
        # name -> (x, smoothed_y)
        self._smoothed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._items: dict[str, pg.PlotDataItem] = {}
        self._frozen = False
        self._x_lo = float(x_bounds[0])
        self._x_hi = float(x_bounds[1])
        self._build_ui()
        self._start_worker(mafft_cmd)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Smoothing window (bp):"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(MIN_WINDOW, MAX_WINDOW)
        self.window_spin.setSingleStep(5)
        self.window_spin.setValue(DEFAULT_WINDOW)
        self.window_spin.valueChanged.connect(self._redraw_all)
        controls.addWidget(self.window_spin)
        self.reset_btn = QPushButton("Reset view")
        self.reset_btn.clicked.connect(self._reset_view)
        controls.addWidget(self.reset_btn)
        controls.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setRange(0, len(self._jobs))
        self.progress.setFixedWidth(160)
        controls.addWidget(self.progress)
        self.status = QLabel("Aligning…")
        controls.addWidget(self.status)
        outer.addLayout(controls)

        body = QHBoxLayout()
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.setLabel("bottom", "Genome position (bp)")
        self.plot.setLabel("left", "Conservation (1 = identical)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.getPlotItem().setContentsMargins(6, 6, 16, 12)
        self.plot.setClipToView(True)
        self.plot.setLimits(yMin=-0.02, yMax=1.05, minYRange=0.1)
        self.plot.setYRange(0, 1, padding=0)
        if self._x_hi > self._x_lo:
            self.plot.setLimits(xMin=self._x_lo, xMax=self._x_hi)
            self.plot.setXRange(self._x_lo, self._x_hi, padding=0.02)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888"))
        self.vline.setVisible(False)
        self.plot.addItem(self.vline, ignoreBounds=True)
        body.addWidget(self.plot, 1)

        panel = QWidget()
        panel_v = QVBoxLayout(panel)
        panel_v.setContentsMargins(0, 0, 0, 0)
        self.readout = QLabel(
            "Hover over a track to read its conservation.\n"
            "Click the plot to freeze the readout, click again to resume"
        )
        self.readout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.readout.setWordWrap(True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.readout)
        scroll.setFixedWidth(240)
        panel_v.addWidget(scroll)
        body.addWidget(panel)
        outer.addLayout(body)

        self._proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved
        )
        self.plot.scene().sigMouseClicked.connect(self._on_click)

    def _start_worker(self, mafft_cmd: list[str]) -> None:
        self._thread = QThread(self)
        self._worker = ConservationWorker(mafft_cmd)
        self._worker.moveToThread(self._thread)
        self.request_run.connect(self._worker.run)
        self._worker.gene_done.connect(self._on_gene_done)
        self._worker.gene_error.connect(self._on_gene_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()
        self.request_run.emit(self._jobs)

    def _color(self) -> str:
        return PALETTE[len(self._items) % len(PALETTE)]

    def _on_gene_done(self, name, x, raw, sampled, n_total) -> None:
        self._raw[name] = (x, raw, sampled, n_total)
        item = self.plot.plot([], [], pen=pg.mkPen(self._color(), width=2))
        self._items[name] = item
        self._draw(name)
        lo, hi = float(x.min()), float(x.max())
        if lo < self._x_lo or hi > self._x_hi:
            self._x_lo = min(self._x_lo, lo)
            self._x_hi = max(self._x_hi, hi)
            self.plot.setLimits(xMin=self._x_lo, xMax=self._x_hi)

    def _on_gene_error(self, name, message) -> None:
        prev = "" if self.readout.text().startswith("Hover") else self.readout.text() + "\n"
        self.readout.setText(prev + f"err {name}: {message}")

    def _on_progress(self, done, total) -> None:
        self.progress.setValue(done)
        self.status.setText(f"Aligning… {done}/{total}")

    def _on_finished(self) -> None:
        n = len(self._items)
        self.status.setText(f"Done — {n} gene track(s)" if n else "No tracks to show")

    def _draw(self, name: str) -> None:
        x, raw, _, _ = self._raw[name]
        y = _smooth(raw, self.window_spin.value())
        self._smoothed[name] = (x, y)
        self._items[name].setData(x, y)

    def _redraw_all(self) -> None:
        for name in self._items:
            self._draw(name)

    def _reset_view(self) -> None:
        if self._x_hi > self._x_lo:
            self.plot.setXRange(self._x_lo, self._x_hi, padding=0.02)
            self.plot.setYRange(0, 1, padding=0)

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._frozen = not self._frozen
        self.setWindowTitle(
            "Sequence conservation - FROZEN (click to resume)"
            if self._frozen
            else "Sequence conservation"
        )

    def _on_mouse_moved(self, event) -> None:
        if self._frozen:
            return
        scene_pos = event[0]
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return
        x = self.plot.getViewBox().mapSceneToView(scene_pos).x()
        self.vline.setPos(x)
        self.vline.setVisible(True)
        lines = [f"Position {int(round(x)):,} bp", ""]
        hits = []
        for name, (xs, ys) in self._smoothed.items():
            if xs.size and xs[0] <= x <= xs[-1]:
                idx = int(np.searchsorted(xs, x))
                idx = min(max(idx, 0), len(ys) - 1)
                sampled, total = self._raw[name][2], self._raw[name][3]
                n = f"n={total}" if total <= sampled else f"n={sampled} of {total}"
                hits.append((name, float(ys[idx]), n))
        if hits:
            hits.sort(key=lambda h: h[1], reverse=True)
            lines.append("Conservation here:")
            lines.extend(f"  • {nm}: {v:.2f}  ({n})" for nm, v, n in hits)
        else:
            lines.append("No gene track at this position.")
        self.readout.setText("\n".join(lines))

    def closeEvent(self, event) -> None:
        self._worker.cancel()
        self._thread.quit()
        self._thread.wait(3000)
        super().closeEvent(event)
