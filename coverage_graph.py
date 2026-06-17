"""
Genome-coverage graph for selected genes

Coverage depth at a genome position = how many of the selected CDS regions span that base
The curve is smoothed with a moving average (adjustable)
Hovering over graphshows which genes cover the position under the cursor
Click the plot to freeze the readout so the gene list can be scrolled
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from download_gene_seq import GeneRow, is_unnamed

DEFAULT_WINDOW = 100
MIN_WINDOW = 10
MAX_WINDOW = 500

def _display_name(row: GeneRow) -> str:
    if is_unnamed(row.gene_name):
        return f"(unnamed) {row.genome_accession}"
    return row.gene_name


def _smooth(cov: np.ndarray, window: int) -> np.ndarray:
    """
    Edge-normalised moving average: each point is divided by the number of positions that contributed, not by the full window
    Interior points are just the mean, near the ends the average uses only the available data instead of going toward zero
    """
    if window <= 1 or cov.size == 0:
        return cov.astype(float)
    values = cov.astype(float)
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values), kernel, mode="same")
    return numerator / denominator


class CoverageDialog(QDialog):
    """
    Modeless window plotting smoothed coverage with a hover readout
    """

    def __init__(self, rows: list[GeneRow], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Genome coverage")
        self.resize(900, 540)
        self._rows = list(rows)
        self._frozen = False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._prepare_data()
        finally:
            QApplication.restoreOverrideCursor()
        self._build_ui()
        self._replot()

    def _prepare_data(self) -> None:
        """
        Build numpy arrays for the curve and for fast hover lookups
        """
        # Flatten every region into parallel arrays (begin, end, gene-name)
        names: dict[str, int] = {}
        begins: list[int] = []
        ends: list[int] = []
        codes: list[int] = []
        for row in self._rows:
            code = names.setdefault(_display_name(row), len(names))
            for begin, end in row.regions:
                lo, hi = (begin, end) if begin <= end else (end, begin)
                begins.append(lo)
                ends.append(hi)
                codes.append(code)
        self._names = list(names)
        self._begins = np.asarray(begins, dtype=np.int64)
        self._ends = np.asarray(ends, dtype=np.int64)
        self._codes = np.asarray(codes, dtype=np.int64)

        self._ymax = 1.0
        if self._begins.size == 0:
            self._lo = self._hi = 0
            self._cov = np.zeros(1, dtype=np.int32)
            return

        self._lo = int(self._begins.min())
        self._hi = int(self._ends.max())
        # Per-base coverage via a difference array + cumulative sum
        diff = np.zeros(self._hi + 2, dtype=np.int32)
        np.add.at(diff, self._begins, 1)
        np.add.at(diff, self._ends + 1, -1)
        self._cov = np.cumsum(diff)[: self._hi + 1]

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Smoothing window (bp):"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(MIN_WINDOW, MAX_WINDOW)
        self.window_spin.setSingleStep(5)
        self.window_spin.setValue(DEFAULT_WINDOW)
        self.window_spin.valueChanged.connect(self._replot)
        controls.addWidget(self.window_spin)
        self.reset_btn = QPushButton("Reset view")
        self.reset_btn.clicked.connect(self._reset_view)
        controls.addWidget(self.reset_btn)
        controls.addStretch(1)
        ngenomes = len({r.genome_accession for r in self._rows})
        controls.addWidget(
            QLabel(f"{len(self._rows)} genes · {ngenomes} genome(s)")
        )
        outer.addLayout(controls)

        body = QHBoxLayout()
        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.setLabel("bottom", "Genome position (bp)")
        self.plot.setLabel("left", "Coverage depth (genomes)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.getPlotItem().setContentsMargins(6, 6, 16, 12)
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")
        if self._hi > self._lo:
            self._ymax = float(self._cov.max()) + 1.0
            self.plot.setLimits(
                xMin=self._lo,
                xMax=self._hi,
                yMin=0,
                yMax=self._ymax,
                minXRange=min(10, self._hi - self._lo),
                minYRange=min(1.0, self._ymax),
            )
            self._reset_view()
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#4aa3ff", width=2))
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888"))
        self.vline.setVisible(False)
        self.plot.addItem(self.vline, ignoreBounds=True)
        body.addWidget(self.plot, 1)

        panel = QWidget()
        panel_v = QVBoxLayout(panel)
        panel_v.setContentsMargins(0, 0, 0, 0)
        self.readout = QLabel(
            "Hover over the curve to list the genes covering that position.\n"
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
            self.plot.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._on_mouse_moved,
        )
        self.plot.scene().sigMouseClicked.connect(self._on_click)

    def _reset_view(self) -> None:
        if self._hi > self._lo:
            self.plot.setXRange(self._lo, self._hi, padding=0)
            self.plot.setYRange(0, self._ymax, padding=0)

    def _replot(self) -> None:
        smoothed = _smooth(self._cov, self.window_spin.value())
        if smoothed.size <= self._lo:
            self.curve.setData([], [])
            return
        x = np.arange(self._lo, smoothed.size)
        self.curve.setData(x, smoothed[self._lo :])

    def _genes_at(self, pos: int) -> list[tuple[str, int]]:
        """
        (gene name, genome count) for genes whose region contains pos
        """
        mask = (self._begins <= pos) & (self._ends >= pos)
        if not mask.any():
            return []
        counts = np.bincount(self._codes[mask], minlength=len(self._names))
        order = np.argsort(counts)[::-1]
        return [(self._names[i], int(counts[i])) for i in order if counts[i] > 0]

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._frozen = not self._frozen
        self.setWindowTitle(
            "Genome coverage - FROZEN (click to resume)"
            if self._frozen
            else "Genome coverage"
        )

    def _on_mouse_moved(self, event) -> None:
        if self._frozen:
            return
        scene_pos = event[0]
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return
        view_pos = self.plot.getViewBox().mapSceneToView(scene_pos)
        pos = int(round(view_pos.x()))
        pos = max(self._lo, min(pos, self._hi))
        self.vline.setPos(pos)
        self.vline.setVisible(True)
        depth = int(self._cov[pos]) if pos < self._cov.size else 0
        genes = self._genes_at(pos)
        lines = [f"Position {pos:,} bp", f"Coverage depth: {depth}", ""]
        if genes:
            lines.append("Genes here:")
            lines.extend(
                f"  • {name}" + (f" ×{n}" if n > 1 else "")
                for name, n in genes
            )
        else:
            lines.append("No genes cover this position.")
        self.readout.setText("\n".join(lines))
