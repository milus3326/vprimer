"""
Stage 2: graphs

Three tabs on one page, built from what stage 1 downloaded:
  Coverage                      how many downloaded CDS regions span each position (drawn at start, no alignment)
  Conservation · whole genome   downloaded whole genomes aligned onto a RefSeq reference
  Conservation · genes          downloaded gene sequences aligned onto the reference's CDS of each gene

The conservation tabs only align (MAFFT) when "Align and generate graph" is pressed.
Finished alignments are kept in seqs/<taxid>/alignments/ and reused. 
Each reference position is scored as:

score = OCCUPANCY * (1 - H / 2)

H is the Shannon entropy (bits) over {A,C,G,T} at that position
OCCUPANCY is the fraction of sequences with a base (not a gap) at that position
The score is in range [0, 1] where
1 = every sequence agrees on a base
0 = maximum variability or gaps

One value per position (overlapping genes read the same)
Left-drag saves a coordinate range (left-click a single position), right-drag pans, wheel zooms
"""

import html
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from _1_download import (
    ALIGN_DIR,
    GeneRow,
    Reference,
    _base_acc,
    annotations_path,
    fasta_ids,
    gene_files,
    genomes_path,
    is_unnamed,
    reference_path,
    seqs_dir,
)

DEFAULT_WINDOW = 100
MIN_WINDOW = 10
MAX_WINDOW = 500
MIN_SEQS = 3            # fewer than this can't show conservation
MIN_REF_COVERAGE = 0.5  # aligned sequences covering less of the reference are dropped
GAP = ord("-")
INTERGENIC = "#8a8a8a"  # conservation curve colour outside reference genes
MOUSE_HELP = "Left-drag: save range · click: save position\nRight-drag: pan · wheel: zoom"

# Colours cycled across reference genes, in genome order
PALETTE = ["#4aa3ff", "#ff6b6b", "#5fd35f", "#ffd93d", "#a06cd5", "#ff8c42", "#2ec4b6", "#e84393", "#5c7cfa", "#f6c453", "#20c997", "#d6336c", "#845ef7", "#74c0fc", "#ff922b", "#38d9a9"]
COMPLEMENT = bytes.maketrans(b"ACGT", b"TGCA")


def genome_alignment_path(taxid: int, ref: Reference) -> Path:
    return seqs_dir(taxid) / ALIGN_DIR / f"{ref.accession}_alignment.fasta"


def isolate_genes_path(taxid: int, ref: Reference) -> Path:
    return seqs_dir(taxid) / ALIGN_DIR / f"{ref.accession}_genes.tsv"


def gene_alignment_path(taxid: int, ref: Reference, gene_file: Path) -> Path:
    return seqs_dir(taxid) / ALIGN_DIR / ref.accession / f"{gene_file.stem}_alignment.fasta"


_SEGMENTS: dict[tuple, Counter] = {}


def genomes_for(taxid: int, ref: Reference) -> int:
    """
    downloaded genomes that would be aligned to reference
    """
    path = genomes_path(taxid)
    if not path.exists():
        return 0
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _SEGMENTS:
        with open(path, encoding="ascii", errors="replace") as f:
            _SEGMENTS[key] = Counter((re.findall(r"\[segment=([^\]]*)\]", line) or [""])[0]
                                     for line in f if line.startswith(">"))
    counts = _SEGMENTS[key]
    return counts.get(ref.segment, 0) if ref.segment else sum(counts.values())


_COUNTS: dict[tuple, int] = {}


def count_records(path: Path) -> int:
    """
    number of FASTA records
    """
    if not path.exists():
        return 0
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _COUNTS:
        n, prev = 0, b"\n"
        with open(path, "rb") as f:
            while chunk := f.read(1 << 24):
                n += chunk.count(b"\n>") + (prev == b"\n" and chunk[:1] == b">")
                prev = chunk[-1:]
        _COUNTS[key] = n
    return _COUNTS[key]


def _spellings(gene_file: Path) -> Counter:
    """
    how many genomes use each spelling of the gene name in a genes/ file
    """
    seen = set()
    with open(gene_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">") and "[gene=" in line:
                seen.add((_base_acc(line[1:].split(None, 1)[0]), line.split("[gene=", 1)[1].split("]", 1)[0]))
    return Counter(name for _, name in seen)


def locate_mafft() -> list[str] | None:
    """
    return a command prefix for MAFFT, or None if it can't be found,
    checks PATH, then the portable package for this OS next to this script (mafft-win / mafft-mac / mafft-linux64)
    or inside a mafft/ folder
    """
    sub = {"win32": "mafft-win", "darwin": "mafft-mac"}.get(sys.platform, "mafft-linux64")
    here = Path(__file__).resolve().parent
    candidates = [shutil.which("mafft")] + [base / name for base in (here, here / "mafft")
                                            for name in (Path(sub) / "mafft.bat", Path("mafft.bat"))]
    for cand in candidates:
        if not cand or not Path(cand).is_file():
            continue
        if not str(cand).lower().endswith(".bat"):
            return [str(cand)]
        # the mac/linux mafft.bat finds its own folder with pushd, which dash (/bin/sh on Debian/Ubuntu)
        # doesn't have, so run it with bash
        return ["cmd", "/c", str(cand)] if sys.platform == "win32" else [shutil.which("bash") or "/bin/sh", str(cand)]
    return None


def _kill_tree(proc: subprocess.Popen) -> None:
    """
    MAFFT is a launcher script that starts worker processes stop the whole tree, so nothing keeps writing after a cancel
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def _as_bytes(rec: SeqRecord) -> np.ndarray:
    return np.frombuffer(str(rec.seq).upper().encode("ascii", "replace"), dtype=np.uint8)


def _align(folder: Path, reference: SeqRecord, mafft_cmd: list[str] | None, cancel_event: threading.Event):
    """
    each sequence is added on its own (--addfragments)
    no --keeplength, insertions stay in the output, so every isolate base maps exactly onto the reference
    keeping only the reference columns afterwards gives exactly what --keeplength would
    returns aligned records, reference first, uppercase, isolate -> reference position maps, how many isolates were dropped (< MIN_REF_COVERAGE)
    """
    SeqIO.write([reference], folder / "reference.fasta", "fasta")
    records = [reference]
    if (folder / "genomes.fasta").exists() and (folder / "genomes.fasta").stat().st_size:
        if mafft_cmd is None:
            raise RuntimeError("MAFFT not found: install it (see README) to align the sequences")
        cmd = mafft_cmd + ["--quiet", "--thread", str(os.cpu_count() or 1), "--6merpair", "--adjustdirection",
                           "--addfragments", "genomes.fasta", "reference.fasta"]
        extra = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True} # own process group, so _kill_tree can stop it all
        with open(folder / "mafft_out.fasta", "w") as out, open(folder / "mafft.log", "w") as log:
            proc = subprocess.Popen(cmd, cwd=folder, stdout=out, stderr=log, **extra)
            while proc.poll() is None:
                if cancel_event.wait(0.5):
                    _kill_tree(proc)
                    raise InterruptedError("cancelled")
        if proc.returncode != 0:
            tail = (folder / "mafft.log").read_text(errors="replace")[-300:]
            raise RuntimeError(f"MAFFT failed ({proc.returncode}): {tail}")
        records = list(SeqIO.parse(folder / "mafft_out.fasta", "fasta"))
    ref_row = next((r for r in records if _base_acc(r.id) == _base_acc(reference.id)), None)
    if ref_row is None:
        raise RuntimeError(f"Reference {reference.id} missing from the MAFFT output")
    ref_bytes = _as_bytes(ref_row)
    ref_cols = ref_bytes != GAP
    ref_pos = np.cumsum(ref_cols) # reference position at each column
    aligned = [SeqRecord(Seq(ref_bytes[ref_cols].tobytes().decode()), id=reference.id, description=reference.description)]
    iso2ref: dict[str, np.ndarray] = {} # isolate position -> reference position
    dropped = 0
    for rec in records:
        if rec is ref_row:
            continue
        row = _as_bytes(rec)
        projected = row[ref_cols]
        if (projected != GAP).mean() < MIN_REF_COVERAGE:
            dropped += 1
            continue
        reverse = rec.id.startswith("_R_")
        acc = rec.id.removeprefix("_R_")
        mapping = np.maximum(ref_pos[row != GAP], 1) # bases inserted before the reference start -> 1
        iso2ref[_base_acc(acc)] = mapping[::-1] if reverse else mapping
        desc = rec.description.removeprefix("_R_")
        if reverse:
            desc += " [reverse-complemented to match the reference]"
        aligned.append(SeqRecord(Seq(projected.tobytes().decode()), id=acc, description=desc))
    return aligned, iso2ref, dropped


def _work_dir(taxid: int, name: str) -> Path:
    """
    tmp folder for one MAFFT run
    """
    work = seqs_dir(taxid) / ALIGN_DIR / f".work_{name}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    return work


def align_segment(taxid: int, ref: Reference, mafft_cmd: list[str] | None, cancel_event: threading.Event) -> list[str]:
    """
    whole-genome mode: align the downloaded genomes of reference segment onto it and write
      alignments/<reference>_alignment.fasta    reference first, then each genome, in reference coordinates
      alignments/<reference>_genes.tsv          the genomes own gene names placed on the reference
    """
    work = _work_dir(taxid, ref.accession)
    try:
        members = [r for r in SeqIO.parse(genomes_path(taxid), "fasta")
                   if not ref.segment or f"[segment={ref.segment}]" in r.description] \
            if genomes_path(taxid).exists() else []
        SeqIO.write(members, work / "genomes.fasta", "fasta")
        aligned, iso2ref, dropped = _align(work, next(SeqIO.parse(reference_path(taxid, ref), "fasta")),
                                           mafft_cmd, cancel_event)
        SeqIO.write(aligned, work / "aligned.fasta", "fasta")
        ann = annotations_path(taxid)
        with open(work / "genes.tsv", "w", encoding="utf-8", newline="\n") as out:
            out.write("accession\tgene\tref_begin\tref_end\n")
            for line in ann.read_text(encoding="utf-8").splitlines()[1:] if ann.exists() else []:
                acc, name, regions = line.split("\t")
                mapping = iso2ref.get(_base_acc(acc))
                for part in regions.split(",") if mapping is not None else []:
                    b, e = sorted(int(x) for x in part.split("-"))
                    if e <= mapping.size:
                        a1, a2 = sorted((int(mapping[b - 1]), int(mapping[e - 1])))
                        out.write(f"{acc}\t{name}\t{a1}\t{a2}\n")
        os.replace(work / "genes.tsv", isolate_genes_path(taxid, ref))
        os.replace(work / "aligned.fasta", genome_alignment_path(taxid, ref))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return [f"{ref.label}: dropped {dropped} genome(s) covering < {MIN_REF_COVERAGE:.0%} of the "
            "reference (too divergent to align)"] if dropped else []


def align_gene(taxid: int, gene_file: Path, ref: Reference, gene: GeneRow, mafft_cmd: list[str] | None,
               cancel_event: threading.Event) -> list[str]:
    """
    gene mode: align the isolates CDS in genes/<gene>.fasta onto the reference CDS of the same gene and write 
      alignments/<reference>/<gene>_alignment.fasta    reference CDS first, then isolates
    """
    out = gene_alignment_path(taxid, ref, gene_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = _work_dir(taxid, f"{ref.accession}_{gene_file.stem}")
    try:
        # reference CDS cut from the genome
        genome = str(next(SeqIO.parse(reference_path(taxid, ref), "fasta")).seq).upper()
        parts = [Seq(genome[min(b, e) - 1 : max(b, e)]) for b, e in gene.regions]
        cds = "".join(str(p.reverse_complement() if b > e else p) for p, (b, e) in zip(parts, gene.regions))
        reference = SeqRecord(Seq(cds), id=f"{ref.accession}:{gene.gene_name}", description=f"[gene={gene.gene_name}] reference CDS")
        isolates = [r for r in SeqIO.parse(gene_file, "fasta") if _base_acc(r.id) != _base_acc(ref.accession)]
        SeqIO.write(isolates, work / "genomes.fasta", "fasta")
        aligned, _, dropped = _align(work, reference, mafft_cmd, cancel_event)
        SeqIO.write(aligned, work / "aligned.fasta", "fasta")
        os.replace(work / "aligned.fasta", out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    notes = []
    if dropped:
        notes.append(f"{gene_file.stem}: dropped {dropped} sequence(s) covering < {MIN_REF_COVERAGE:.0%} of the reference CDS")
    copies = sum(g.gene_name.casefold() == gene.gene_name.casefold() for g in ref.genes)
    if copies > 1:
        notes.append(f"{gene.gene_name} is annotated {copies}× on {ref.accession}; the graph places it at the first copy")
    return notes


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """
    edge-normalised moving average
    """
    values = values.astype(float)
    if window <= 1 or values.size == 0:
        return values
    valid = ~np.isnan(values)
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    denominator = np.convolve(valid.astype(float), kernel, mode="same")
    out = np.full(values.shape, np.nan)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _rich_label() -> QLabel:
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


class RangeStore(QObject):
    """
    saved ranges as (reference accession, lo, hi)
    shared and showed on all graphs
    """
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.ranges: list[tuple[str, int, int]] = []

    def add(self, frame: str, a: float, b: float) -> None:
        lo, hi = sorted((int(round(a)), int(round(b))))
        if (frame, lo, hi) not in self.ranges:
            self.ranges = sorted(self.ranges + [(frame, lo, hi)])
            self.changed.emit()

    def remove(self, indices: list[int]) -> None:
        drop = set(indices)
        self.ranges = [r for i, r in enumerate(self.ranges) if i not in drop]
        self.changed.emit()

    def clear(self) -> None:
        self.ranges = []
        self.changed.emit()


class RangeViewBox(pg.ViewBox):
    """
    replaces default mouse mapping
    left-drag selects an x axis range saved on release
    left-click saves one position
    right-drag pans
    wheel zooms
    """

    def __init__(self, store: RangeStore) -> None:
        super().__init__()
        self._store = store
        self.frame: str | None = None # reference accession of the plotted coordinates
        self._preview = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(74, 163, 255, 60))
        self._preview.setZValue(10)
        self._preview.hide()
        self.addItem(self._preview, ignoreBounds=True)

    def _x(self, scene_pos) -> float:
        lo, hi = self.viewRange()[0]
        return min(max(self.mapSceneToView(scene_pos).x(), lo), hi)

    def mouseDragEvent(self, ev, axis=None) -> None:
        if axis is not None:
            return super().mouseDragEvent(ev, axis)
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            if self.frame is None:
                return
            a, b = self._x(ev.buttonDownScenePos()), self._x(ev.scenePos())
            self._preview.setRegion((a, b))
            self._preview.show()
            if ev.isFinish():
                self._preview.hide()
                self._store.add(self.frame, a, b)
        elif ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            d = self.mapSceneToView(ev.lastScenePos()) - self.mapSceneToView(ev.scenePos())
            self.translateBy(x=d.x(), y=d.y())
        else:
            super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mouseClickEvent(ev)
        ev.accept()
        if self.frame is not None:
            x = self._x(ev.scenePos())
            self._store.add(self.frame, x, x)


class RangePanel(QWidget):
    """
    saved range list for the side panel
    bands on the plot for ranges in the plotted frame
    """

    def __init__(self, store: RangeStore, plot: pg.PlotWidget) -> None:
        super().__init__()
        self._store = store
        self._plot = plot
        self._frame: str | None = None
        self._bands: list[pg.LinearRegionItem] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("Saved ranges:"))
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self.list)
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        v.addWidget(self.remove_btn)
        # delete on Windows/Linux, Mac sends backspace
        for key in (QKeySequence.StandardKey.Delete, QKeySequence(Qt.Key.Key_Backspace)):
            sc = QShortcut(key, self.list)
            sc.setContext(Qt.ShortcutContext.WidgetShortcut)
            sc.activated.connect(self._remove_selected)
        store.changed.connect(self._refresh)
        self._refresh()

    def set_frame(self, frame: str | None) -> None:
        self._frame = frame
        self._plot.getViewBox().frame = frame
        self._refresh()

    def detach(self) -> None:
        self._store.changed.disconnect(self._refresh)

    def _remove_selected(self) -> None:
        rows = [self.list.row(item) for item in self.list.selectedItems()]
        if rows:
            self._store.remove(rows)

    def _refresh(self) -> None:
        self.list.clear()
        for band in self._bands:
            self._plot.removeItem(band)
        self._bands = []
        for frame, lo, hi in self._store.ranges:
            self.list.addItem(f"{frame}: {lo:,} – {hi:,}  ({hi - lo + 1:,} bp)")
            if frame == self._frame:
                band = pg.LinearRegionItem((lo, hi), movable=False, brush=pg.mkBrush(255, 200, 0, 45))
                self._plot.addItem(band, ignoreBounds=True)
                self._bands.append(band)
        self.remove_btn.setEnabled(bool(self._store.ranges))


def _plot_widget(store: RangeStore, left_label: str, bottom_label: str) -> pg.PlotWidget:
    plot = pg.PlotWidget(viewBox=RangeViewBox(store))
    plot.setBackground(None)
    plot.setLabel("bottom", bottom_label)
    plot.setLabel("left", left_label)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.getPlotItem().setContentsMargins(6, 6, 16, 12)
    plot.setClipToView(True)
    plot.setDownsampling(auto=True, mode="peak")
    return plot


class CoverageTab(QWidget):
    """
    Smoothed coverage of the downloaded CDS regions with a hover readout
    the ticked genes (genes/*.fasta) and every gene of the downloaded whole genomes (annotations.tsv)
    """
    def __init__(self, taxid: int, store: RangeStore, frame: str | None) -> None:
        super().__init__()
        self._store = store
        self._frame = frame
        # (accession, gene, regions text) -> isolate
        genes: dict[tuple[str, str, str], str] = {}
        for gf in gene_files(taxid):
            with open(gf, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = re.match(r">(\S+?):(\S+)\s+\[gene=([^\]]*)\]\s+\[isolate=([^\]]*)\]", line)
                    if m:
                        genes.setdefault((m.group(1), m.group(3), m.group(2)), m.group(4))
        ann = annotations_path(taxid)
        for line in ann.read_text(encoding="utf-8").splitlines()[1:] if ann.exists() else []:
            genes.setdefault(tuple(line.split("\t")), "")
        self._n_genes = len(genes)
        self._n_genomes = len({acc for acc, _, _ in genes})

        # every region as parallel arrays (begin, end, gene-name code)
        names: dict[str, int] = {}
        begins: list[int] = []
        ends: list[int] = []
        codes: list[int] = []
        for acc, name, regions in genes:
            code = names.setdefault(f"(unnamed) {acc}" if is_unnamed(name) else name, len(names))
            for part in regions.split(","):
                lo, hi = sorted(int(x) for x in part.split("-"))
                begins.append(lo)
                ends.append(hi)
                codes.append(code)
        self._names = list(names)
        self._begins = np.asarray(begins, dtype=np.int64)
        self._ends = np.asarray(ends, dtype=np.int64)
        self._codes = np.asarray(codes, dtype=np.int64)
        self._ymax = 1.0
        self._lo = int(self._begins.min()) if begins else 0
        self._hi = int(self._ends.max()) if begins else 0
        # coverage via a difference array + cumulative sum
        diff = np.zeros(self._hi + 2, dtype=np.int32)
        np.add.at(diff, self._begins, 1)
        np.add.at(diff, self._ends + 1, -1)
        self._cov = np.cumsum(diff)[: self._hi + 1]
        self._build_ui()
        self._replot()

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
        controls.addWidget(QLabel(f"{self._n_genes} downloaded genes · {self._n_genomes} genome(s)"))
        outer.addLayout(controls)

        body = QHBoxLayout()
        pg.setConfigOptions(antialias=False)
        self.plot = _plot_widget(self._store, "Coverage depth (genomes)", "Genome position (bp)")
        if self._hi > self._lo:
            self._ymax = float(self._cov.max()) + 1.0
            self.plot.setLimits(xMin=self._lo, xMax=self._hi, yMin=0, yMax=self._ymax,
                                minXRange=min(10, self._hi - self._lo), minYRange=min(1.0, self._ymax))
            self._reset_view()
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#4aa3ff", width=2))
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888"))
        self.vline.setVisible(False)
        self.plot.addItem(self.vline, ignoreBounds=True)
        body.addWidget(self.plot, 1)

        panel = QWidget()
        panel_v = QVBoxLayout(panel)
        panel_v.setContentsMargins(0, 0, 0, 0)
        self.readout = QLabel("Hover over the curve to list the genes covering that position.\n\n" + MOUSE_HELP
                              if self._n_genes else "Nothing downloaded yet.")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.readout.setWordWrap(True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.readout)
        panel_v.addWidget(scroll, 3)
        self.ranges = RangePanel(self._store, self.plot)
        self.ranges.set_frame(self._frame)
        if self._frame is None:
            note = QLabel("Range saving is off here: positions mix segments or there is "
                          "no reference. Use a conservation graph.")
            note.setWordWrap(True)
            panel_v.addWidget(note)
        panel_v.addWidget(self.ranges, 2)
        panel.setFixedWidth(260)
        body.addWidget(panel)
        outer.addLayout(body)

        self._proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved)

    def close_down(self) -> None:
        self.ranges.detach()

    def _reset_view(self) -> None:
        if self._hi > self._lo:
            self.plot.setXRange(self._lo, self._hi, padding=0)
            self.plot.setYRange(0, self._ymax, padding=0)

    def _replot(self) -> None:
        smoothed = _smooth(self._cov, self.window_spin.value())
        if smoothed.size <= self._lo:
            self.curve.setData([], [])
            return
        self.curve.setData(np.arange(self._lo, smoothed.size), smoothed[self._lo:])

    def _on_mouse_moved(self, event) -> None:
        scene_pos = event[0]
        if not self._n_genes or not self.plot.sceneBoundingRect().contains(scene_pos):
            return
        pos = int(round(self.plot.getViewBox().mapSceneToView(scene_pos).x()))
        pos = max(self._lo, min(pos, self._hi))
        self.vline.setPos(pos)
        self.vline.setVisible(True)
        depth = int(self._cov[pos]) if pos < self._cov.size else 0
        lines = [f"Position {pos:,} bp", f"Coverage depth: {depth}", ""]
        mask = (self._begins <= pos) & (self._ends >= pos)
        if not mask.any():
            lines.append("No genes cover this position.")
        else:
            counts = np.bincount(self._codes[mask], minlength=len(self._names))
            lines.append("Genes here:")
            lines.extend(f"  • {self._names[i]}" + (f" ×{counts[i]}" if counts[i] > 1 else "")
                         for i in np.argsort(counts)[::-1] if counts[i] > 0)
        self.readout.setText("\n".join(lines))



def _score(counts: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """
    occupancy * (1 - H/2) from A/C/G/T counts at position and the number of sequences observed there
    """
    n_acgt = counts.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / n_acgt
        entropy = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=0)
        score = (n_acgt / observed) * (1.0 - entropy / 2.0)
    score[n_acgt == 0] = 0.0
    score[observed == 0] = np.nan
    return score


def _genome_scores(path: Path) -> tuple[np.ndarray, int]:
    """
    whole-genome mode: scores of every column of <reference>_alignment.fasta
    """
    counts = None
    n = 0
    for rec in SeqIO.parse(path, "fasta"):
        row = _as_bytes(rec)
        if counts is None:
            counts = np.zeros((4, row.size), dtype=np.int32)
        for i, base in enumerate(b"ACGT"):
            counts[i] += row == base
        n += 1
    if counts is None:
        return np.zeros(0), 0
    return _score(counts, np.full(counts.shape[1], n, dtype=float)), n


def cds_genome_positions(regions: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """
    for each position of a reference CDS return its 1-based genome position, and which strand
    """
    pos = [np.arange(b, e + 1) if b <= e else np.arange(b, e - 1, -1) for b, e in regions]
    minus = [np.full(abs(e - b) + 1, b > e) for b, e in regions]
    return np.concatenate(pos), np.concatenate(minus)


def _gene_scores(length: int, entries: list[tuple[GeneRow, Path]]) -> tuple[np.ndarray, int]:
    """
    gene mode: one value per reference genome position from gene alignments
    each gene is mapped to genome positions
    where genes overlap every genome counts once
    """
    cover = np.zeros(length + 1, dtype=np.int32)
    for gene, _ in entries:
        np.add.at(cover, cds_genome_positions(gene.regions)[0], 1)
    overlap = np.flatnonzero(cover > 1)
    counts = np.zeros((4, length + 1), dtype=np.int64)
    observed = np.zeros(length + 1, dtype=np.int64)
    overlap_rows: dict[str, np.ndarray] = {}
    genomes: set[str] = set()
    for gene, path in entries:
        pos, minus = cds_genome_positions(gene.regions)
        rows: dict[str, np.ndarray] = {}
        for rec in SeqIO.parse(path, "fasta"):
            key = _base_acc(rec.id)
            if key in rows:
                continue   # one row per genome
            row = _as_bytes(rec).copy()
            row[minus] = np.frombuffer(row[minus].tobytes().translate(COMPLEMENT), dtype=np.uint8)
            rows[key] = row
        if not rows:
            continue
        genomes |= rows.keys()
        mat = np.stack(list(rows.values()))
        single = cover[pos] == 1
        p1 = pos[single]
        sub = mat[:, single]
        for i, base in enumerate(b"ACGT"):
            counts[i, p1] += (sub == base).sum(axis=0)
        observed[p1] += mat.shape[0]
        if (~single).any():
            idx = np.searchsorted(overlap, pos[~single])
            vals = mat[:, ~single]
            for k, key in enumerate(rows):
                arr = overlap_rows.setdefault(key, np.zeros(overlap.size, dtype=np.uint8))
                empty = arr[idx] == 0
                arr[idx[empty]] = vals[k, empty]
    if overlap_rows:
        m = np.stack(list(overlap_rows.values()))
        observed[overlap] += (m != 0).sum(axis=0)
        for i, base in enumerate(b"ACGT"):
            counts[i, overlap] += (m == base).sum(axis=0)
    return _score(counts, observed)[1:], len(genomes)


@dataclass
class Result:
    scores: np.ndarray            # per reference position
    n_seqs: int                   # sequences / genomes that went in
    notes: list[str]
    isolate_names: tuple | None   # whole-genome mode: (names, name code, accession code, begin, end) arrays
    genes: list[GeneRow] | None   # gene mode: the reference genes that were aligned
    spellings: dict | None        # gene mode: reference gene name -> Counter of isolate spellings
    aligned: dict | None = None   # gene mode: reference gene name -> genomes in its alignment


class ConservationWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(str, object) # reference accession
    failed = pyqtSignal(str)

    @pyqtSlot(object)
    def run(self, job: dict) -> None:
        try:
            self.done.emit(job["reference"].accession, self._genome(job) if job["mode"] == "genome" else self._genes(job))
        except InterruptedError:
            self.failed.emit("Cancelled")
        except Exception as e:
            self.failed.emit(str(e)[:300])

    def _genome(self, job: dict) -> Result:
        taxid, ref = job["taxid"], job["reference"]
        notes = []
        if not genome_alignment_path(taxid, ref).exists():
            self.progress.emit(f"Aligning {genomes_for(taxid, ref)} genome(s) to {ref.accession} with MAFFT")
            notes = align_segment(taxid, ref, job["mafft"], job["cancel"])
        if job["cancel"].is_set():
            raise InterruptedError("cancelled")
        self.progress.emit(f"Scoring {ref.accession}")
        raw, n = _genome_scores(genome_alignment_path(taxid, ref))
        # the genomes own gene names placed on the reference, as arrays for lookup
        names: dict[str, int] = {}
        accs: dict[str, int] = {}
        rows = []
        tsv = isolate_genes_path(taxid, ref)
        for line in tsv.read_text(encoding="utf-8").splitlines()[1:] if tsv.exists() else []:
            acc, name, b, e = line.split("\t")
            rows.append((names.setdefault(name, len(names)), accs.setdefault(acc, len(accs)), int(b), int(e)))
        arr = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
        return Result(raw, n, notes, (list(names), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]), None, None)

    def _genes(self, job: dict) -> Result:
        taxid, ref = job["taxid"], job["reference"]
        notes: list[str] = []
        entries = []
        for gf, gene in job["genes"]:
            path = gene_alignment_path(taxid, ref, gf)
            if not path.exists():
                self.progress.emit(f"Aligning {gf.stem} ({count_records(gf)} sequences) with MAFFT")
                notes += align_gene(taxid, gf, ref, gene, job["mafft"], job["cancel"])
            if job["cancel"].is_set():
                raise InterruptedError("cancelled")
            entries.append((gene, path, gf))
        self.progress.emit(f"Scoring {ref.accession}")
        length = len(next(SeqIO.parse(reference_path(taxid, ref), "fasta")).seq)
        scores, n = _gene_scores(length, [(gene, path) for gene, path, _ in entries])
        return Result(scores, n, notes, None, [gene for gene, _, _ in entries],
                      {gene.gene_name: _spellings(gf) for gene, _, gf in entries},
                      {gene.gene_name: len({_base_acc(i) for i in fasta_ids(path)} - {_base_acc(ref.accession)})
                       for gene, path, _ in entries})


class ConservationTab(QWidget):
    """
    two modes - genome and gene
    reference box picks which RefSeq record to align to
    nothing is aligned until "Align and generate graph"
    finished alignments are reused
    """
    request = pyqtSignal(object)

    def __init__(self, mode: str, taxid: int, references: list[Reference], store: RangeStore,
                 mafft_cmd: list[str] | None) -> None:
        super().__init__()
        self._mode = mode
        self._taxid = taxid
        self._references = references
        self._store = store
        self._mafft = mafft_cmd
        self._results: dict[str, Result] = {}
        self._cancel = threading.Event()
        self._started = 0.0
        self._phase = ""
        self._smoothed = np.zeros(0)
        self._length = 1
        self._genes_here: list[tuple[str, str, list[tuple[int, int]]]] = [] # name, colour, regions
        self._gene_items: list[tuple[pg.PlotDataItem, int, int]] = []
        self._build_ui()
        self._thread = QThread(self)
        self._worker = ConservationWorker()
        self._worker.moveToThread(self._thread)
        self.request.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        # spellings per gene file
        self._file_names = {gf: {n.casefold() for n in _spellings(gf)} | {gf.stem.casefold()}
                            for gf in gene_files(taxid)}
        self.reference_combo.blockSignals(True)
        for ref in references:
            self.reference_combo.addItem(ref.label, ref.accession)
        # start on the first reference that has something to align in this mode
        first = next((i for i, ref in enumerate(references)
                      if (genomes_for(taxid, ref) if mode == "genome" else self._placed(ref))), 0)
        self.reference_combo.setCurrentIndex(first)
        self.reference_combo.blockSignals(False)
        self._describe_inputs()

    def _placed(self, ref: Reference) -> list[tuple[Path, GeneRow]]:
        """
        downloaded gene files whose gene is annotated on ref
        """
        out = []
        for gf, names in self._file_names.items():
            gene = next((g for g in ref.genes if not is_unnamed(g.gene_name) and g.gene_name.casefold() in names), None)
            if gene is not None:
                out.append((gf, gene))
        return out

    def _describe_inputs(self) -> None:
        """
        what would be aligned to the chosen reference, shown before anything runs
        """
        ref = self._ref
        lines = []
        self._inputs = []
        if ref is None:
            lines.append("No RefSeq reference for this taxon, so there is nothing to align to.")
        elif not reference_path(self._taxid, ref).exists():
            lines.append("This reference genome isn't downloaded yet: it comes with your next download.")
        elif self._mode == "genome":
            n = genomes_for(self._taxid, ref)
            self._inputs = [ref] if n else []
            if not n:
                what = f"segment {ref.segment}" if ref.segment else "this reference"
                lines.append(f"No whole genomes downloaded for {what}. Go back and tick the <b>Genome</b> "
                             "box on the genomes you want.")
        else:
            placed = self._placed(ref)
            unticked = {self.gene_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.gene_list.count())
                        if self.gene_list.item(i).checkState() != Qt.CheckState.Checked}
            self._inputs = placed
            self.gene_list.blockSignals(True)
            self.gene_list.clear()
            for gf, gene in placed:
                n = len({_base_acc(i) for i in fasta_ids(gf)} - {_base_acc(ref.accession)})
                lo, hi = min(min(b, e) for b, e in gene.regions), max(max(b, e) for b, e in gene.regions)
                state = "aligned" if gene_alignment_path(self._taxid, ref, gf).exists() else "to align"
                item = QListWidgetItem(f"{gene.gene_name}   {lo:,}–{hi:,} · {n} genome{'s' if n != 1 else ''} {state}")
                item.setData(Qt.ItemDataRole.UserRole, gf.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked if gf.name in unticked else Qt.CheckState.Checked)
                self.gene_list.addItem(item)
            self.gene_list.blockSignals(False)
            self._update_genes_button()
            if not placed:
                lines.append("None of the downloaded genes is annotated on this reference.")
        if self._mafft is None:
            lines.append("<b>MAFFT not found</b>: put it on PATH or next to run.py (see README) to align.")
        self.legend.setText("<br>".join(lines))
        self.align_btn.setEnabled(bool(self._inputs))
        self.status.setText("Not generated yet")

    def _update_genes_button(self) -> None:
        n = self.gene_list.count()
        ticked = sum(self.gene_list.item(i).checkState() == Qt.CheckState.Checked for i in range(n))
        self.genes_btn.setText(f"Genes: {ticked} of {n} ▾")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Reference:"))
        self.reference_combo = QComboBox()
        self.reference_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.reference_combo.currentIndexChanged.connect(self._on_reference_changed)
        controls.addWidget(self.reference_combo)
        self.gene_list = QListWidget()
        if self._mode == "genes":
            # dropdown that stays open while genes are ticked
            self.genes_btn = QToolButton()
            self.genes_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = QMenu(self.genes_btn)
            holder = QWidgetAction(menu)
            self.gene_list.setMinimumWidth(380)
            self.gene_list.setMinimumHeight(220)
            holder.setDefaultWidget(self.gene_list)
            menu.addAction(holder)
            self.genes_btn.setMenu(menu)
            self.gene_list.itemChanged.connect(self._update_genes_button)
            controls.addWidget(self.genes_btn)
        self.align_btn = QPushButton("Align and generate graph")
        self.align_btn.clicked.connect(self._start)
        controls.addWidget(self.align_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(lambda: self._cancel.set())
        controls.addWidget(self.cancel_btn)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Smoothing window (bp):"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(MIN_WINDOW, MAX_WINDOW)
        self.window_spin.setSingleStep(5)
        self.window_spin.setValue(DEFAULT_WINDOW)
        self.window_spin.valueChanged.connect(self._draw)
        controls.addWidget(self.window_spin)
        self.reset_btn = QPushButton("Reset view")
        self.reset_btn.clicked.connect(self._reset_view)
        controls.addWidget(self.reset_btn)
        controls.addStretch(1)
        self.status = QLabel("")
        controls.addWidget(self.status)
        outer.addLayout(controls)

        body = QHBoxLayout()
        pg.setConfigOptions(antialias=True)
        self.plot = _plot_widget(self._store, "Conservation (1 = identical)", "Reference position (bp)")
        self.plot.setLimits(yMin=-0.02, yMax=1.05, minYRange=0.1)
        self.curve = self.plot.plot([], [], pen=pg.mkPen(INTERGENIC, width=2))
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#888"))
        self.vline.setVisible(False)
        self.plot.addItem(self.vline, ignoreBounds=True)
        body.addWidget(self.plot, 1)

        panel = QWidget()
        panel_v = QVBoxLayout(panel)
        panel_v.setContentsMargins(0, 0, 0, 0)
        self.readout = _rich_label()
        self.legend = _rich_label()
        content = QWidget()
        content_v = QVBoxLayout(content)
        content_v.addWidget(self.readout)
        content_v.addWidget(self.legend, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        panel_v.addWidget(scroll, 3)
        self.ranges = RangePanel(self._store, self.plot)
        panel_v.addWidget(self.ranges, 2)
        panel.setFixedWidth(280)
        body.addWidget(panel)
        outer.addLayout(body)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved)

    def _start(self) -> None:
        job = {"mode": self._mode, "taxid": self._taxid, "mafft": self._mafft, "reference": self._ref,
               "cancel": threading.Event()} # fresh per run
        if self._mode == "genes":
            job["genes"] = [entry for i, entry in enumerate(self._inputs)
                            if self.gene_list.item(i).checkState() == Qt.CheckState.Checked]
            if not job["genes"]:
                self.status.setText("Tick at least one gene")
                return
        self._cancel = job["cancel"]
        self.align_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self._phase = "Starting"
        self._started = time.monotonic()
        self._timer.start()
        self._tick()
        self.request.emit(job)

    def close_down(self) -> None:
        """
        kill any running MAFFT, then stop the thread
        """
        self._cancel.set()
        self._thread.quit()
        self._thread.wait()
        self.ranges.detach()

    def _tick(self) -> None:
        secs = int(time.monotonic() - self._started)
        self.status.setText(f"{self._phase}… {secs // 60}:{secs % 60:02d}")

    def _finish(self) -> None:
        self._timer.stop()
        self.align_btn.setEnabled(bool(self._inputs))
        self.cancel_btn.setVisible(False)

    def _on_progress(self, message: str) -> None:
        self._phase = message
        self._tick()

    def _on_failed(self, message: str) -> None:
        self._finish()
        self.status.setText(message)

    def _on_done(self, accession: str, result: Result) -> None:
        self._finish()
        self._results[accession] = result
        self._describe_inputs() # refresh "to align" / "alignment ready"
        self._show()

    @property
    def _ref(self) -> Reference | None:
        acc = self.reference_combo.currentData()
        return next((r for r in self._references if r.accession == acc), None)

    def _on_reference_changed(self) -> None:
        if self.cancel_btn.isHidden():
            self._describe_inputs()
        self._show()

    def _show(self) -> None:
        """
        draw the chosen reference result, or clear the plot if it hasn't been generated
        """
        ref = self._ref
        if ref is None:
            return
        self.ranges.set_frame(ref.accession)
        res = self._results.get(ref.accession)
        if res is None:
            self.curve.setData([], [])
            for item, _, _ in self._gene_items:
                item.setData([], [])
            return
        self._length = max(res.scores.size, 1)
        self.plot.setLimits(xMin=1, xMax=self._length, minXRange=min(10, self._length))
        # colours follow the reference's genes in genome order
        named = sorted((min(min(b, e) for b, e in g.regions), g.gene_name, g.regions)
                       for g in ref.genes if not is_unnamed(g.gene_name))
        colours = {name: PALETTE[i % len(PALETTE)] for i, (_, name, _) in enumerate(named)}
        shown = named if res.genes is None else sorted(
            (min(min(b, e) for b, e in g.regions), g.gene_name, g.regions) for g in res.genes)
        self._genes_here = [(name, colours.get(name, INTERGENIC), regions) for _, name, regions in shown]
        for item, _, _ in self._gene_items:
            self.plot.removeItem(item)
        self._gene_items = []
        for name, colour, regions in self._genes_here:
            length = sum(abs(e - b) + 1 for b, e in regions)
            for b, e in regions:
                item = self.plot.plot([], [], pen=pg.mkPen(colour, width=2), connect="finite")
                item.setZValue(1 + 1 / length) # shorter gene on top in case of full overlap
                self._gene_items.append((item, min(b, e), max(b, e)))
        self.legend.setText("<br>".join(["<b>Notes:</b>"] + [html.escape(n) for n in res.notes]) if res.notes else "")
        self._draw()
        self._reset_view()
        n = res.n_seqs - 1
        self.status.setText(f"{ref.accession}: needs ≥{MIN_SEQS} sequences (reference + isolates)" if res.n_seqs < MIN_SEQS
                            else f"{ref.accession}: {n} genome{'s' if n != 1 else ''} + reference")

    def _draw(self) -> None:
        res = self._results.get(self._ref.accession) if self._ref else None
        if res is None:
            return
        self._smoothed = _smooth(res.scores, self.window_spin.value())
        x = np.arange(1, self._smoothed.size + 1)
        if res.genes is None:
            self.curve.setData(x, self._smoothed, connect="finite")
        else:
            self.curve.setData([], [])
        for item, lo, hi in self._gene_items:
            item.setData(x[lo - 1 : hi], self._smoothed[lo - 1 : hi], connect="finite")

    def _reset_view(self) -> None:
        self.plot.setXRange(1, self._length, padding=0.01)
        self.plot.setYRange(0, 1.05, padding=0)

    def _on_mouse_moved(self, event) -> None:
        scene_pos = event[0]
        res = self._results.get(self._ref.accession) if self._ref else None
        if res is None or not self.plot.sceneBoundingRect().contains(scene_pos):
            return
        x = self.plot.getViewBox().mapSceneToView(scene_pos).x()
        pos = int(round(x))
        self.vline.setPos(x)
        self.vline.setVisible(True)
        lines = [f"<b>Position {pos:,} bp</b>"]
        if 1 <= pos <= self._smoothed.size:
            value = self._smoothed[pos - 1]
            lines.append(f"Conservation: {value:.2f}" if np.isfinite(value) else "No data here (outside the graphed genes)")
        here = [(n, c, r) for n, c, r in self._genes_here if any(min(b, e) <= pos <= max(b, e) for b, e in r)]
        here_names = {n for n, _, _ in here}
        lines.append("")
        lines.append("<b>Reference genes/ORFs here:</b>" if here else "No reference gene here.")
        aligned = res.aligned or {}
        key = "".join(
            f'<tr><td width="18" bgcolor="{c}">&nbsp;</td><td>&nbsp;{html.escape(n)} <small>'
            f'{min(min(b, e) for b, e in r):,}–{max(max(b, e) for b, e in r):,}'
            + (f" · {aligned[n]} gene{'s' if aligned[n] != 1 else ''} + reference" if n in aligned else "")
            + "</small></td></tr>"
            for n, c, r in here)
        key = f'<table cellspacing="2" cellpadding="0">{key}</table>' if key else ""
        others: Counter = Counter()
        if res.spellings is not None:
            for name in here_names:
                others.update(res.spellings.get(name, Counter()))
        else:
            names, codes, accs, begins, ends = res.isolate_names
            mask = (begins <= pos) & (ends >= pos)
            pairs = np.unique(np.stack([codes[mask], accs[mask]]), axis=1)
            others = Counter({names[i]: int(c) for i, c in enumerate(np.bincount(pairs[0], minlength=len(names))) if c})
        others_list = sorted(((n, c) for n, c in others.items() if n not in here_names), key=lambda t: -t[1])
        tail = []
        if others_list:
            tail = ["<b>Other gene names here:</b>"]
            tail.extend(f"{html.escape(n)} <small>({c} genome{'s' if c != 1 else ''})</small>" for n, c in others_list)
        self.readout.setText("<br>".join(lines) + key + "<br>".join(tail))


class GraphsPage(QWidget):
    """
    three graphs as tabs
    load() rebuilds them from what's on disk
    close_down() stops any running alignment
    """
    back_requested = pyqtSignal()

    def __init__(self, store: RangeStore) -> None:
        super().__init__()
        self._store = store
        self._tabs: list = []
        outer = QVBoxLayout(self)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        bottom = QHBoxLayout()
        back = QPushButton("← Back to download")
        back.clicked.connect(self.back_requested)
        bottom.addWidget(back)
        bottom.addStretch(1)
        self.next_btn = QPushButton("Next →")
        self.next_btn.setEnabled(False)
        self.next_btn.setToolTip("The next stage isn't built yet")
        bottom.addWidget(self.next_btn)
        outer.addLayout(bottom)

    def load(self, taxid: int, references: list[Reference], title: str) -> None:
        self.close_down()
        mafft = locate_mafft()
        frame = references[0].accession if len(references) == 1 else None
        self._tabs = [
            CoverageTab(taxid, self._store, frame),
            ConservationTab("genome", taxid, references, self._store, mafft),
            ConservationTab("genes", taxid, references, self._store, mafft),
        ]
        for tab, label in zip(self._tabs, ["Coverage", "Conservation · whole genome", "Conservation · genes"]):
            self.tabs.addTab(tab, label)
        self.summary.setText(f"<b>{html.escape(title)}</b> — downloaded: {len(gene_files(taxid))} gene file(s), "
                             f"{count_records(genomes_path(taxid))} whole genome(s)")

    def close_down(self) -> None:
        for tab in self._tabs:
            tab.close_down()
        self.tabs.clear()
        for tab in self._tabs:
            tab.deleteLater()
        self._tabs = []
