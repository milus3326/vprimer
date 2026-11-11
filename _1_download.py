"""
Stage 1: search NCBI and download
"""

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from dotenv import load_dotenv, set_key


DATASETS_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2"
ENV_PATH = Path(__file__).parent / ".env"
FETCH_BATCH = 100              # records per efetch / dataset-report request
GENES_DIR = "genes"            # seqs/<taxid>/genes/<gene>.fasta (CDS of the ticked gene rows)
GENOMES_DIR = "genomes"        # seqs/<taxid>/genomes/genomes.fasta + <reference>.fasta + annotations.tsv
ALIGN_DIR = "alignments"       # seqs/<taxid>/alignments/ (written by stage 2, invalidated here)


def _log(msg: str) -> None:
    print(f"[log] {msg}", flush=True)


@dataclass
class GeneRow:
    gene_name: str
    genome_accession: str
    isolate_name: str
    regions: list[tuple[int, int]]
    cds_title: str


@dataclass
class Reference:
    accession: str
    segment: str
    isolate: str
    genes: list[GeneRow]

    @property
    def label(self) -> str:
        return self.accession + (f" · segment {self.segment}" if self.segment else "") + (f" · {self.isolate}" if self.isolate else "")


def configure_entrez(email: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    """
    single entry point for NCBI credentials
    called with no arguments (startup): load EMAIL / NCBI_API_KEY from .env
    called with arguments (the user typed them in the GUI): overwrite .env
    returns the (email, api_key)
    """
    if email is None and api_key is None:
        load_dotenv(ENV_PATH)
        email = os.getenv("EMAIL", "").strip()
        api_key = os.getenv("NCBI_API_KEY", "").strip()
    else:
        email = (email or "").strip()
        api_key = (api_key or "").strip()
        ENV_PATH.touch(exist_ok=True)
        set_key(str(ENV_PATH), "EMAIL", email, quote_mode="auto")
        set_key(str(ENV_PATH), "NCBI_API_KEY", api_key, quote_mode="auto")
    Entrez.email = email
    Entrez.api_key = api_key or None
    return email, api_key


def is_unnamed(gene_name: str) -> bool:
    name = (gene_name or "").strip()
    return not name or name.startswith("CDS:")


def _base_acc(acc: str) -> str:
    return acc.split(".", 1)[0]


def resolve_taxon(query: str) -> tuple[int, str]:
    """
    taxid or taxon name -> (taxid, scientific name)
    """
    q = query.strip()
    if q.isdigit():
        taxid = int(q)
    else:
        with Entrez.esearch(db="taxonomy", term=q) as handle:
            ids = Entrez.read(handle).get("IdList") or []
        if not ids:
            raise LookupError(f"No taxon matches '{q}'")
        taxid = int(ids[0])
    with Entrez.esummary(db="taxonomy", id=str(taxid)) as handle:
        summaries = Entrez.read(handle)
    if not summaries:
        raise LookupError(f"No taxon with taxid {taxid}")
    name = str(summaries[0].get("ScientificName") or summaries[0].get("CommonName") or taxid)
    _log(f"resolve_taxon: '{query}' -> {name} (taxid {taxid})")
    return taxid, name


def _paged_reports(url: str, params: dict[str, str], api_key: str):
    """
    yield each page's JSON from a paginated NCBI Datasets v2 report endpoint
    """
    headers = {"accept": "application/json", **({"api-key": api_key} if api_key else {})}
    params = dict(params)
    page = 0
    with httpx.Client(timeout=60.0) as client:
        while True:
            page += 1
            _log(f"page {page}: GET {url} params={params}")
            start = time.monotonic()
            try:
                response = client.get(url, params=params, headers=headers)
            except httpx.TimeoutException as e:
                _log(f"page {page}: TIMED OUT after {time.monotonic() - start:.1f}s ({type(e).__name__})")
                raise RuntimeError("NCBI no response") from e
            _log(f"page {page}: HTTP {response.status_code} in {time.monotonic() - start:.2f}s "
                 f"({len(response.content)} bytes)")
            body = response.text[:300]
            if api_key:
                body = body.replace(api_key, "***")
            if response.status_code >= 500:
                _log(f"page {page}: server error body: {body}")
                if "forwarding request" in body.lower():
                    raise RuntimeError(
                        "NCBI couldn't compute this search in time (HTTP 500 after ~20s). "
                        "Huge taxa (e.g. influenza A) always fail with 'Complete only': "
                        "uncheck it and narrow the search with 'Released since' or 'Location'."
                    )
                raise RuntimeError(f"NCBI returned a server error (HTTP {response.status_code}). "
                                   "This may be transient — click Search again.")
            if response.status_code >= 400:
                _log(f"page {page}: client error body: {body}")
                raise RuntimeError(f"HTTP {response.status_code}: {body}")
            data = response.json()
            yield data
            if not data.get("next_page_token"):
                return
            params["page_token"] = data["next_page_token"]


def list_genes(taxid: int, refseq_only: bool, api_key: str = "", *, geo_location: str = "",
    annotated_only: bool = False, complete_only: bool = False, released_since: str = "",
    progress: Callable[[int], None] | None = None,) -> list[GeneRow]:
    """
    list annotated genes for a taxon by NCBI annotation report
    """
    url = f"{DATASETS_BASE}/virus/taxon/{taxid}/annotation_report"
    params: dict[str, str] = {"page_size": "500"}
    if refseq_only:
        params["filter.refseq_only"] = "true"
    else:
        if geo_location.strip():
            params["filter.geo_location"] = geo_location.strip()
        if annotated_only:
            params["filter.annotated_only"] = "true"
        if complete_only:
            params["filter.complete_only"] = "true"
        if released_since.strip():
            params["filter.released_since"] = released_since.strip()
    _log(f"list_genes start: taxid={taxid} params={params}")

    rows: list[GeneRow] = []
    for data in _paged_reports(url, params, api_key):
        for report in data.get("reports") or []:
            for gene in report.get("genes") or []:
                nucleotide = ((gene.get("cds") or [{}])[0]).get("nucleotide") or {}
                regions = [(int(r["begin"]), int(r["end"])) for r in nucleotide.get("range") or []
                           if str(r.get("begin", "")).isdigit() and str(r.get("end", "")).isdigit()]
                if not regions:
                    continue
                rows.append(GeneRow((gene.get("name") or "").strip(), report.get("accession") or "",
                                    report.get("isolate_name") or "", regions, nucleotide.get("title") or ""))
        _log(f"list_genes: total {len(rows)} rows")
        if progress is not None:
            progress(len(rows))
    return rows


def _parse_location(loc: str) -> list[tuple[int, int]]:
    """
    GenBank location -> regions in transcription order
    """
    outer = loc.startswith("complement(")
    parts = re.findall(r"(complement\()?<?(\d+)\.\.>?(\d+)", loc)
    regions = [(int(b), int(a)) if (outer or c) else (int(a), int(b)) for c, a, b in parts]
    return regions[::-1] if outer else regions


def _span_key(regions: list[tuple[int, int]]) -> tuple:
    return tuple(sorted((min(b, e), max(b, e)) for b, e in regions))


def _efetch(accessions: list[str], rettype: str, cancel_event: threading.Event | None = None,
            on_batch: Callable[[], None] | None = None):
    """
    yield the records of `accessions`
    """
    for i in range(0, len(accessions), FETCH_BATCH):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("cancelled")
        ids = ",".join(accessions[i : i + FETCH_BATCH])
        with Entrez.efetch(db="nuccore", id=ids, rettype=rettype, retmode="text") as handle:
            yield from SeqIO.parse(handle, "fasta")
        if on_batch is not None:
            on_batch()


def _cds_location(rec: SeqRecord) -> tuple[str, list[tuple[int, int]]] | None:
    """
    fasta_cds_na record -> from its [location=..] tag
    """
    m = re.search(r"\[location=([^\]]+)\]", rec.description)
    if m is None:
        return None
    return _base_acc(re.sub(r"^lcl\|", "", rec.id).split("_cds_")[0]), _parse_location(m.group(1))


def list_references(taxid: int, api_key: str = "") -> list[Reference]:
    """
    RefSeq references for a taxon: segment + isolate from the dataset report, gene coordinates
    from the annotation report, re-oriented by the fasta_cds_na location tags
    """
    meta: dict[str, tuple[str, str, str]] = {}   # accession -> organism, segment, isolate
    url = f"{DATASETS_BASE}/virus/taxon/{taxid}/dataset_report"
    for data in _paged_reports(url, {"filter.refseq_only": "true", "page_size": "500"}, api_key):
        for rep in data.get("reports") or []:
            acc = rep.get("accession") or ""
            meta[acc] = ((rep.get("virus") or {}).get("organism_name") or acc, str(rep.get("segment") or ""),
                         (rep.get("isolate") or {}).get("name") or "")
    oriented: dict[tuple, list[tuple[int, int]]] = {}
    for rec in _efetch(list(meta), "fasta_cds_na"):
        if loc := _cds_location(rec):
            oriented[(loc[0], _span_key(loc[1]))] = loc[1]
    genes: dict[str, list[GeneRow]] = {}
    for row in list_genes(taxid, True, api_key):
        row.regions = oriented.get((_base_acc(row.genome_accession), _span_key(row.regions)), row.regions)
        genes.setdefault(row.genome_accession, []).append(row)
    seg_key = lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s)
    order = sorted(meta, key=lambda a: (meta[a][0], seg_key(meta[a][1])))
    return [Reference(a, meta[a][1], meta[a][2], genes.get(a, [])) for a in order]


def seqs_dir(taxid: int) -> Path:
    return Path(__file__).parent / "seqs" / str(taxid)


def gene_files(taxid: int) -> list[Path]:
    return sorted((seqs_dir(taxid) / GENES_DIR).glob("*.fasta"))


def genomes_path(taxid: int) -> Path:
    return seqs_dir(taxid) / GENOMES_DIR / "genomes.fasta"


def reference_path(taxid: int, ref: Reference) -> Path:
    return seqs_dir(taxid) / GENOMES_DIR / f"{ref.accession}.fasta"


def annotations_path(taxid: int) -> Path:
    return seqs_dir(taxid) / GENOMES_DIR / "annotations.tsv"


def fasta_ids(path: Path) -> set[str]:
    """
    record ids of a FASTA file from its header lines only
    """
    if not path.exists():
        return set()
    with open(path, encoding="ascii", errors="replace") as f:
        return {line[1:].split(None, 1)[0] for line in f if line.startswith(">") and len(line) > 1}


def _append_new(path: Path, records: list[SeqRecord], key: Callable[[str], str] = lambda i: i) -> int:
    """
    append the records whose key(id) isn't in the file yet
    returns how many were added
    """
    seen = {key(i) for i in fasta_ids(path)}
    new = []
    for r in records:
        if key(r.id) not in seen:
            seen.add(key(r.id))
            new.append(r)
    if new:
        with open(path, "a", encoding="ascii", errors="replace") as f:
            SeqIO.write(new, f, "fasta")
    return len(new)


def _genome_segments(accessions: list[str], api_key: str) -> dict[str, str]:
    """
    accession -> segment number NCBI stores for the record
    """
    out: dict[str, str] = {}
    for i in range(0, len(accessions), FETCH_BATCH):
        url = f"{DATASETS_BASE}/virus/accession/{','.join(accessions[i : i + FETCH_BATCH])}/dataset_report"
        for data in _paged_reports(url, {"page_size": "500"}, api_key):
            for rep in data.get("reports") or []:
                out[_base_acc(rep.get("accession") or "")] = str(rep.get("segment") or "")
    return out


def download_selected(
    taxid: int,
    gene_rows: list[GeneRow],
    genome_accessions: list[str],
    annotations: list[GeneRow],
    references: list[Reference],
    api_key: str,
    cancel_event: threading.Event,
    on_progress: Callable[[int, int, str], None],
) -> list[str]:
    """
    download what was ticked, adding to what seqs/<taxid>/ already holds, plus every reference
    records already there are skipped; alignments of whatever changed are deleted
    returns notes about anything skipped
    """
    notes: list[str] = []
    out_dir = seqs_dir(taxid)
    align_dir = out_dir / ALIGN_DIR
    cds_accs = sorted({r.genome_accession for r in gene_rows})
    fetch_accs = list(dict.fromkeys(sorted(genome_accessions) + [r.accession for r in references]))
    total = -(-len(cds_accs) // FETCH_BATCH) + -(-len(fetch_accs) // FETCH_BATCH)
    done = 0

    def batch_done(what: str) -> None:
        nonlocal done
        done += 1
        on_progress(done, total, f"Fetching {what}… ({done} / {total} requests)")

    # one file per gene name
    if gene_rows:
        wanted = {(_base_acc(r.genome_accession), _span_key(r.regions)) for r in gene_rows}
        cds: dict[tuple, SeqRecord] = {}
        for rec in _efetch(cds_accs, "fasta_cds_na", cancel_event, lambda: batch_done("gene sequences")):
            loc = _cds_location(rec)
            if loc and (key := (loc[0], _span_key(loc[1]))) in wanted:
                cds[key] = rec
        genes_dir = out_dir / GENES_DIR
        genes_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.stem.casefold(): p for p in genes_dir.glob("*.fasta")}
        by_file: dict[Path, list[SeqRecord]] = {}
        missing = []
        for r in gene_rows:
            rec = cds.get((_base_acc(r.genome_accession), _span_key(r.regions)))
            if rec is None:
                missing.append(f"{r.gene_name} on {r.genome_accession}")
                continue
            stem = re.sub(r"[^\w.-]", "_", r.gene_name) if not is_unnamed(r.gene_name) else \
                f"no_gene_name_{_base_acc(r.genome_accession)}_{'_'.join(f'{b}-{e}' for b, e in r.regions)}"
            # case variants (ORF6 / orf6) share one file, whichever spelling came first
            path = existing.setdefault(stem.casefold(), genes_dir / f"{stem}.fasta")
            regions = ",".join(f"{b}-{e}" for b, e in r.regions)
            desc = f"[gene={r.gene_name}] [isolate={r.isolate_name}] {r.cds_title}".strip()
            by_file.setdefault(path, []).append(SeqRecord(rec.seq, id=f"{r.genome_accession}:{regions}", description=desc))
        added = 0
        for path, records in by_file.items():
            n = _append_new(path, records)
            added += n
            for stale in align_dir.glob(f"*/{path.stem}_alignment.fasta") if n else []:
                stale.unlink()
        notes.append(f"Gene sequences: {added} new, {len(gene_rows) - len(missing) - added} already downloaded")
        if missing:
            notes.append(f"NCBI has no CDS record for {len(missing)} ticked gene(s), skipped: {', '.join(missing[:5])}")

    # references and ticked whole genomes
    genomes = {_base_acc(rec.id): rec for rec in _efetch(fetch_accs, "fasta", cancel_event, lambda: batch_done("genomes"))}
    missing = [a for a in fetch_accs if _base_acc(a) not in genomes]
    if missing:
        notes.append(f"NCBI returned no sequence for {len(missing)} record(s): {', '.join(missing[:5])}")
    genomes_path(taxid).parent.mkdir(parents=True, exist_ok=True)
    for ref in references:
        if (rec := genomes.get(_base_acc(ref.accession))) is not None:
            SeqIO.write([rec], reference_path(taxid, ref), "fasta")

    ref_ids = {_base_acc(r.accession) for r in references}
    isolates = [a for a in genome_accessions if _base_acc(a) not in ref_ids and _base_acc(a) in genomes]
    if not isolates:
        return notes
    if any(r.segment for r in references):
        segments = _genome_segments(isolates, api_key)
        for a in isolates:
            genomes[_base_acc(a)].description += f" [segment={segments.get(_base_acc(a), '')}]"
        unknown = sum(1 for a in isolates if not segments.get(_base_acc(a)))
        if unknown:
            notes.append(f"{unknown} genome(s) have no segment number at NCBI, so they can't be aligned")
    added = _append_new(genomes_path(taxid), [genomes[_base_acc(a)] for a in isolates], _base_acc)
    ann = annotations_path(taxid)
    lines = ann.read_text(encoding="utf-8").splitlines()[1:] if ann.exists() else []
    known = set(lines)
    wanted_accs = {_base_acc(a) for a in isolates}
    for g in annotations:
        if is_unnamed(g.gene_name) or _base_acc(g.genome_accession) not in wanted_accs:
            continue
        line = f"{g.genome_accession}\t{g.gene_name}\t{','.join(f'{b}-{e}' for b, e in g.regions)}"
        if line not in known:
            known.add(line)
            lines.append(line)
    ann.write_text("accession\tgene\tregions\n" + "".join(l + "\n" for l in lines), encoding="utf-8")
    for stale in [*align_dir.glob("*_alignment.fasta"), *align_dir.glob("*_genes.tsv")] if added else []:
        stale.unlink()
    notes.append(f"Whole genomes: {added} new, {len(isolates) - added} already downloaded")
    return notes
