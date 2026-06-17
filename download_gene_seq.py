import os
import re
import time
import threading
from typing import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from dotenv import load_dotenv, set_key


DATASETS_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2"
ENV_PATH = Path(__file__).parent / ".env"


def _log(msg: str) -> None:
    print(f"[log] {msg}", flush=True)


@dataclass
class Taxon:
    taxid: int
    scientific_name: str


@dataclass
class GeneRow:
    gene_name: str
    gene_id: int
    genome_accession: str
    isolate_name: str
    regions: list[tuple[int, int]]
    cds_title: str


def configure_entrez(email: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    """
    The single entry point for NCBI credentials.
    Called with no arguments (startup): load EMAIL / NCBI_API_KEY from .env
    Called with arguments (the user typed them in the GUI): overwrite .env
    Either way, returns the (email, api_key) now in effect.
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


def is_rdrp(gene_name: str) -> bool:
    n = (gene_name or "").lower()
    return "rdrp" in n or "rna-dependent rna polymerase" in n


def group_into_files(taxid: int, rows: list[GeneRow]) -> dict[Path, list[GeneRow]]:
    """
    Map each output FASTA path to the rows written into it.
    The directory is the taxid; each named gene becomes <gene>.fasta
    A CDS with no gene name instead becomes no_gene_name_<accession>_<region>.fasta
    Rows that share a file land in it together.
    """
    out_dir = Path(__file__).parent / "seqs" / str(taxid)
    groups: dict[Path, list[GeneRow]] = {}
    for row in rows:
        if is_unnamed(row.gene_name):
            range_str = ",".join(f"{b}-{e}" for b, e in row.regions)
            stem = f"no_gene_name_{row.genome_accession}_{range_str}"
        else:
            stem = row.gene_name
        groups.setdefault(out_dir / f"{stem}.fasta", []).append(row)
    return groups


def resolve_taxon(query: str) -> Taxon:
    """
    Resolve a taxid or a taxon name to a Taxon dataclass
    """
    q = query.strip()
    if not q:
        raise ValueError("Empty taxon query")
    if q.isdigit():
        taxid = int(q)
    else:
        with Entrez.esearch(db="taxonomy", term=q) as handle:
            result = Entrez.read(handle)
        ids = result.get("IdList") or []
        if not ids:
            raise LookupError(f"No taxon matches '{q}'")
        taxid = int(ids[0])
    with Entrez.esummary(db="taxonomy", id=str(taxid)) as handle:
        summaries = Entrez.read(handle)
    if not summaries:
        raise LookupError(f"No taxon with taxid {taxid}")
    name = (
        summaries[0].get("ScientificName")
        or summaries[0].get("CommonName")
        or str(taxid)
    )
    _log(f"resolve_taxon: '{query}' -> {name} (taxid {taxid})")
    return Taxon(taxid=taxid, scientific_name=str(name))


def list_genes(taxid: int, refseq_only: bool, api_key: str = "", *, geo_location: str = "",
    annotated_only: bool = False, complete_only: bool = False, released_since: str = "",
    progress: Callable[[int], None] | None = None,) -> list[GeneRow]:
    """
    List annotated genes for a taxon by NCBI annotation report.

    """
    url = f"{DATASETS_BASE}/virus/taxon/{taxid}/annotation_report"
    headers = {"accept": "application/json"}
    if api_key:
        headers["api-key"] = api_key
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

    _log(
        f"list_genes start: taxid={taxid} refseq_only={refseq_only} "
        f"geo_location={geo_location.strip()!r} annotated_only={annotated_only} "
        f"complete_only={complete_only} released_since={released_since.strip()!r} "
        f"params={params}"
    )

    rows: list[GeneRow] = []
    next_token: str | None = None
    page = 0
    with httpx.Client(timeout=60.0) as client:
        while True:
            page += 1
            if next_token:
                params["page_token"] = next_token
            _log(f"page {page}: GET {url} params={params}")
            start = time.monotonic()
            try:
                response = client.get(url, params=params, headers=headers)
            except httpx.TimeoutException as e:
                _log(
                    f"page {page}: TIMED OUT after {time.monotonic() - start:.1f}s "
                    f"({type(e).__name__})"
                )
                raise RuntimeError("NCBI no response") from e
            elapsed = time.monotonic() - start
            _log(
                f"page {page}: HTTP {response.status_code} in {elapsed:.2f}s "
                f"({len(response.content)} bytes)"
            )
            if response.status_code >= 500:
                body = response.text[:300]
                _log(f"page {page}: server error body: {body}")
                if "forwarding request" in body.lower():
                    raise RuntimeError(
                        "NCBI couldn't compute this search in time (HTTP 500 after ~20s)."
                    )
                raise RuntimeError(
                    f"NCBI returned a server error (HTTP {response.status_code}). "
                    "This may be transient — click Search again."
                )
            if response.status_code >= 400:
                _log(f"page {page}: client error body: {response.text[:300]}")
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
            data = response.json()
            reports = data.get("reports") or []
            rows_before = len(rows)
            for report in reports:
                genome_accession = report.get("accession", "") or ""
                isolate_name = report.get("isolate_name", "") or ""
                for gene in report.get("genes") or []:
                    cds_list = gene.get("cds") or []
                    if not cds_list:
                        continue
                    nucleotide = cds_list[0].get("nucleotide") or {}
                    regions: list[tuple[int, int]] = []
                    for r in nucleotide.get("range") or []:
                        try:
                            regions.append((int(r["begin"]), int(r["end"])))
                        except (KeyError, TypeError, ValueError):
                            continue
                    if not regions:
                        continue
                    gene_id = int(gene.get("gene_id") or 0)
                    rows.append(
                        GeneRow(
                            gene_name=(gene.get("name") or "").strip(),
                            gene_id=gene_id,
                            genome_accession=genome_accession,
                            isolate_name=isolate_name,
                            regions=regions,
                            cds_title=nucleotide.get("title", "") or "",
                        )
                    )
            next_token = data.get("next_page_token") or None
            _log(
                f"page {page}: {len(reports)} reports, "
                f"+{len(rows) - rows_before} rows (total {len(rows)}), "
                f"more_pages={bool(next_token)}"
            )
            if progress is not None:
                progress(len(rows))
            if not next_token:
                break
    _log(f"list_genes done: {len(rows)} rows across {page} page(s)")
    return rows


def _location_coords(description: str) -> tuple[tuple[int, int], ...]:
    """
    Extract the coordinate set from a fasta_cds_na header's [location=...] tag
    This matches a GeneRow's regions regardless of strand
    """
    match = re.search(r"\[location=([^\]]+)\]", description)
    if not match:
        return ()
    return tuple(sorted(
        (int(a), int(b))
        for a, b in re.findall(r"(\d+)\.\.(\d+)", match.group(1))
    ))


def _genome_cds_index(accession: str,) -> dict[tuple[tuple[int, int], ...], SeqRecord]:
    """
    Fetch every CDS nucleotide sequence for one genome, keyed by coordinate set:
    resolves join() for multi-exon CDS, so a minus-strand gene comes back as its true coding
    sequence, not reverse-complemented genome slice.
    """
    with Entrez.efetch(
        db="nuccore", id=accession, rettype="fasta_cds_na", retmode="text"
    ) as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    index: dict[tuple[tuple[int, int], ...], SeqRecord] = {}
    for record in records:
        coords = _location_coords(record.description)
        if coords:
            index[coords] = record
    return index


def download_selected(taxid: int, rows: list[GeneRow], cancel_event: threading.Event, on_file_done: Callable[[Path, int], None],) -> None:
    """
    Download every selected gene's CDS and write one FASTA per gene name.
    """
    cds_cache: dict[str, dict[tuple[tuple[int, int], ...], SeqRecord]] = {}
    for path, group_rows in group_into_files(taxid, rows).items():
        if cancel_event.is_set():
            raise InterruptedError("cancelled")
        records: list[SeqRecord] = []
        for row in group_rows:
            if cancel_event.is_set():
                raise InterruptedError("cancelled")
            if row.genome_accession not in cds_cache:
                cds_cache[row.genome_accession] = _genome_cds_index(
                    row.genome_accession
                )
            cds = cds_cache[row.genome_accession].get(tuple(sorted(row.regions)))
            if cds is None:
                raise RuntimeError(
                    f"No CDS at {row.regions} on {row.genome_accession} "
                    f"for gene {row.gene_name}"
                )
            range_str = ",".join(f"{b}-{e}" for b, e in row.regions)
            records.append(
                SeqRecord(
                    cds.seq,
                    id=f"{row.genome_accession}:{range_str}",
                    description=row.cds_title or row.gene_name,
                )
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        SeqIO.write(records, path, "fasta")
        on_file_done(path, len(records))
