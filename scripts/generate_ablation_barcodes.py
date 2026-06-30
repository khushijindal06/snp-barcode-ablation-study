#!/usr/bin/env python3
"""Generate three SNP barcode variants for an ablation study.

Variants:
  E1_context_only    : SNP context grayscale columns only.
  E2_context_scaled  : SNP context plus vertical SNP-center scaling.
  E3_full_barcode    : SNP context, SNP scaling, log genomic gaps, markers.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


BASE_TO_GRAY: Dict[str, int] = {
    "A": 255,
    "C": 85,
    "G": 170,
    "T": 0,
    "N": 128,
}

DEFAULT_GAP_COLOR = 32
DEFAULT_MARKER_COLOR = 220


@dataclass(frozen=True)
class VariantPreset:
    experiment_id: str
    output_dir: str
    label: str
    use_snp_scaling: bool
    use_gap_encoding: bool
    use_boundary_markers: bool
    purpose: str


VARIANT_PRESETS: Dict[str, VariantPreset] = {
    "E1": VariantPreset(
        experiment_id="E1",
        output_dir="E1_context_only",
        label="SNP-context barcode only",
        use_snp_scaling=False,
        use_gap_encoding=False,
        use_boundary_markers=False,
        purpose="Tests whether local nucleotide context around SNPs is sufficient.",
    ),
    "E2": VariantPreset(
        experiment_id="E2",
        output_dir="E2_context_scaled",
        label="SNP-context plus SNP-center scaling",
        use_snp_scaling=True,
        use_gap_encoding=False,
        use_boundary_markers=False,
        purpose="Tests whether visual emphasis of the variant position improves learning.",
    ),
    "E3": VariantPreset(
        experiment_id="E3",
        output_dir="E3_full_barcode",
        label="Full genomic barcode",
        use_snp_scaling=True,
        use_gap_encoding=True,
        use_boundary_markers=True,
        purpose="Tests the complete proposed representation with local context and global SNP spacing.",
    ),
}


@dataclass
class VariantRecord:
    chrom: str
    pos: int
    ref: str
    alts: Tuple[str, ...]
    qual: Optional[float]
    filters: Tuple[str, ...]
    genotypes: Dict[str, Tuple[Optional[int], ...]]


@dataclass
class BarcodeStats:
    records_seen: int = 0
    snps_encoded: int = 0
    skipped_no_alt: int = 0
    skipped_non_snp: int = 0
    skipped_bad_base: int = 0
    skipped_sample_ref: int = 0
    skipped_missing_chrom: int = 0
    skipped_filter: int = 0
    skipped_low_qual: int = 0
    skipped_ref_mismatch: int = 0
    gaps_inserted: int = 0
    gap_pixels_inserted: int = 0
    chromosome_separators: int = 0


class SimpleFastaReader:
    """Small FASTA reader used when pysam is unavailable."""

    def __init__(self, path: Path) -> None:
        self.sequences = self._read_fasta(path)

    def _read_fasta(self, path: Path) -> Dict[str, str]:
        sequences: Dict[str, List[str]] = {}
        current_name: Optional[str] = None

        with open_text(path) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    current_name = line[1:].split()[0]
                    if current_name in sequences:
                        raise ValueError(f"duplicate FASTA record: {current_name}")
                    sequences[current_name] = []
                    continue
                if current_name is None:
                    raise ValueError("FASTA sequence appeared before a header")
                sequences[current_name].append(line.upper())

        if not sequences:
            raise ValueError(f"no FASTA records found in {path}")

        return {name: "".join(parts) for name, parts in sequences.items()}

    def get_reference_length(self, chrom: str) -> int:
        if chrom not in self.sequences:
            raise KeyError(chrom)
        return len(self.sequences[chrom])

    def fetch(self, chrom: str, start: int, end: int) -> str:
        if chrom not in self.sequences:
            raise KeyError(chrom)
        return self.sequences[chrom][start:end]

    def close(self) -> None:
        return None


class PysamFastaReader:
    def __init__(self, path: Path) -> None:
        import pysam  # type: ignore

        self._fasta = pysam.FastaFile(str(path))

    def get_reference_length(self, chrom: str) -> int:
        return self._fasta.get_reference_length(chrom)

    def fetch(self, chrom: str, start: int, end: int) -> str:
        return self._fasta.fetch(chrom, start, end)

    def close(self) -> None:
        self._fasta.close()


class SimpleVcfReader:
    """Streaming VCF reader for plain text or gzipped VCF files."""

    def __init__(self, path: Path) -> None:
        self._handle = open_text(path)
        self.samples = self._read_header()

    def _read_header(self) -> List[str]:
        for raw_line in self._handle:
            line = raw_line.rstrip("\n")
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.split("\t")
                return fields[9:] if len(fields) > 9 else []
            if line.startswith("#"):
                continue
            raise ValueError("VCF header is missing #CHROM line")
        raise ValueError("VCF header is missing #CHROM line")

    def __iter__(self) -> Iterator[VariantRecord]:
        for raw_line in self._handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) < 8:
                continue

            yield VariantRecord(
                chrom=fields[0],
                pos=int(fields[1]),
                ref=fields[3].upper(),
                alts=tuple(alt.upper() for alt in fields[4].split(",") if alt and alt != "."),
                qual=parse_qual(fields[5]),
                filters=parse_filter_value(fields[6]),
                genotypes=parse_genotypes(fields, self.samples),
            )

    def close(self) -> None:
        self._handle.close()


class PysamVcfReader:
    def __init__(self, path: Path) -> None:
        import pysam  # type: ignore

        self._vcf = pysam.VariantFile(str(path))
        self.samples = list(self._vcf.header.samples)

    def __iter__(self) -> Iterator[VariantRecord]:
        for record in self._vcf:
            genotypes: Dict[str, Tuple[Optional[int], ...]] = {}
            for sample in self.samples:
                gt = record.samples[sample].get("GT")
                if gt is not None:
                    genotypes[sample] = tuple(gt)
            filter_keys = tuple(str(name) for name in record.filter.keys())

            yield VariantRecord(
                chrom=record.chrom,
                pos=record.pos,
                ref=record.ref.upper(),
                alts=tuple(alt.upper() for alt in (record.alts or ())),
                qual=float(record.qual) if record.qual is not None else None,
                filters=filter_keys if filter_keys else ("PASS",),
                genotypes=genotypes,
            )

    def close(self) -> None:
        self._vcf.close()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_gt(gt_text: str) -> Tuple[Optional[int], ...]:
    alleles: List[Optional[int]] = []
    for part in gt_text.replace("|", "/").split("/"):
        if part in {"", "."}:
            alleles.append(None)
            continue
        try:
            alleles.append(int(part))
        except ValueError:
            alleles.append(None)
    return tuple(alleles)


def parse_qual(raw_qual: str) -> Optional[float]:
    if raw_qual in {"", "."}:
        return None
    try:
        return float(raw_qual)
    except ValueError:
        return None


def parse_filter_value(raw_filter: str) -> Tuple[str, ...]:
    if raw_filter in {"", "."}:
        return tuple()
    return tuple(part for part in raw_filter.split(";") if part)


def parse_genotypes(fields: Sequence[str], samples: Sequence[str]) -> Dict[str, Tuple[Optional[int], ...]]:
    if len(fields) < 10 or not samples:
        return {}

    format_keys = fields[8].split(":")
    if "GT" not in format_keys:
        return {}

    gt_index = format_keys.index("GT")
    genotypes: Dict[str, Tuple[Optional[int], ...]] = {}

    for sample_name, sample_field in zip(samples, fields[9:]):
        values = sample_field.split(":")
        if gt_index < len(values):
            genotypes[sample_name] = parse_gt(values[gt_index])

    return genotypes


def open_readers(reference_path: Path, vcf_path: Path, backend: str):
    if backend not in {"auto", "pysam", "simple"}:
        raise ValueError(f"unsupported backend: {backend}")

    if backend in {"auto", "pysam"}:
        try:
            import pysam  # noqa: F401  # type: ignore

            return PysamFastaReader(reference_path), PysamVcfReader(vcf_path), "pysam"
        except Exception as exc:
            if backend == "pysam":
                raise RuntimeError(f"could not initialize pysam backend: {exc}") from exc

    return SimpleFastaReader(reference_path), SimpleVcfReader(vcf_path), "simple"


def choose_sample(vcf_reader, requested_sample: Optional[str], ignore_genotypes: bool) -> Optional[str]:
    samples = list(getattr(vcf_reader, "samples", []))

    if ignore_genotypes:
        return None

    if requested_sample:
        if requested_sample not in samples:
            available = ", ".join(samples) if samples else "(none)"
            raise ValueError(f"sample '{requested_sample}' was not found; available samples: {available}")
        return requested_sample

    if len(samples) == 1:
        return samples[0]

    return None


def select_alt_base(record: VariantRecord, sample: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not record.alts:
        return None, "no_alt"

    if sample is None:
        return record.alts[0], None

    genotype = record.genotypes.get(sample)
    if genotype is None:
        return None, "sample_ref"

    for allele_index in genotype:
        if allele_index is None or allele_index == 0:
            continue
        alt_index = allele_index - 1
        if 0 <= alt_index < len(record.alts):
            return record.alts[alt_index], None

    return None, "sample_ref"


def is_valid_snp(ref_base: str, alt_base: str) -> Tuple[bool, str]:
    if len(ref_base) != 1 or len(alt_base) != 1:
        return False, "non_snp"
    if ref_base.upper() not in BASE_TO_GRAY or alt_base.upper() not in BASE_TO_GRAY:
        return False, "bad_base"
    return True, ""


def passes_quality_filters(
    record: VariantRecord,
    require_pass_filter: bool,
    allow_unfiltered_filter: bool,
    min_qual: Optional[float],
) -> Tuple[bool, str]:
    if require_pass_filter:
        has_pass = record.filters == ("PASS",)
        is_unfiltered = len(record.filters) == 0
        if not has_pass and not (allow_unfiltered_filter and is_unfiltered):
            return False, "filter"

    if min_qual is not None:
        if record.qual is None or record.qual < min_qual:
            return False, "qual"

    return True, ""


def fetch_context(ref_reader, chrom: str, pos: int, flank: int) -> str:
    center_0based = pos - 1
    start = center_0based - flank
    end = center_0based + flank + 1

    chrom_len = ref_reader.get_reference_length(chrom)
    left_pad = max(0, -start)
    right_pad = max(0, end - chrom_len)
    safe_start = max(0, start)
    safe_end = min(chrom_len, end)

    seq = ref_reader.fetch(chrom, safe_start, safe_end).upper()
    return ("N" * left_pad) + seq + ("N" * right_pad)


def gap_pixels(distance: int) -> int:
    """Convert inter-SNP distance into non-overlapping log-scaled gap pixels.

    The bins match the Objective 1 presentation:
      0 px = 0-10 bp
      1 px = 11-100 bp
      2 px = 101-1,000 bp
      3 px = 1,001-10,000 bp
      4 px = 10,001-100,000 bp
      and so on.
    """
    if distance <= 10:
        return 0
    return int(math.floor(math.log10(distance - 1)))


def encode_column(context: str, alt_base: str, flank: int, snp_scale: int) -> np.ndarray:
    bases = list(context.upper())
    bases[flank] = alt_base.upper()

    values: List[int] = []
    values.extend(BASE_TO_GRAY.get(base, BASE_TO_GRAY["N"]) for base in bases[:flank])
    values.extend([BASE_TO_GRAY.get(bases[flank], BASE_TO_GRAY["N"])] * snp_scale)
    values.extend(BASE_TO_GRAY.get(base, BASE_TO_GRAY["N"]) for base in bases[flank + 1 :])
    return np.array(values, dtype=np.uint8).reshape(-1, 1)


def output_stem(vcf_path: Path, sample_used: Optional[str], explicit_id: Optional[str]) -> str:
    if explicit_id:
        return explicit_id
    if sample_used:
        return sample_used
    name = vcf_path.name
    for suffix in (".vcf.gz", ".vcf", ".bcf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return vcf_path.stem


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def build_variant(
    preset: VariantPreset,
    reference_path: Path,
    vcf_path: Path,
    output_root: Path,
    flank: int,
    scaled_snp_scale: int,
    sample: Optional[str],
    ignore_genotypes: bool,
    backend: str,
    gap_color: int,
    marker_color: int,
    chromosome_gap_pixels: int,
    max_snps: Optional[int],
    explicit_output_id: Optional[str],
    require_pass_filter: bool = True,
    allow_unfiltered_filter: bool = False,
    min_qual: Optional[float] = None,
    check_ref_match: bool = True,
) -> Tuple[Path, Path, Path, BarcodeStats]:
    ref_reader, vcf_reader, backend_used = open_readers(reference_path, vcf_path, backend)
    selected_sample = choose_sample(vcf_reader, sample, ignore_genotypes)
    sample_stem = output_stem(vcf_path, selected_sample, explicit_output_id)

    snp_scale = scaled_snp_scale if preset.use_snp_scaling else 1
    height = flank + snp_scale + flank
    stats = BarcodeStats()
    columns: List[np.ndarray] = []
    encoded_rows: List[Dict[str, object]] = []
    current_width = 0

    if preset.use_boundary_markers:
        columns.append(np.full((height, 1), marker_color, dtype=np.uint8))
        current_width += 1

    previous_pos: Optional[int] = None
    previous_chrom: Optional[str] = None

    try:
        for record in vcf_reader:
            stats.records_seen += 1

            passes_filters, filter_reason = passes_quality_filters(
                record=record,
                require_pass_filter=require_pass_filter,
                allow_unfiltered_filter=allow_unfiltered_filter,
                min_qual=min_qual,
            )
            if not passes_filters:
                if filter_reason == "filter":
                    stats.skipped_filter += 1
                else:
                    stats.skipped_low_qual += 1
                continue

            alt_base, reason = select_alt_base(record, selected_sample)
            if reason == "no_alt":
                stats.skipped_no_alt += 1
                continue
            if reason == "sample_ref":
                stats.skipped_sample_ref += 1
                continue
            if alt_base is None:
                stats.skipped_no_alt += 1
                continue

            ok, invalid_reason = is_valid_snp(record.ref, alt_base)
            if not ok:
                if invalid_reason == "non_snp":
                    stats.skipped_non_snp += 1
                else:
                    stats.skipped_bad_base += 1
                continue

            try:
                context = fetch_context(ref_reader, record.chrom, record.pos, flank)
            except KeyError:
                stats.skipped_missing_chrom += 1
                continue

            if len(context) != (2 * flank + 1):
                stats.skipped_missing_chrom += 1
                continue

            reference_base = context[flank].upper()
            if check_ref_match and reference_base != record.ref.upper():
                stats.skipped_ref_mismatch += 1
                continue

            gap_before = 0
            if preset.use_gap_encoding:
                if previous_pos is not None and previous_chrom == record.chrom:
                    width = gap_pixels(max(0, record.pos - previous_pos))
                    if width > 0:
                        columns.append(np.full((height, width), gap_color, dtype=np.uint8))
                        stats.gaps_inserted += 1
                        stats.gap_pixels_inserted += width
                        current_width += width
                        gap_before = width
                elif previous_pos is not None and previous_chrom != record.chrom:
                    columns.append(np.full((height, chromosome_gap_pixels), gap_color, dtype=np.uint8))
                    stats.gaps_inserted += 1
                    stats.gap_pixels_inserted += chromosome_gap_pixels
                    stats.chromosome_separators += 1
                    current_width += chromosome_gap_pixels
                    gap_before = chromosome_gap_pixels

            barcode_column = current_width
            columns.append(encode_column(context, alt_base, flank, snp_scale))
            current_width += 1
            stats.snps_encoded += 1
            encoded_rows.append(
                {
                    "barcode_column": barcode_column,
                    "chrom": record.chrom,
                    "pos": record.pos,
                    "ref": record.ref,
                    "reference_base": reference_base,
                    "alt": alt_base,
                    "qual": "" if record.qual is None else record.qual,
                    "filter": ";".join(record.filters) if record.filters else ".",
                    "gap_before": gap_before,
                    "center_row_start": flank,
                    "center_row_end": flank + snp_scale - 1,
                    "context_start": max(1, record.pos - flank),
                    "context_end": record.pos + flank,
                }
            )
            previous_pos = record.pos
            previous_chrom = record.chrom

            if max_snps is not None and stats.snps_encoded >= max_snps:
                break

    finally:
        ref_reader.close()
        vcf_reader.close()

    if preset.use_boundary_markers:
        columns.append(np.full((height, 1), marker_color, dtype=np.uint8))
        current_width += 1

    if not columns:
        raise ValueError(f"{preset.experiment_id}: no SNPs were encoded")

    image_array = np.concatenate(columns, axis=1)
    variant_dir = output_root / preset.output_dir
    variant_dir.mkdir(parents=True, exist_ok=True)

    output_png = variant_dir / f"{sample_stem}_{preset.output_dir}.png"
    metadata_json = variant_dir / f"{sample_stem}_{preset.output_dir}.json"
    encoded_snps_csv = variant_dir / f"{sample_stem}_{preset.output_dir}_encoded_snps.csv"
    Image.fromarray(image_array).save(output_png)
    write_encoded_snps(encoded_snps_csv, encoded_rows)

    metadata = {
        "experiment_id": preset.experiment_id,
        "label": preset.label,
        "purpose": preset.purpose,
        "reference_fasta": portable_path(reference_path),
        "vcf": portable_path(vcf_path),
        "sample_used": selected_sample,
        "backend_used": backend_used,
        "output_png": portable_path(output_png),
        "encoded_snps_csv": portable_path(encoded_snps_csv),
        "height": int(image_array.shape[0]),
        "width": int(image_array.shape[1]),
        "flank": flank,
        "snp_scale": snp_scale,
        "components": {
            "snp_context_encoding": True,
            "snp_center_scaling": preset.use_snp_scaling,
            "gap_encoding": preset.use_gap_encoding,
            "boundary_markers": preset.use_boundary_markers,
        },
        "base_to_gray": BASE_TO_GRAY,
        "gap_color": gap_color,
        "marker_color": marker_color,
        "chromosome_gap_pixels": chromosome_gap_pixels,
        "filters": {
            "require_pass_filter": require_pass_filter,
            "allow_unfiltered_filter": allow_unfiltered_filter,
            "min_qual": min_qual,
            "check_ref_match": check_ref_match,
        },
        "stats": asdict(stats),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        f"{preset.experiment_id}: saved {output_png} "
        f"({image_array.shape[0]} rows x {image_array.shape[1]} columns, "
        f"{stats.snps_encoded} SNPs)"
    )

    return output_png, metadata_json, encoded_snps_csv, stats


def write_encoded_snps(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "barcode_column",
        "chrom",
        "pos",
        "ref",
        "reference_base",
        "alt",
        "qual",
        "filter",
        "gap_before",
        "center_row_start",
        "center_row_end",
        "context_start",
        "context_end",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate E1/E2/E3 SNP barcode ablation variants from one reference FASTA and SNP VCF."
    )
    parser.add_argument("--reference", required=True, type=Path, help="Reference FASTA path.")
    parser.add_argument("--vcf", required=True, type=Path, help="SNP VCF path, plain or .gz.")
    parser.add_argument("--output-root", type=Path, default=Path("barcodes"), help="Output folder for variant folders.")
    parser.add_argument("--flank", type=int, default=50, help="Number of flanking bases on each side.")
    parser.add_argument("--scaled-snp-scale", type=int, default=10, help="SNP-center scale used for E2 and E3.")
    parser.add_argument("--sample", help="VCF sample name for genotype-aware ALT selection.")
    parser.add_argument("--output-id", help="Override output filename stem.")
    parser.add_argument("--ignore-genotypes", action="store_true", help="Always use the first ALT allele.")
    parser.add_argument("--backend", choices=("auto", "pysam", "simple"), default="auto")
    parser.add_argument("--gap-color", type=int, default=DEFAULT_GAP_COLOR)
    parser.add_argument("--marker-color", type=int, default=DEFAULT_MARKER_COLOR)
    parser.add_argument("--chromosome-gap-pixels", type=int, default=5)
    parser.add_argument("--min-qual", type=float, help="Optional minimum VCF QUAL threshold.")
    parser.add_argument(
        "--no-require-pass-filter",
        action="store_true",
        help="Do not require VCF FILTER=PASS before encoding a SNP.",
    )
    parser.add_argument(
        "--allow-unfiltered-filter",
        action="store_true",
        help="When PASS filtering is enabled, also allow VCF records with FILTER='.'.",
    )
    parser.add_argument(
        "--no-check-ref-match",
        action="store_true",
        help="Do not verify that the VCF REF allele matches the reference FASTA base.",
    )
    parser.add_argument("--max-snps", type=int, help="Optional cap for previews or tests.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("E1", "E2", "E3", "all"),
        default=["all"],
        help="Ablation variants to generate.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.flank < 0:
        print("error: --flank must be >= 0", file=sys.stderr)
        return 1
    if args.scaled_snp_scale < 1:
        print("error: --scaled-snp-scale must be >= 1", file=sys.stderr)
        return 1

    selected_variant_ids = ["E1", "E2", "E3"] if "all" in args.variants else args.variants

    try:
        for variant_id in selected_variant_ids:
            build_variant(
                preset=VARIANT_PRESETS[variant_id],
                reference_path=args.reference,
                vcf_path=args.vcf,
                output_root=args.output_root,
                flank=args.flank,
                scaled_snp_scale=args.scaled_snp_scale,
                sample=args.sample,
                ignore_genotypes=args.ignore_genotypes,
                backend=args.backend,
                gap_color=args.gap_color,
                marker_color=args.marker_color,
                chromosome_gap_pixels=args.chromosome_gap_pixels,
                max_snps=args.max_snps,
                explicit_output_id=args.output_id,
                require_pass_filter=not args.no_require_pass_filter,
                allow_unfiltered_filter=args.allow_unfiltered_filter,
                min_qual=args.min_qual,
                check_ref_match=not args.no_check_ref_match,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
