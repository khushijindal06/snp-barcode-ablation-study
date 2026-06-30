#!/usr/bin/env python3
"""Generate E1/E2/E3 barcode datasets from a sample manifest.

Manifest columns:
  sample_id,label,vcf_path

Optional columns:
  reference,strain_name,sample_name

If the manifest does not include a per-row reference column, pass --reference.
Relative paths are resolved from the manifest file's directory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from generate_ablation_barcodes import VARIANT_PRESETS, build_variant, portable_path


REQUIRED_COLUMNS = {"sample_id", "label", "vcf_path"}


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest is empty")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"manifest is missing required columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def selected_variants(values: Sequence[str]) -> List[str]:
    return ["E1", "E2", "E3"] if "all" in values else list(values)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate E1/E2/E3 barcode datasets from a CSV manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with sample_id,label,vcf_path columns.")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference FASTA used when the manifest does not include a reference column.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("barcodes"))
    parser.add_argument("--dataset-index", type=Path, default=Path("dataset_index.csv"))
    parser.add_argument("--flank", type=int, default=50)
    parser.add_argument("--scaled-snp-scale", type=int, default=10)
    parser.add_argument("--backend", choices=("auto", "pysam", "simple"), default="auto")
    parser.add_argument("--ignore-genotypes", action="store_true")
    parser.add_argument("--gap-color", type=int, default=32)
    parser.add_argument("--marker-color", type=int, default=220)
    parser.add_argument("--chromosome-gap-pixels", type=int, default=5)
    parser.add_argument("--min-qual", type=float)
    parser.add_argument("--no-require-pass-filter", action="store_true")
    parser.add_argument("--allow-unfiltered-filter", action="store_true")
    parser.add_argument("--no-check-ref-match", action="store_true")
    parser.add_argument("--max-snps", type=int)
    parser.add_argument("--keep-going", action="store_true", help="Continue after sample failures.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("E1", "E2", "E3", "all"),
        default=["all"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest_dir = manifest_path.parent
    rows = read_manifest(manifest_path)
    variant_ids = selected_variants(args.variants)
    index_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []

    for row_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        label = row["label"].strip()
        if not sample_id or not label:
            raise ValueError(f"row {row_number}: sample_id and label are required")

        reference_text = row.get("reference", "").strip()
        if reference_text:
            reference_path = resolve_path(reference_text, manifest_dir)
        elif args.reference is not None:
            reference_path = args.reference.resolve()
        else:
            raise ValueError("pass --reference or include a reference column in the manifest")

        vcf_path = resolve_path(row["vcf_path"].strip(), manifest_dir)
        strain_name = row.get("strain_name", sample_id).strip() or sample_id
        sample_name = row.get("sample_name", "").strip() or sample_id

        for variant_id in variant_ids:
            try:
                output_png, metadata_json, encoded_snps_csv, stats = build_variant(
                    preset=VARIANT_PRESETS[variant_id],
                    reference_path=reference_path,
                    vcf_path=vcf_path,
                    output_root=args.output_root,
                    flank=args.flank,
                    scaled_snp_scale=args.scaled_snp_scale,
                    sample=sample_name,
                    ignore_genotypes=args.ignore_genotypes,
                    backend=args.backend,
                    gap_color=args.gap_color,
                    marker_color=args.marker_color,
                    chromosome_gap_pixels=args.chromosome_gap_pixels,
                    max_snps=args.max_snps,
                    explicit_output_id=sample_id,
                    require_pass_filter=not args.no_require_pass_filter,
                    allow_unfiltered_filter=args.allow_unfiltered_filter,
                    min_qual=args.min_qual,
                    check_ref_match=not args.no_check_ref_match,
                )
                index_rows.append(
                    {
                        "sample_id": sample_id,
                        "strain_name": strain_name,
                        "label": label,
                        "variant": variant_id,
                        "barcode_type": VARIANT_PRESETS[variant_id].output_dir,
                        "image_path": output_png,
                        "metadata_path": metadata_json,
                        "encoded_snps_path": encoded_snps_csv,
                        "reference": portable_path(reference_path),
                        "vcf_path": portable_path(vcf_path),
                        "snps_encoded": stats.snps_encoded,
                    }
                )
            except Exception as exc:
                failure = {
                    "row_number": row_number,
                    "sample_id": sample_id,
                    "variant": variant_id,
                    "error": str(exc),
                }
                failure_rows.append(failure)
                print(f"error: {failure}", file=sys.stderr)
                if not args.keep_going:
                    raise

    write_csv(
        args.dataset_index,
        index_rows,
        [
            "sample_id",
            "strain_name",
            "label",
            "variant",
            "barcode_type",
            "image_path",
            "metadata_path",
            "encoded_snps_path",
            "reference",
            "vcf_path",
            "snps_encoded",
        ],
    )

    if failure_rows:
        write_csv(args.dataset_index.with_suffix(".failures.csv"), failure_rows, ["row_number", "sample_id", "variant", "error"])

    print(f"Wrote dataset index: {args.dataset_index}")
    print(f"Samples: {len(rows)}")
    print(f"Barcode rows: {len(index_rows)}")
    if failure_rows:
        print(f"Failures: {len(failure_rows)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
