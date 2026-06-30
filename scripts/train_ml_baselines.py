#!/usr/bin/env python3
"""Train traditional ML baselines on SNP presence/absence vectors."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_filter(raw_filter: str) -> Tuple[str, ...]:
    if raw_filter in {"", "."}:
        return tuple()
    return tuple(part for part in raw_filter.split(";") if part)


def parse_qual(raw_qual: str) -> Optional[float]:
    if raw_qual in {"", "."}:
        return None
    try:
        return float(raw_qual)
    except ValueError:
        return None


def passes_filters(
    filters: Tuple[str, ...],
    qual: Optional[float],
    require_pass_filter: bool,
    allow_unfiltered_filter: bool,
    min_qual: Optional[float],
) -> bool:
    if require_pass_filter:
        has_pass = filters == ("PASS",)
        is_unfiltered = len(filters) == 0
        if not has_pass and not (allow_unfiltered_filter and is_unfiltered):
            return False
    if min_qual is not None and (qual is None or qual < min_qual):
        return False
    return True


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def resolve_path(text: str, base_dir: Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (base_dir / path).resolve()


def snp_features_from_vcf(
    path: Path,
    require_pass_filter: bool,
    allow_unfiltered_filter: bool,
    min_qual: Optional[float],
) -> Dict[str, int]:
    features: Dict[str, int] = {}
    with open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, ref, alt_text = fields[0], fields[1], fields[3].upper(), fields[4].upper()
            qual = parse_qual(fields[5])
            filters = parse_filter(fields[6])
            if not passes_filters(filters, qual, require_pass_filter, allow_unfiltered_filter, min_qual):
                continue
            for alt in alt_text.split(","):
                if len(ref) == 1 and len(alt) == 1 and alt not in {".", "*"}:
                    features[f"{chrom}:{pos}:{ref}>{alt}"] = 1
    return features


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def stratify_or_none(labels: Sequence[str]) -> Optional[Sequence[str]]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return labels if counts and min(counts.values()) >= 2 else None


def split_indices(labels: Sequence[str], seed: int, test_size: float, val_size: float):
    indices = list(range(len(labels)))
    train_val, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_or_none([labels[i] for i in indices]),
    )
    if val_size <= 0 or len(train_val) < 3:
        return train_val, [], test
    relative_val = val_size / max(1e-9, 1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=stratify_or_none([labels[i] for i in train_val]),
    )
    return train, val, test


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], probabilities, classes: Sequence[str]) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if probabilities is not None:
        try:
            if len(classes) == 2:
                metrics["auc"] = roc_auc_score(y_true, probabilities[:, 1])
            else:
                metrics["auc"] = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        except ValueError:
            metrics["auc"] = ""
    else:
        metrics["auc"] = ""
    return metrics


def make_models(seed: int):
    models = {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            random_state=seed,
            eval_metric="logloss",
        )
    except Exception:
        pass
    return models


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SNP-vector ML baselines.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ml_baselines"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-qual", type=float)
    parser.add_argument("--no-require-pass-filter", action="store_true")
    parser.add_argument("--allow-unfiltered-filter", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    base_dir = manifest_path.parent
    rows = read_manifest(manifest_path)
    labels = [row["label"] for row in rows]
    if len(set(labels)) < 2:
        raise ValueError("at least two classes are required")

    feature_dicts = [
        snp_features_from_vcf(
            resolve_path(row["vcf_path"], base_dir),
            require_pass_filter=not args.no_require_pass_filter,
            allow_unfiltered_filter=args.allow_unfiltered_filter,
            min_qual=args.min_qual,
        )
        for row in rows
    ]
    train_idx, val_idx, test_idx = split_indices(labels, args.seed, args.test_size, args.val_size)
    train_idx = train_idx + val_idx
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    x_train = [feature_dicts[i] for i in train_idx]
    x_test = [feature_dicts[i] for i in test_idx]
    classes = sorted(set(labels))

    metrics_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in make_models(args.seed).items():
        pipeline = Pipeline([("vectorizer", DictVectorizer(sparse=True)), ("model", model)])
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None
        metrics = compute_metrics(y_test, y_pred, probabilities, classes)
        metrics.update({"model": model_name, "train_samples": len(train_idx), "test_samples": len(test_idx)})
        model_path = args.output_dir / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        metrics["model_path"] = str(model_path)
        metrics_rows.append(metrics)

        for idx, true_label, pred_label in zip(test_idx, y_test, y_pred):
            prediction_rows.append(
                {
                    "model": model_name,
                    "sample_id": rows[idx]["sample_id"],
                    "true_label": true_label,
                    "predicted_label": pred_label,
                }
            )

    fieldnames = sorted({key for row in metrics_rows for key in row.keys()})
    write_csv(args.output_dir / "summary_metrics.csv", metrics_rows, fieldnames)
    write_csv(args.output_dir / "predictions.csv", prediction_rows, ["model", "sample_id", "true_label", "predicted_label"])
    write_json(
        args.output_dir / "feature_summary.json",
        {
            "samples": len(rows),
            "unique_features": len(set().union(*[set(features) for features in feature_dicts])),
            "classes": classes,
        },
    )
    print(f"Wrote ML baseline results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
