#!/usr/bin/env python3
"""Train and test CNN models on E1/E2/E3 barcode datasets."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class BarcodeDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, str]], label_to_index: Dict[str, int], image_size: int) -> None:
        self.rows = list(rows)
        self.label_to_index = label_to_index
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("L").resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        label = torch.tensor(self.label_to_index[row["label"]], dtype=torch.long)
        return tensor, label, row["sample_id"]


class SimpleBarcodeCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratify_or_none(labels: Sequence[str]) -> Optional[Sequence[str]]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return labels if counts and min(counts.values()) >= 2 else None


def split_samples(rows: Sequence[Dict[str, str]], seed: int, test_size: float, val_size: float):
    sample_to_label = {row["sample_id"]: row["label"] for row in rows}
    sample_ids = sorted(sample_to_label)
    labels = [sample_to_label[sample_id] for sample_id in sample_ids]
    if len(set(labels)) < 2:
        raise ValueError("at least two classes are required for training")
    if len(sample_ids) < 3:
        raise ValueError("at least three samples are required for train/validation/test splitting")

    train_val_ids, test_ids = train_test_split(
        sample_ids,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_or_none(labels),
    )

    train_val_labels = [sample_to_label[sample_id] for sample_id in train_val_ids]
    relative_val = val_size / max(1e-9, 1.0 - test_size)
    if val_size <= 0 or len(train_val_ids) < 3:
        train_ids, val_ids = train_val_ids, []
    else:
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=relative_val,
            random_state=seed,
            stratify=stratify_or_none(train_val_labels),
        )
    return set(train_ids), set(val_ids), set(test_ids)


def filter_rows(rows: Sequence[Dict[str, str]], variant: str, sample_ids: set[str]) -> List[Dict[str, str]]:
    return [row for row in rows if row["variant"] == variant and row["sample_id"] in sample_ids]


def run_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total_items += x.size(0)
    return total_loss / max(1, total_items)


def loss_on_loader(model, loader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            loss = criterion(model(x), y)
            total_loss += float(loss.item()) * x.size(0)
            total_items += x.size(0)
    return total_loss / max(1, total_items)


def predict(model, loader, device: torch.device):
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []
    all_samples: List[str] = []
    with torch.no_grad():
        for x, y, sample_ids in loader:
            probs = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y.numpy().tolist())
            all_samples.extend(sample_ids)
    return np.vstack(all_probs), np.array(all_labels, dtype=int), all_samples


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, labels: Sequence[str]) -> Dict[str, object]:
    y_pred = probabilities.argmax(axis=1)
    metrics: Dict[str, object] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        if len(labels) == 2:
            metrics["auc"] = roc_auc_score(y_true, probabilities[:, 1])
        else:
            metrics["auc"] = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auc"] = ""
    return metrics


def train_variant(
    rows: Sequence[Dict[str, str]],
    variant: str,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    label_to_index: Dict[str, int],
    index_to_label: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    variant_dir = args.output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    train_rows = filter_rows(rows, variant, train_ids)
    val_rows = filter_rows(rows, variant, val_ids)
    test_rows = filter_rows(rows, variant, test_ids)
    if not train_rows or not test_rows:
        raise ValueError(f"{variant}: missing train or test rows")

    train_loader = DataLoader(
        BarcodeDataset(train_rows, label_to_index, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        BarcodeDataset(val_rows or test_rows, label_to_index, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        BarcodeDataset(test_rows, label_to_index, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SimpleBarcodeCNN(num_classes=len(index_to_label)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_val_loss = float("inf")

    history_rows: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = loss_on_loader(model, val_loader, criterion, device)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    probabilities, y_true, sample_ids = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, probabilities, index_to_label)
    metrics.update(
        {
            "variant": variant,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
            "epochs": args.epochs,
            "image_size": args.image_size,
        }
    )

    checkpoint_path = variant_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_to_index": label_to_index,
            "index_to_label": index_to_label,
            "image_size": args.image_size,
            "variant": variant,
        },
        checkpoint_path,
    )
    metrics["checkpoint"] = str(checkpoint_path)

    prediction_rows = []
    for sample_id, true_index, probs in zip(sample_ids, y_true, probabilities):
        row: Dict[str, object] = {
            "sample_id": sample_id,
            "true_label": index_to_label[int(true_index)],
            "predicted_label": index_to_label[int(probs.argmax())],
        }
        for label, prob in zip(index_to_label, probs):
            row[f"prob_{label}"] = float(prob)
        prediction_rows.append(row)

    write_csv(variant_dir / "history.csv", history_rows, ["epoch", "train_loss", "val_loss"])
    write_csv(variant_dir / "predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    write_json(variant_dir / "metrics.json", metrics)
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNNs on E1/E2/E3 barcode images.")
    parser.add_argument("--dataset-index", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cnn"))
    parser.add_argument("--variants", nargs="+", choices=("E1", "E2", "E3", "all"), default=["all"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    rows = read_csv(args.dataset_index)
    variant_ids = ["E1", "E2", "E3"] if "all" in args.variants else args.variants
    labels = sorted({row["label"] for row in rows})
    label_encoder = LabelEncoder().fit(labels)
    label_to_index = {label: int(label_encoder.transform([label])[0]) for label in labels}
    index_to_label = list(label_encoder.classes_)
    train_ids, val_ids, test_ids = split_samples(rows, args.seed, args.test_size, args.val_size)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    split_rows = []
    for split_name, sample_ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for sample_id in sorted(sample_ids):
            label = next(row["label"] for row in rows if row["sample_id"] == sample_id)
            split_rows.append({"sample_id": sample_id, "label": label, "split": split_name})
    write_csv(args.output_dir / "splits.csv", split_rows, ["sample_id", "label", "split"])

    metrics_rows = []
    for variant in variant_ids:
        metrics = train_variant(rows, variant, train_ids, val_ids, test_ids, label_to_index, index_to_label, args, device)
        metrics_rows.append(metrics)

    fieldnames = sorted({key for row in metrics_rows for key in row.keys()})
    write_csv(args.output_dir / "summary_metrics.csv", metrics_rows, fieldnames)
    print(f"Wrote CNN results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
