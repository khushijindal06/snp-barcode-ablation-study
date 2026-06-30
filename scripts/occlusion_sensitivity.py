#!/usr/bin/env python3
"""Run 1D occlusion sensitivity on a trained barcode CNN."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image
import torch

from train_cnn import SimpleBarcodeCNN


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def save_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_heatmap(path: Path, scores: Sequence[float], image_size: int) -> None:
    values = np.array(scores, dtype=np.float32)
    if values.size == 0:
        raise ValueError("no occlusion scores to plot")
    values = values - values.min()
    if values.max() > 0:
        values = values / values.max()
    row = np.interp(np.arange(image_size), np.linspace(0, image_size - 1, len(values)), values)
    heatmap = np.tile((row * 255).astype(np.uint8), (image_size, 1))
    Image.fromarray(heatmap, mode="L").save(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Occlusion sensitivity for barcode CNNs.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-heatmap", required=True, type=Path)
    parser.add_argument("--target-label", help="Class label to explain. Defaults to predicted class.")
    parser.add_argument("--window-width", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--mask-value", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    index_to_label = checkpoint["index_to_label"]
    label_to_index = checkpoint["label_to_index"]
    image_size = int(checkpoint["image_size"])
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = SimpleBarcodeCNN(num_classes=len(index_to_label)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = load_image(args.image, image_size).to(device)
    with torch.no_grad():
        base_probs = torch.softmax(model(image), dim=1)[0]
    target_index = int(base_probs.argmax()) if args.target_label is None else int(label_to_index[args.target_label])
    base_confidence = float(base_probs[target_index].item())

    rows = []
    score_track = []
    for start in range(0, image_size, args.stride):
        end = min(image_size, start + args.window_width)
        masked = image.clone()
        masked[:, :, :, start:end] = args.mask_value
        with torch.no_grad():
            confidence = float(torch.softmax(model(masked), dim=1)[0, target_index].item())
        drop = base_confidence - confidence
        rows.append(
            {
                "window_start": start,
                "window_end": end,
                "target_label": index_to_label[target_index],
                "base_confidence": base_confidence,
                "masked_confidence": confidence,
                "confidence_drop": drop,
            }
        )
        score_track.append(drop)

    save_rows(args.output_csv, rows)
    save_heatmap(args.output_heatmap, score_track, image_size)
    print(f"Wrote occlusion CSV: {args.output_csv}")
    print(f"Wrote occlusion heatmap: {args.output_heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
