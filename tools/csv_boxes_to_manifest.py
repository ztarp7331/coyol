#!/usr/bin/env python3
"""Convert CSV absolute boxes (image,xmin,ymin,xmax,ymax) to PGM + manifest."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


def write_pgm(path: Path, pixels: bytes, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(pixels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directory of source images")
    parser.add_argument("--csv", required=True, help="CSV with image,xmin,ymin,xmax,ymax")
    parser.add_argument("--output", required=True, help="Output split directory")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--class-id", type=int, default=0)
    args = parser.parse_args()

    images_dir = Path(args.images)
    boxes_by_image: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with Path(args.csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["image"].strip()
            x1 = float(row["xmin"])
            y1 = float(row["ymin"])
            x2 = float(row["xmax"])
            y2 = float(row["ymax"])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_by_image[name].append((x1, y1, x2, y2))

    names = sorted(boxes_by_image.keys())
    rng = random.Random(args.seed)
    rng.shuffle(names)
    val_count = max(1, int(len(names) * args.val_fraction))
    splits = {
        "valid": names[:val_count],
        "train": names[val_count:],
    }

    for split, split_names in splits.items():
        out_images = Path(args.output) / split / "images"
        lines: list[str] = []
        for index, name in enumerate(split_names):
            src = images_dir / name
            if not src.is_file():
                continue
            with Image.open(src) as image:
                gray = image.convert("L")
                width, height = gray.size
                pixels = gray.tobytes()
            out_name = f"{index:06d}.pgm"
            write_pgm(out_images / out_name, pixels, width, height)
            tokens = []
            for x1, y1, x2, y2 in boxes_by_image[name]:
                x1c = max(0.0, min(float(width), x1))
                y1c = max(0.0, min(float(height), y1))
                x2c = max(0.0, min(float(width), x2))
                y2c = max(0.0, min(float(height), y2))
                if x2c <= x1c or y2c <= y1c:
                    continue
                tokens.append(
                    f"{x1c:.4f},{y1c:.4f},{x2c:.4f},{y2c:.4f},{args.class_id}"
                )
            line = f"images/{out_name}"
            if tokens:
                line += " " + " ".join(tokens)
            lines.append(line)
        manifest = Path(args.output) / split / "manifest.txt"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"{split}: {len(lines)} samples -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
