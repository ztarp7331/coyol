#!/usr/bin/env python3
"""Convert a YOLO-format detection dataset into C-OLOY PGM + manifest form.

Manifest lines (see include/det.h):
  image_path [x1,y1,x2,y2,class ...]

Images are written as binary P5 PGM grayscale. Paths in the manifest are relative
to the manifest directory and contain no whitespace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, width: int, height: int
) -> tuple[float, float, float, float]:
    x1 = (cx - w * 0.5) * width
    y1 = (cy - h * 0.5) * height
    x2 = (cx + w * 0.5) * width
    y2 = (cy + h * 0.5) * height
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return x1, y1, x2, y2


def write_pgm(path: Path, pixels: bytes, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(pixels)


def convert_split(
    images_dir: Path,
    labels_dir: Path,
    out_images: Path,
    manifest_path: Path,
    max_samples: int | None,
) -> int:
    image_files = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if max_samples is not None:
        image_files = image_files[:max_samples]

    lines: list[str] = []
    written = 0
    for index, image_path in enumerate(image_files):
        label_path = labels_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            gray = image.convert("L")
            width, height = gray.size
            pixels = gray.tobytes()

        out_name = f"{index:06d}.pgm"
        write_pgm(out_images / out_name, pixels, width, height)

        boxes: list[str] = []
        if label_path.is_file():
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, width, height)
                boxes.append(
                    f"{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f},{class_id}"
                )

        rel = f"images/{out_name}"
        if boxes:
            lines.append(rel + " " + " ".join(boxes))
        else:
            lines.append(rel)
        written += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Root containing train|valid|test with images/ and labels/",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for manifests and PGM images",
    )
    parser.add_argument(
        "--splits",
        default="train,valid,test",
        help="Comma-separated split names (default: train,valid,test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional per-split cap for quick smoke runs",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        print(f"source not found: {source}", file=sys.stderr)
        return 1

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        images_dir = source / split / "images"
        labels_dir = source / split / "labels"
        if not images_dir.is_dir():
            # Some exports use val instead of valid.
            alt = "val" if split == "valid" else None
            if alt is not None and (source / alt / "images").is_dir():
                images_dir = source / alt / "images"
                labels_dir = source / alt / "labels"
            else:
                print(f"skip missing split: {split}", file=sys.stderr)
                continue
        # Put the manifest beside images/ so relative paths resolve as
        # images/<id>.pgm without an extra split prefix.
        manifest_path = output / split / "manifest.txt"
        count = convert_split(
            images_dir,
            labels_dir,
            output / split / "images",
            manifest_path,
            args.max_samples,
        )
        print(f"{split}: {count} samples -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
