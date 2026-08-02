#!/usr/bin/env python3
"""Draw GT (green) and PRED (red) boxes from viz_detect output onto PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_pgm(path: Path) -> Image.Image:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P5":
            raise ValueError(f"expected P5 PGM: {path}")
        # skip comments
        while True:
            line = handle.readline()
            if not line.startswith(b"#"):
                break
        width, height = map(int, line.split())
        maxval = int(handle.readline())
        if maxval != 255:
            raise ValueError(f"unsupported maxval {maxval}")
        data = handle.read(width * height)
    return Image.frombytes("L", (width, height), data).convert("RGB")


def parse_report(path: Path):
    samples = []
    current = None
    model_w = model_h = 160
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# model_input"):
            # "# model_input 160x160 ..."
            parts = raw.split()
            if len(parts) >= 3 and "x" in parts[2]:
                model_w, model_h = map(int, parts[2].split("x"))
            continue
        if raw.startswith("#") or not raw.strip():
            continue
        toks = raw.split()
        if toks[0] == "SAMPLE":
            if current is not None:
                samples.append(current)
            current = {
                "index": int(toks[1]),
                "image_rel": toks[2],
                "gt": [],
                "pred": [],
            }
        elif toks[0] == "GT" and current is not None:
            current["gt"].append(
                {
                    "box": tuple(map(float, toks[1:5])),
                    "class_id": int(toks[5]),
                    "name": toks[6],
                }
            )
        elif toks[0] == "PRED" and current is not None:
            current["pred"].append(
                {
                    "box": tuple(map(float, toks[1:5])),
                    "class_id": int(toks[5]),
                    "name": toks[6],
                    "score": float(toks[7]),
                }
            )
    if current is not None:
        samples.append(current)
    return samples, model_w, model_h


def scale_box(box, src_w, src_h, dst_w, dst_h):
    x1, y1, x2, y2 = box
    return (
        x1 * dst_w / src_w,
        y1 * dst_h / src_h,
        x2 * dst_w / src_w,
        y2 * dst_h / src_h,
    )


def draw_box(draw, box, color, label, width=3):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    # label background
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = label
    if font is not None:
        bbox = draw.textbbox((x1, max(0, y1 - 12)), text, font=font)
    else:
        bbox = (x1, max(0, y1 - 12), x1 + 8 * len(text), y1)
    draw.rectangle(bbox, fill=color)
    draw.text((bbox[0] + 1, bbox[1]), text, fill=(255, 255, 255), font=font)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="results/detections.txt")
    parser.add_argument(
        "--images-root",
        default="datasets/prepared/cars/valid",
        help="Directory containing images/ relative to report paths",
    )
    parser.add_argument("--out-dir", default="results/viz")
    args = parser.parse_args()

    report = Path(args.report)
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, model_w, model_h = parse_report(report)
    print(f"samples={len(samples)} model_input={model_w}x{model_h}")

    for sample in samples:
        img_path = images_root / sample["image_rel"]
        if not img_path.is_file():
            print(f"missing image: {img_path}")
            continue
        image = load_pgm(img_path)
        src_w, src_h = image.size
        draw = ImageDraw.Draw(image)

        # GT boxes in report are already in model input coords (after det resize).
        for gt in sample["gt"]:
            box = scale_box(gt["box"], model_w, model_h, src_w, src_h)
            draw_box(draw, box, (0, 200, 0), f"GT {gt['name']}", width=3)

        for pred in sample["pred"]:
            box = scale_box(pred["box"], model_w, model_h, src_w, src_h)
            label = f"{pred['name']} {pred['score']:.2f}"
            draw_box(draw, box, (220, 40, 40), label, width=2)

        out_path = out_dir / f"sample_{sample['index']:03d}.png"
        image.save(out_path)
        print(
            f"wrote {out_path}  gt={len(sample['gt'])} pred={len(sample['pred'])}"
        )

    # also write a simple index
    index = out_dir / "README.txt"
    index.write_text(
        "Green boxes = ground truth labels\n"
        "Red boxes   = model predictions (class + confidence)\n"
        "Model was trained one pass on cars train split, then evaluated on valid.\n",
        encoding="utf-8",
    )
    print(f"index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
