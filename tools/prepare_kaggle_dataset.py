"""Convert downloaded XML detection datasets into C manifest/PGM format."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def image_for_xml(xml_path: Path, filename: str) -> Path:
    name = Path(filename).name
    candidates = [xml_path.parent / name]
    stem = Path(name).stem
    candidates.extend(xml_path.parent / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image for {xml_path}: {name}")


def read_voc(root: Path):
    records = []
    for xml_path in sorted(root.rglob("*.xml")):
        document = ET.parse(xml_path).getroot()
        filename = document.findtext("filename")
        if not filename:
            continue
        image_path = image_for_xml(xml_path, filename)
        boxes = []
        for obj in document.findall("object"):
            label = (obj.findtext("name") or "unknown").strip()
            box = obj.find("bndbox")
            if box is None:
                continue
            values = [box.findtext(key) for key in ("xmin", "ymin", "xmax", "ymax")]
            if any(value is None for value in values):
                continue
            boxes.append((*map(float, values), label))
        if boxes:
            records.append((image_path, boxes))
    return records


def read_cvat(root: Path):
    document = ET.parse(root / "annotations.xml").getroot()
    records = []
    for image in document.findall(".//image"):
        name = image.attrib["name"]
        candidates = [root / name, root / "boxes" / Path(name).name]
        image_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f"image for CVAT record: {name}")
        boxes = []
        for box in image.findall("box"):
            label = box.attrib.get("label", "unknown")
            boxes.append(tuple(float(box.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr")) + (label,))
        if boxes:
            records.append((image_path, boxes))
    return records


def write_manifest(records, output: Path):
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted({box[4] for _, boxes in records for box in boxes})
    class_ids = {label: index for index, label in enumerate(labels)}
    lines = []
    for index, (source, boxes) in enumerate(records):
        target = image_dir / f"{index:06d}.pgm"
        with Image.open(source) as image:
            width, height = image.size
            image.convert("L").save(target)
        tokens = []
        for x1, y1, x2, y2, label in boxes:
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            tokens.append(f"{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f},{class_ids[label]}")
        if not tokens:
            continue
        lines.append(f"images/{target.name} {' '.join(tokens)}\n")
    (output / "manifest.txt").write_text("".join(lines), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (output / "source_files.json").write_text(
        json.dumps({"images": len(records), "classes": labels}, indent=2) + "\n",
        encoding="utf-8",
    )
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("voc", "cvat"), required=True)
    args = parser.parse_args()
    records = read_voc(args.source) if args.format == "voc" else read_cvat(args.source)
    if not records:
        raise SystemExit("no annotated images found")
    args.output.mkdir(parents=True, exist_ok=True)
    labels = write_manifest(records, args.output)
    print(f"images={len(records)} classes={labels} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
