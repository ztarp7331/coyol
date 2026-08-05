#!/usr/bin/env python3
"""Cleaner Phase D streams (no architecture changes).

Variants under datasets/prepared/:
  cars_plus_od/   cars train + car_od train (class 0->2), no forced empty BG
  cars_plus_bg/   cars train + hard empty backgrounds only
  cars_1c_merged/ 1-class merged stream (cars class-2 filter + car_od)

Valid for 5-class remains original cars/valid (copied).
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/c/Users/sayal/OneDrive/Documents/c-oloy")
CARS = ROOT / "datasets/prepared/cars"
CAR_OD = ROOT / "datasets/prepared/car_od"
CAR_CLASS = 2


def parse_manifest(path: Path, base: Path):
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        img = (base / parts[0]).resolve()
        rows.append((img, parts[1:]))
    return rows


def remap(boxes, cid):
    out = []
    for t in boxes:
        v = t.split(",")
        if len(v) != 5:
            continue
        v[4] = str(cid)
        out.append(",".join(v))
    return out


def write_stream(name: str, items, valid_rows):
    out = ROOT / "datasets/prepared" / name
    if out.exists():
        shutil.rmtree(out)
    tr = out / "train/images"
    va = out / "valid/images"
    tr.mkdir(parents=True)
    va.mkdir(parents=True)
    lines = []
    seen = set()
    for src, boxes in items:
        key = str(src)
        if key in seen or not src.is_file():
            continue
        seen.add(key)
        idx = len(lines)
        name_img = f"{idx:06d}.pgm"
        shutil.copy2(src, tr / name_img)
        rel = f"images/{name_img}"
        lines.append(rel if not boxes else rel + " " + " ".join(boxes))
    (out / "train/manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    vlines = []
    for i, (src, boxes) in enumerate(valid_rows):
        name_img = f"{i:06d}.pgm"
        shutil.copy2(src, va / name_img)
        rel = f"images/{name_img}"
        vlines.append(rel if not boxes else rel + " " + " ".join(boxes))
    (out / "valid/manifest.txt").write_text("\n".join(vlines) + "\n", encoding="utf-8")
    c = Counter()
    for _, boxes in items:
        for t in boxes:
            v = t.split(",")
            if len(v) == 5:
                c[int(float(v[4]))] += 1
    print(f"{name}: train={len(lines)} unique valid={len(vlines)} classes={dict(sorted(c.items()))}")


def main():
    cars_tr = parse_manifest(CARS / "train/manifest.txt", CARS / "train")
    cars_va = parse_manifest(CARS / "valid/manifest.txt", CARS / "valid")
    od_tr = parse_manifest(CAR_OD / "train/manifest.txt", CAR_OD / "train")
    od_va = parse_manifest(CAR_OD / "valid/manifest.txt", CAR_OD / "valid")

    plus_od = list(cars_tr) + [(p, remap(b, CAR_CLASS)) for p, b in od_tr]
    write_stream("cars_plus_od", plus_od, cars_va)

    plus_bg = list(cars_tr) + [(p, []) for p, _ in od_va]
    write_stream("cars_plus_bg", plus_bg, cars_va)

    # 1-class: cars class-2 only + car_od, all class 0
    one = []
    for p, boxes in cars_tr:
        kept = []
        for t in boxes:
            v = t.split(",")
            if len(v) == 5 and int(float(v[4])) == CAR_CLASS:
                v[4] = "0"
                kept.append(",".join(v))
        if kept:
            one.append((p, kept))
    for p, b in od_tr:
        one.append((p, remap(b, 0)))
    # 1-class valid from cars valid class-2 only
    one_va = []
    for p, boxes in cars_va:
        kept = []
        for t in boxes:
            v = t.split(",")
            if len(v) == 5 and int(float(v[4])) == CAR_CLASS:
                v[4] = "0"
                kept.append(",".join(v))
        if kept:
            one_va.append((p, kept))
    write_stream("cars_1c_expanded", one, one_va)
    print("DONE")


if __name__ == "__main__":
    main()
