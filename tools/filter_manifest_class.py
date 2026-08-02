#!/usr/bin/env python3
"""Filter a manifest to one class and optionally remap class ids to 0."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-class", type=int, required=True)
    parser.add_argument("--remap-to", type=int, default=0)
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop images that have no remaining boxes",
    )
    args = parser.parse_args()

    lines_out: list[str] = []
    for raw in Path(args.input).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        image = parts[0]
        kept = []
        for token in parts[1:]:
            vals = token.split(",")
            if len(vals) != 5:
                continue
            class_id = int(float(vals[4]))
            if class_id != args.keep_class:
                continue
            vals[4] = str(args.remap_to)
            kept.append(",".join(vals))
        if not kept and args.drop_empty:
            continue
        lines_out.append(image if not kept else image + " " + " ".join(kept))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    print(f"wrote {len(lines_out)} lines -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
