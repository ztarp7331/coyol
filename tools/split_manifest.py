"""Create deterministic class-aware train/validation manifest splits."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path


def classes_in(line: str) -> set[int]:
    return {int(token.split(",")[4]) for token in line.split()[1:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("validation fraction must be in (0,1)")

    lines = [line + "\n" for line in args.manifest.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) < 2:
        raise SystemExit("manifest needs at least two samples")
    labels = [classes_in(line) for line in lines]
    totals = Counter(label for sample in labels for label in sample)
    desired = {label: max(1, round(count * args.validation_fraction))
               for label, count in totals.items()}
    validation_target = max(1, min(len(lines) - 1,
                                   round(len(lines) * args.validation_fraction)))

    order = list(range(len(lines)))
    random.Random(args.seed).shuffle(order)
    order.sort(key=lambda index: min((totals[label] for label in labels[index]), default=0))
    validation: set[int] = set()
    selected = Counter()
    for index in order:
        if len(validation) >= validation_target:
            break
        if any(selected[label] < desired[label] for label in labels[index]):
            validation.add(index)
            selected.update(labels[index])
    for index in order:
        if len(validation) >= validation_target:
            break
        validation.add(index)

    train_path = args.manifest.with_name("manifest_train.txt")
    valid_path = args.manifest.with_name("manifest_valid.txt")
    train_path.write_text("".join(line for index, line in enumerate(lines)
                                  if index not in validation), encoding="utf-8")
    valid_path.write_text("".join(line for index, line in enumerate(lines)
                                  if index in validation), encoding="utf-8")
    print(f"train={len(lines) - len(validation)} validation={len(validation)} "
          f"train_manifest={train_path} validation_manifest={valid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
