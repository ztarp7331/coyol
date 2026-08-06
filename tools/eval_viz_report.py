"""Evaluate a C viz_detect report with class-aware IoU metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


def iou(a: Box, b: Box) -> float:
    left, top = max(a.x1, b.x1), max(a.y1, b.y1)
    right, bottom = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def read_report(path: Path):
    samples = []
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "SAMPLE":
            if current is not None:
                samples.append(current)
            current = {"gt": [], "pred": []}
        elif fields[0] == "GT" and current is not None:
            current["gt"].append(Box(*map(float, fields[1:5]), 1.0, int(fields[5])))
        elif fields[0] == "PRED" and current is not None:
            current["pred"].append(
                Box(*map(float, fields[1:5]), float(fields[7]), int(fields[5]))
            )
    if current is not None:
        samples.append(current)
    return samples


def average_precision(samples, threshold: float, class_id: int | None = None) -> float:
    ranked = []
    total_truth = 0
    for sample_index, sample in enumerate(samples):
        truth = [box for box in sample["gt"] if class_id is None or box.class_id == class_id]
        total_truth += len(truth)
        for prediction in sample["pred"]:
            if class_id is None or prediction.class_id == class_id:
                ranked.append((prediction.score, sample_index, prediction))
    ranked.sort(key=lambda item: item[0], reverse=True)
    used = [[False] * len(sample["gt"]) for sample in samples]
    tp = fp = 0
    precision = []
    recall = []
    for _, sample_index, prediction in ranked:
        truth = samples[sample_index]["gt"]
        best = -1
        best_iou = threshold
        for index, target in enumerate(truth):
            if used[sample_index][index] or prediction.class_id != target.class_id:
                continue
            overlap = iou(prediction, target)
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best >= 0:
            used[sample_index][best] = True
            tp += 1
        else:
            fp += 1
        precision.append(tp / (tp + fp))
        recall.append(tp / total_truth if total_truth else 0.0)
    return sum(
        max((p for p, r in zip(precision, recall) if r >= level / 100.0), default=0.0)
        for level in range(101)
    ) / 101.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    samples = read_report(args.report)
    labels = [line.strip() for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    tp = fp = fn = 0
    for sample in samples:
        used = [False] * len(sample["gt"])
        for prediction in sorted(sample["pred"], key=lambda box: box.score, reverse=True):
            best, best_iou = -1, 0.5
            for index, target in enumerate(sample["gt"]):
                if used[index] or prediction.class_id != target.class_id:
                    continue
                overlap = iou(prediction, target)
                if overlap >= best_iou:
                    best, best_iou = index, overlap
            if best >= 0:
                used[best] = True
                tp += 1
            else:
                fp += 1
        fn += sum(not value for value in used)
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom else 0.0
    ap50 = average_precision(samples, 0.5)
    map50_95 = sum(average_precision(samples, threshold) for threshold in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)) / 10.0
    print(f"samples={len(samples)} gt={tp + fn} pred={tp + fp} tp={tp} fp={fp} fn={fn} f1={f1:.4f} ap50={ap50:.4f} map50_95={map50_95:.4f}")
    for class_id, label in enumerate(labels):
        print(f"class={label} ap50={average_precision(samples, 0.5, class_id):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
