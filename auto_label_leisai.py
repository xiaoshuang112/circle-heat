#!/usr/bin/env python3
"""Generate checked pseudo-labels for unannotated images in a LabelMe zip."""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
import torch

from model import TrayCenterNet
from train import IMAGENET_MEAN, IMAGENET_STD, letterbox
from train_leisai import CLASSES, decode


def check_detections(name, detections, width, height):
    stem = PurePosixPath(name).stem
    expected = (4, 0) if stem.startswith("Image_") or stem in {"left", "right"} else (4, 2)
    counts = tuple(sum(point[2] == category for point in detections) for category in range(2))
    if counts != expected:
        return f"count {counts}, expected {expected}"
    if min(point[3] for point in detections) < 0.4:
        return "low confidence"
    if min(math.dist(a[:2], b[:2]) for i, a in enumerate(detections) for b in detections[i + 1:]) < 0.15 * min(width, height):
        return "centers too close"
    if expected[1]:
        center = np.mean([point[:2] for point in detections], axis=0)
        vectors = [np.asarray(point[:2]) - center for point in detections]
        radii = np.linalg.norm(vectors, axis=1)
        holes = [vectors[i] for i, point in enumerate(detections) if point[2] == 1]
        opposition = float(np.dot(holes[0], holes[1]) / (np.linalg.norm(holes[0]) * np.linalg.norm(holes[1])))
        if radii.std() / radii.mean() > 0.1 or opposition > -0.85:
            return "irregular circle geometry"
    return "accepted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/leisai/fine.zip"))
    parser.add_argument("--exclude-archive", action="append", type=Path, default=[], help="skip images already human-labeled in another zip")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/leisai/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/leisai-auto-labels.jsonl"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        points = [(100 + 60 * math.cos(i * math.pi / 3), 100 + 60 * math.sin(i * math.pi / 3),
                   int(i in (0, 3)), 0.8) for i in range(6)]
        assert check_detections("flange.png", points, 200, 200) == "accepted"
        assert check_detections("flange.png", points[:-1], 200, 200).startswith("count")
        print("self-check passed: count and circle geometry checks")
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(checkpoint.get("classes", ())) != CLASSES:
        parser.error(f"checkpoint classes must be {CLASSES}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrayCenterNet(pretrained=False, heatmap_channels=len(CLASSES)).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    records = []
    annotated = set()
    for path in args.exclude_archive:
        with ZipFile(path) as archive:
            annotated.update(PurePosixPath(name).stem for name in archive.namelist() if name.endswith(".json"))
    with ZipFile(args.source) as archive:
        annotated.update(PurePosixPath(name).stem for name in archive.namelist() if name.endswith(".json"))
        names = sorted(name for name in archive.namelist() if name.endswith(".png")
                       and PurePosixPath(name).stem not in annotated)
        for start in range(0, len(names), 16):
            images, metas, sizes = [], [], []
            for name in names[start:start + 16]:
                image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"cannot decode {name}")
                height, width = image.shape[:2]
                canvas, _, _, meta = letterbox(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None, None)
                tensor = (canvas.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
                images.append(torch.from_numpy(tensor.transpose(2, 0, 1)))
                metas.append(torch.from_numpy(meta))
                sizes.append((width, height))
            with torch.no_grad():
                outputs = model(torch.stack(images).to(device))["heatmap"].cpu()
            for name, output, meta, (width, height) in zip(names[start:start + 16], outputs, metas, sizes):
                detections = []
                for point in decode(output, meta):
                    if (0 <= point[0] < width and 0 <= point[1] < height
                            and all(math.dist(point[:2], prior[:2]) > max(8, 0.035 * min(width, height)) for prior in detections)):
                        detections.append(point)
                reason = check_detections(name, detections, width, height)
                records.append({
                    "archive": os.path.relpath(args.source, args.output.parent),
                    "teacher_checkpoint": str(args.checkpoint),
                    "image": name,
                    "width": width,
                    "height": height,
                    "accepted": reason == "accepted",
                    "reason": reason,
                    "points": [{"x": x, "y": y, "label": CLASSES[category], "confidence": score}
                               for x, y, category, score in detections],
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(f"wrote {len(records)} candidates to {args.output}")
    print("quality:", dict(Counter(record["reason"] for record in records)))


if __name__ == "__main__":
    main()
