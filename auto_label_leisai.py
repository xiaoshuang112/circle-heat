#!/usr/bin/env python3
"""Generate checked pseudo-labels for unannotated images in a LabelMe zip."""

import argparse
import itertools
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


def select_cycle_detections(detections, image=None):
    squares = []
    for indices in itertools.combinations(range(len(detections)), 4):
        points = [detections[i] for i in indices]
        xy = np.asarray([point[:2] for point in points])
        center = xy.mean(axis=0)
        vectors = xy - center
        radii = np.linalg.norm(vectors, axis=1)
        if radii.min() < 1:
            continue
        angles = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi]) * 180 / math.pi
        radial_error = radii.std() / radii.mean()
        angular_error = max(abs(gaps - 90))
        if radial_error < 0.08 and angular_error < 16 and min(point[3] for point in points) >= 0.35:
            penalty = 2 * radial_error + angular_error / 90 + 0.08 * sum(point[2] == 1 for point in points)
            squares.append((penalty, indices, center, radii.mean()))
    if not squares:
        return detections, "no four-screw square"
    _, indices, center, radius = min(squares, key=lambda item: item[0])
    screws = [detections[i] for i in indices]
    if min(point[3] for point in screws) < 0.4:
        return detections, "low screw confidence"
    selected = [(x, y, 0, score) for x, y, _, score in screws]
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        response = cv2.GaussianBlur(gray, (0, 0), 7) - cv2.GaussianBlur(gray, (0, 0), 2)
        angles = sorted(math.atan2(y - center[1], x - center[0]) for x, y, *_ in screws)
        search_radius = max(8, round(0.15 * radius))
        holes = []
        for i, angle in enumerate(angles):
            midway = angle + ((angles[(i + 1) % 4] - angle) % (2 * math.pi)) / 2
            x, y = center + radius * np.array([math.cos(midway), math.sin(midway)])
            x0, x1 = max(0, round(x) - search_radius), min(image.shape[1], round(x) + search_radius + 1)
            y0, y1 = max(0, round(y) - search_radius), min(image.shape[0], round(y) + search_radius + 1)
            if x0 >= x1 or y0 >= y1:
                return detections, "hole search outside image"
            _, peak, _, (px, py) = cv2.minMaxLoc(response[y0:y1, x0:x1])
            holes.append((x0 + px, y0 + py, peak))
        pair = max(((0, 2), (1, 3)), key=lambda pair: min(holes[i][2] for i in pair))
        score = min(holes[i][2] for i in pair)
        # ponytail: this contrast cutoff fits this camera batch; recalibrate for other lighting.
        if score >= 45:
            selected.extend((holes[i][0], holes[i][1], 1, min(0.99, holes[i][2] / 100)) for i in pair)
        return selected, "accepted"
    ring = []
    for i, point in enumerate(detections):
        if i not in indices and point[3] >= 0.4:
            vector = np.asarray(point[:2]) - center
            if abs(np.linalg.norm(vector) - radius) / radius < 0.12:
                ring.append((point, vector))
    if len(ring) not in (0, 2):
        return detections, f"{len(ring)} possible black holes"
    if ring:
        a, b = ring[0][1], ring[1][1]
        if np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) > -0.85:
            return detections, "black holes not opposite"
    selected.extend((point[0], point[1], 1, point[3]) for point, _ in ring)
    return selected, "accepted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/leisai/fine.zip"))
    parser.add_argument("--exclude-archive", action="append", type=Path, default=[], help="skip images already human-labeled in another zip")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/leisai/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/leisai-auto-labels.jsonl"))
    parser.add_argument("--layout", choices=("legacy", "cycle"), default="legacy")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        points = [(100 + 60 * math.cos(i * math.pi / 3), 100 + 60 * math.sin(i * math.pi / 3),
                   int(i in (0, 3)), 0.8) for i in range(6)]
        assert check_detections("flange.png", points, 200, 200) == "accepted"
        assert check_detections("flange.png", points[:-1], 200, 200).startswith("count")
        screws = [(100 + 60 * math.cos(i * math.pi / 2), 100 + 60 * math.sin(i * math.pi / 2), 0, 0.8)
                  for i in range(4)]
        holes = [(100 + 60 * math.cos(i * math.pi / 4), 100 + 60 * math.sin(i * math.pi / 4), 0, 0.7)
                 for i in (1, 5)]
        assert len(select_cycle_detections(screws)[0]) == 4
        selected, reason = select_cycle_detections(screws + holes)
        assert reason == "accepted" and [point[2] for point in selected].count(1) == 2
        assert select_cycle_detections(screws[:3])[1] != "accepted"
        image = np.full((200, 200, 3), 180, dtype=np.uint8)
        assert len(select_cycle_detections(screws, image)[0]) == 4
        for i in (1, 5):
            angle = i * math.pi / 4
            cv2.circle(image, (round(100 + 60 * math.cos(angle)), round(100 + 60 * math.sin(angle))), 6, (10, 10, 10), -1)
        selected, reason = select_cycle_detections(screws, image)
        assert reason == "accepted" and [point[2] for point in selected].count(1) == 2
        print("self-check passed: count, circle and cycle layout checks")
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
            images, originals, metas, sizes = [], [], [], []
            for name in names[start:start + 16]:
                image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"cannot decode {name}")
                height, width = image.shape[:2]
                canvas, _, _, meta = letterbox(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None, None)
                tensor = (canvas.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
                images.append(torch.from_numpy(tensor.transpose(2, 0, 1)))
                originals.append(image)
                metas.append(torch.from_numpy(meta))
                sizes.append((width, height))
            with torch.no_grad():
                outputs = model(torch.stack(images).to(device))["heatmap"].cpu()
            for name, image, output, meta, (width, height) in zip(names[start:start + 16], originals, outputs, metas, sizes):
                detections = []
                for point in decode(output, meta):
                    if (0 <= point[0] < width and 0 <= point[1] < height
                            and all(math.dist(point[:2], prior[:2]) > max(8, 0.035 * min(width, height)) for prior in detections)):
                        detections.append(point)
                if args.layout == "cycle":
                    detections, reason = select_cycle_detections(detections, image)
                else:
                    reason = check_detections(name, detections, width, height)
                records.append({
                    "archive": os.path.relpath(args.source, args.output.parent),
                    "teacher_checkpoint": str(args.checkpoint),
                    "layout": args.layout,
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
