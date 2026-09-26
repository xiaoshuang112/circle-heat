#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert one-circle LabelMe annotations to training JSONL")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    annotations = sorted(args.source.glob("*.json"))
    if not annotations:
        parser.error(f"no JSON annotations in {args.source}")

    samples = []
    for annotation in annotations:
        data = json.loads(annotation.read_text(encoding="utf-8"))
        shapes = data.get("shapes", [])
        if len(shapes) != 1:
            raise ValueError(f"{annotation}: expected exactly one shape, got {len(shapes)}")
        shape = shapes[0]
        label = shape.get("label")
        points = shape.get("points", [])
        if label not in {"empty", "occupied"} or shape.get("shape_type") != "circle" or len(points) < 1:
            raise ValueError(f"{annotation}: expected an empty/occupied circle with at least one point")

        image_name = data.get("imagePath")
        if not image_name or Path(image_name).name != image_name:
            raise ValueError(f"{annotation}: invalid imagePath {image_name!r}")
        image = args.source / image_name
        if not image.is_file():
            raise FileNotFoundError(image)
        center_x, center_y = map(float, points[0])
        width, height = data.get("imageWidth"), data.get("imageHeight")
        if not (width and height and 0 <= center_x < width and 0 <= center_y < height):
            raise ValueError(f"{annotation}: center {points[0]} is outside {width}x{height}")

        frame = annotation.stem.split("__", 1)[0]
        match = re.match(r"^(production-[0-9a-f-]+)-v\d+-", frame)
        group = match.group(1) if match else frame
        samples.append((group, frame, image, label, center_x, center_y))

    groups = sorted({sample[0] for sample in samples})
    random.Random(args.seed).shuffle(groups)
    val_count = max(1, round(len(groups) * args.val_fraction))
    validation_groups = set(groups[:val_count])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for group, frame, image, label, center_x, center_y in samples:
            split = "val" if group in validation_groups else "train"
            counts[(split, label)] += 1
            record = {
                "image": str(image.relative_to(args.output.parent)),
                "source_image": frame,
                "split_group": group,
                "split": split,
                "center_x": center_x,
                "center_y": center_y,
                "label_state": "localizable",
                "occupancy": label,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    for split in ("train", "val"):
        if not all(counts[(split, label)] for label in ("empty", "occupied")):
            raise ValueError(f"{split} split does not contain both labels; change --seed")
    print(f"wrote {len(samples)} samples from {len(groups)} groups to {args.output}")
    print(" ".join(f"{split}/{label}={counts[(split, label)]}" for split in ("train", "val") for label in ("empty", "occupied")))


if __name__ == "__main__":
    main()
