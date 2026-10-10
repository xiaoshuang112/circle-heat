#!/usr/bin/env python3
"""Snapshot reviewed full-frame LabelMe pairs and append them to fixed splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from train import crop_bbox, load_records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pairs(annotation):
    data = json.loads(annotation.read_text(encoding="utf-8"))
    name = data.get("imagePath")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError(f"{annotation}: imagePath must be a local filename")
    image_path = annotation.parent / name
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"{annotation}: unreadable image")
    if (data.get("imageHeight"), data.get("imageWidth")) != image.shape[:2]:
        raise ValueError(f"{annotation}: image dimensions disagree")
    groups = {}
    for shape in data.get("shapes", []):
        kind, label = shape.get("shape_type"), shape.get("label")
        if (kind, label) not in {("rectangle", "bbox"), ("circle", "empty"), ("circle", "occupied")}:
            raise ValueError(f"{annotation}: unexpected shape {kind}/{label}")
        points = np.asarray(shape.get("points"), dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            raise ValueError(f"{annotation}: invalid shape points")
        group = shape.get("group_id")
        if group is None or not isinstance(group, (str, int)):
            raise ValueError(f"{annotation}: pair requires group_id")
        pair = groups.setdefault(str(group), {})
        if kind in pair:
            raise ValueError(f"{annotation}: duplicate {kind} in group {group}")
        pair[kind] = shape
    if not groups:
        raise ValueError(f"{annotation}: no pairs")
    pairs = []
    for group, pair in groups.items():
        if set(pair) != {"rectangle", "circle"}:
            raise ValueError(f"{annotation}: incomplete pair {group}")
        corners = np.asarray(pair["rectangle"]["points"], dtype=float)
        bbox = [*corners.min(axis=0).tolist(), *corners.max(axis=0).tolist()]
        x, y = map(float, pair["circle"]["points"][0])
        crop_bbox(image, x, y, bbox)
        radius = float(np.linalg.norm(np.asarray(pair["circle"]["points"])[1] - [x, y]))
        if radius <= 0:
            raise ValueError(f"{annotation}: circle has zero radius")
        pairs.append(dict(slot_id=group, bbox=bbox, center_x=x, center_y=y,
                          occupancy=pair["circle"]["label"], annotation_radius=radius))
    return image_path, image, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--reviewed", action="store_true", help="explicit human approval of all labels")
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("human-reviewed labels required; prediction flags are not approval")
    baseline = load_records(args.baseline)
    heldout = load_records(args.heldout)
    annotations = sorted(args.source.rglob("*.json"))
    if not annotations:
        parser.error("no annotations")
    selected, duplicates = {}, []
    for annotation in annotations:
        image_path, image, pairs = read_pairs(annotation)
        key = hashlib.sha256(image.tobytes()).hexdigest()
        entry = (annotation, image_path, pairs)
        if key in selected:
            old = selected[key]
            if annotation.stat().st_mtime_ns == old[0].stat().st_mtime_ns and pairs != old[2]:
                raise ValueError("duplicate pixels with conflicting labels and tied modification times")
            keep, skip = sorted([old, entry], key=lambda e: e[0].stat().st_mtime_ns, reverse=True)
            selected[key] = keep
            duplicates.append(dict(kept=str(keep[0]), excluded=str(skip[0]), rule="latest_saved_annotation"))
        else:
            selected[key] = entry
    # All scenes in this acquisition batch stay in train; historical evaluation stays frozen.
    validation_pixels = set()
    for records, root in [(baseline, args.baseline.parent), (heldout, args.heldout.parent)]:
        for record in records:
            if record["split"] != "val":
                continue
            image = cv2.imread(str(root / record["image"]))
            if image is None:
                raise ValueError(f"unreadable validation image: {record['image']}")
            validation_pixels.add(hashlib.sha256(image.tobytes()).hexdigest())
    for annotation, image_path, pairs in selected.values():
        image = cv2.imread(str(image_path))
        for pair in pairs:
            crop, *_ = crop_bbox(image, pair["center_x"], pair["center_y"], pair["bbox"])
            if hashlib.sha256(crop.tobytes()).hexdigest() in validation_pixels:
                raise ValueError(f"{annotation}: crop duplicates a validation/heldout image")
    fit = args.output.with_name(args.output.stem + "-new-trainfit.jsonl")
    report = args.output.with_suffix(".audit.json")
    if any(p.exists() for p in [args.snapshot, args.output, fit, report]):
        raise FileExistsError("snapshot/output already exists; use fresh paths")
    args.snapshot.mkdir(parents=True)
    new = []
    for annotation, image_path, pairs in selected.values():
        relative = annotation.relative_to(args.source)
        dest = args.snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(annotation, dest)
        shutil.copy2(image_path, dest.parent / image_path.name)
        for pair in pairs:
            new.append(dict(image=str((dest.parent / image_path.name).relative_to(args.output.parent)),
                            source_image=str(relative.with_suffix("")), split_group=args.snapshot.name,
                            split="train", label_state="localizable", human_reviewed=True,
                            image_sha256=sha(image_path), annotation_sha256=sha(annotation), **pair))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_text = args.baseline.read_text(encoding="utf-8")
    new_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
    args.output.write_text(old_text.rstrip("\n") + "\n" + new_text, encoding="utf-8")
    fit.write_text(new_text, encoding="utf-8")
    audit = dict(source=str(args.source), baseline=str(args.baseline), baseline_sha256=sha(args.baseline),
                 heldout_sha256=sha(args.heldout), human_reviewed=True, duplicate_resolution=duplicates,
                 images_received=len(annotations), images_kept=len(selected), new_samples=len(new),
                 new_classes=dict(Counter(r["occupancy"] for r in new)),
                 total_splits=dict(Counter(r["split"] for r in baseline + new)),
                 crop_jitter="new bbox records only: 50% unchanged, 50% translation up to +/-3%; val unchanged",
                 inputs={str(p): sha(p) for a, im, _ in selected.values() for p in (a, im)},
                 output_sha256=sha(args.output), snapshot=str(args.snapshot))
    report.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_records(args.output)
    print(json.dumps({k: v for k, v in audit.items() if k != "inputs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
