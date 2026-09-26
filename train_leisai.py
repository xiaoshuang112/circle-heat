#!/usr/bin/env python3
"""Train a two-class, multi-center heatmap on the Leisai LabelMe archive."""

import argparse
import base64
import json
import random
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import HEATMAP_SIZE, INPUT_SIZE, TrayCenterNet
from train import IMAGENET_MEAN, IMAGENET_STD, PAD_VALUE, letterbox

CLASSES = ("luosi", "heikong")
STRIDE = INPUT_SIZE // HEATMAP_SIZE
GRID_Y, GRID_X = torch.meshgrid(torch.arange(HEATMAP_SIZE), torch.arange(HEATMAP_SIZE), indexing="ij")


def read_archive(path: Path):
    samples = []
    with ZipFile(path) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".json")):
            annotation = json.loads(archive.read(name))
            raw = annotation.get("imageData")
            if raw:
                image_bytes = base64.b64decode(raw)
            else:
                image_name = annotation.get("imagePath", "")
                if PurePosixPath(image_name).name != image_name:
                    raise ValueError(f"{name}: invalid imagePath")
                image_bytes = archive.read(str(PurePosixPath(name).parent / image_name))
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (annotation.get("imageHeight"), annotation.get("imageWidth")):
                raise ValueError(f"{name}: missing image or image size mismatch")
            points = []
            for shape in annotation.get("shapes", []):
                if shape.get("shape_type") != "circle" or shape.get("label") not in CLASSES:
                    raise ValueError(f"{name}: unsupported shape {shape.get('label')!r}")
                x, y = map(float, shape["points"][0])
                if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                    raise ValueError(f"{name}: circle center outside image")
                points.append((x, y, CLASSES.index(shape["label"])))
            if not points:
                raise ValueError(f"{name}: no circles")
            samples.append((PurePosixPath(name).stem, image, points))
    if not samples:
        raise ValueError(f"{path}: no LabelMe annotations")
    return samples


def read_auto_labels(path: Path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted = [record for record in records if record["accepted"]]
    if not accepted:
        raise ValueError(f"{path}: no accepted pseudo-labels")
    archives = {record["archive"] for record in accepted}
    if len(archives) != 1:
        raise ValueError(f"{path}: expected one source archive")
    samples = []
    with ZipFile(path.parent / archives.pop()) as archive:
        for record in accepted:
            image = cv2.imdecode(np.frombuffer(archive.read(record["image"]), np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (record["height"], record["width"]):
                raise ValueError(f"{record['image']}: image size mismatch")
            points = []
            for point in record["points"]:
                x, y, label = float(point["x"]), float(point["y"]), point["label"]
                if label not in CLASSES or not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                    raise ValueError(f"{record['image']}: invalid pseudo-label")
                points.append((x, y, CLASSES.index(label)))
            samples.append((PurePosixPath(record["image"]).stem, image, points))
    return samples


def split_samples(samples, seed: int, val_fraction: float):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = random.Random(seed)
    train, val = [], []
    for has_hole in (False, True):
        group = [sample for sample in samples if any(p[2] == 1 for p in sample[2]) == has_hole]
        rng.shuffle(group)
        count = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        val.extend(group[:count])
        train.extend(group[count:])
    if not val or not train:
        raise ValueError("need at least two annotated images for a train/val split")
    return train, val


class LeisaiDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        _, bgr, points = self.samples[index]
        image, _, _, meta = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None, None)
        dx, dy = (random.randint(-8, 8), random.randint(-8, 8)) if self.augment else (0, 0)
        if self.augment:
            image = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (INPUT_SIZE, INPUT_SIZE), borderValue=(PAD_VALUE,) * 3)
            image = np.clip(image.astype(np.float32) * random.uniform(0.9, 1.1) + random.uniform(-12, 12), 0, 255).astype(np.uint8)
        meta[2:] += (dx, dy)
        target = torch.zeros(len(CLASSES), HEATMAP_SIZE, HEATMAP_SIZE)
        for x, y, category in points:
            px = (x + 0.5) * meta[0] - 0.5 + meta[2]
            py = (y + 0.5) * meta[1] - 0.5 + meta[3]
            hx, hy = round((px + 0.5) / STRIDE - 0.5), round((py + 0.5) / STRIDE - 0.5)
            if 0 <= hx < HEATMAP_SIZE and 0 <= hy < HEATMAP_SIZE:
                gaussian = torch.exp(-((GRID_X - hx) ** 2 + (GRID_Y - hy) ** 2) / (2 * 1.5**2))
                target[category] = torch.maximum(target[category], gaussian)
        image = (image.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(image.transpose(2, 0, 1)), target, points, torch.from_numpy(meta)


def focal_loss(logits, target):
    probability = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    positive = target == 1
    positive_loss = -(positive * (1 - probability).square() * probability.log())
    negative_loss = -((~positive) * (1 - target).pow(4) * probability.square() * torch.log1p(-probability))
    counts = positive.sum(dim=(0, 2, 3))
    return ((positive_loss + negative_loss).sum(dim=(0, 2, 3)) / counts.clamp_min(1)).mean()


def decode(logits, meta, threshold=0.3):
    probability = logits.float().sigmoid()
    peaks = (probability == F.max_pool2d(probability[None], 3, 1, 1)[0]) & (probability >= threshold)
    detections = []
    for category, y, x in peaks.nonzero().tolist():
        px, py = (x + 0.5) * STRIDE - 0.5, (y + 0.5) * STRIDE - 0.5
        detections.append((((px - meta[2] + 0.5) / meta[0] - 0.5).item(),
                           ((py - meta[3] + 0.5) / meta[1] - 0.5).item(),
                           category, probability[category, y, x].item()))
    return sorted(detections, key=lambda item: -item[3])


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    for module in model.backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    losses = []
    tp, fp, fn = [0] * len(CLASSES), [0] * len(CLASSES), [0] * len(CLASSES)
    for batch in loader:
        images = torch.stack([sample[0] for sample in batch]).to(device)
        targets = torch.stack([sample[1] for sample in batch]).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)["heatmap"]
            loss = focal_loss(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        if not training:
            for prediction, (_, _, points, meta) in zip(logits.detach().cpu(), batch):
                matched = set()
                for x, y, category, _ in decode(prediction, meta):
                    candidates = [((x - gx) ** 2 + (y - gy) ** 2, i) for i, (gx, gy, c) in enumerate(points)
                                  if c == category and i not in matched]
                    nearest = min(candidates) if candidates else None
                    if nearest and nearest[0] <= 10**2:
                        matched.add(nearest[1])
                        tp[category] += 1
                    else:
                        fp[category] += 1
                for i, (_, _, category) in enumerate(points):
                    fn[category] += i not in matched
    return float(np.mean(losses)), tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/leisai/train-0927-1.zip"))
    parser.add_argument("--extra-archive", action="append", type=Path, default=[], help="additional human-labeled LabelMe zip")
    parser.add_argument("--auto-labels", type=Path, help="accepted pseudo-label manifest from auto_label_leisai.py")
    parser.add_argument("--output", type=Path, default=Path("runs/leisai"))
    parser.add_argument("--init-checkpoint", type=Path, help="optional V5 checkpoint for backbone/pyramid initialization")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        model = TrayCenterNet(pretrained=False, heatmap_channels=2)
        output = model(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))["heatmap"]
        target = torch.zeros_like(output)
        target[0, 0, 10, 20] = target[0, 1, 30, 40] = 1
        loss = focal_loss(output, target)
        loss.backward()
        assert output.shape == (1, 2, HEATMAP_SIZE, HEATMAP_SIZE) and torch.isfinite(loss)
        assert len(decode(target[0] * 20 - 10, torch.tensor([1., 1., 0., 0.]))) == 2
        print("self-check passed: two-class forward, loss backward and multi-peak decode")
        return
    if args.epochs < 1 or args.batch_size < 1 or args.freeze_epochs < 0:
        parser.error("epochs and batch-size must be positive; freeze-epochs cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manual_samples = read_archive(args.archive)
    for archive in args.extra_archive:
        manual_samples.extend(read_archive(archive))
    train_samples, val_samples = split_samples(manual_samples, args.seed, args.val_fraction)
    manual_count = len(train_samples)
    auto_count = 0
    if args.auto_labels:
        auto_samples = read_auto_labels(args.auto_labels)
        if {sample[0] for sample in auto_samples} & {sample[0] for sample in manual_samples}:
            raise ValueError("pseudo-labels overlap human-labeled images")
        auto_count = len(auto_samples)
        train_samples = train_samples * 4 + auto_samples
    train_loader = DataLoader(LeisaiDataset(train_samples, augment=True), batch_size=args.batch_size, shuffle=True, collate_fn=list)
    val_loader = DataLoader(LeisaiDataset(val_samples), batch_size=args.batch_size, collate_fn=list)
    model = TrayCenterNet(pretrained=args.init_checkpoint is None, heatmap_channels=2)
    nn.init.constant_(model.heatmap_head.bias, -2.19)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)["model"]
        if state["heatmap_head.weight"].shape[0] == 1:
            state = {key: value for key, value in state.items() if not key.startswith("heatmap_head.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or set(missing) not in (set(), {"heatmap_head.weight", "heatmap_head.bias"}):
            raise ValueError(f"incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    print(f"device={device} manual_train={manual_count} auto_train={auto_count} val={len(val_samples)} classes={CLASSES}")
    print("validation images:", ", ".join(sample[0] for sample in val_samples))
    for epoch in range(args.epochs):
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(epoch >= args.freeze_epochs)
        train_loss, _, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, tp, fp, fn = run_epoch(model, val_loader, device)
        precision = sum(tp) / max(1, sum(tp) + sum(fp))
        recall = [tp[i] / max(1, tp[i] + fn[i]) for i in range(len(CLASSES))]
        print(f"epoch={epoch+1:03d} train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
              f"precision={precision:.3f} luosi_recall={recall[0]:.3f} heikong_recall={recall[1]:.3f}", flush=True)
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "classes": CLASSES, "input_size": INPUT_SIZE, "heatmap_size": HEATMAP_SIZE}, args.output / "best.pt")


if __name__ == "__main__":
    main()
