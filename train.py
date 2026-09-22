#!/usr/bin/env python3
import argparse
import json
import math
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

INPUT_SIZE = 320
HEATMAP_SIZE = 80
STRIDE = INPUT_SIZE // HEATMAP_SIZE
PAD_VALUE = 114
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            record["_line"] = line_number
            records.append(record)

    if not records:
        raise ValueError(f"{path}: no labels found")

    groups: dict[str, str] = {}
    for record in records:
        state = record.get("label_state")
        split = record.get("split")
        if state not in {"localizable", "reject", "unreviewed"}:
            raise ValueError(f"{path}:{record['_line']}: invalid label_state {state!r}")
        if state != "unreviewed" and split not in {"train", "val"}:
            raise ValueError(f"{path}:{record['_line']}: split must be train or val")
        if record.get("occupancy", "unknown") not in {"empty", "occupied", "unknown"}:
            raise ValueError(f"{path}:{record['_line']}: invalid occupancy")
        if state == "localizable" and not {"center_x", "center_y"} <= record.keys():
            raise ValueError(f"{path}:{record['_line']}: localizable sample needs center_x/center_y")
        group = record.get("split_group") or record.get("source_image")
        if group and split in {"train", "val"}:
            previous = groups.setdefault(str(group), split)
            if previous != split:
                raise ValueError(f"group {group!r} appears in both train and val")
    return records


def letterbox(image: np.ndarray, x: float | None, y: float | None):
    height, width = image.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (INPUT_SIZE - new_width) // 2
    top = (INPUT_SIZE - new_height) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized
    scale_x, scale_y = new_width / width, new_height / height
    if x is not None:
        x = (x + 0.5) * scale_x - 0.5 + left
        y = (y + 0.5) * scale_y - 0.5 + top
    meta = np.array([scale_x, scale_y, left, top], dtype=np.float32)
    return canvas, x, y, meta


def gaussian_target(x: float, y: float, sigma: float = 1.5) -> torch.Tensor:
    center_x = (x + 0.5) / STRIDE - 0.5
    center_y = (y + 0.5) / STRIDE - 0.5
    grid_y, grid_x = torch.meshgrid(
        torch.arange(HEATMAP_SIZE, dtype=torch.float32),
        torch.arange(HEATMAP_SIZE, dtype=torch.float32),
        indexing="ij",
    )
    target = torch.exp(-((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2 * sigma**2))
    return target / target.sum()


class SlotDataset(Dataset):
    def __init__(self, labels: Path, split: str, augment: bool = False, heatmap_sigma: float = 1.0):
        self.root = labels.parent
        self.records = [
            record
            for record in load_records(labels)
            if record.get("split") == split and record["label_state"] != "unreviewed"
        ]
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        if not self.records:
            raise ValueError(f"no {split!r} samples in {labels}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record["image"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        localizable = record["label_state"] == "localizable"
        x = float(record["center_x"]) if localizable else None
        y = float(record["center_y"]) if localizable else None
        if localizable and not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            raise ValueError(f"{path}: center ({x}, {y}) is outside the ROI")
        image, x, y, meta = letterbox(image, x, y)

        if self.augment:
            dx, dy = random.randint(-8, 8), random.randint(-8, 8)
            image = cv2.warpAffine(
                image,
                np.float32([[1, 0, dx], [0, 1, dy]]),
                (INPUT_SIZE, INPUT_SIZE),
                borderValue=(PAD_VALUE,) * 3,
            )
            meta[2:] += (dx, dy)
            if localizable:
                x, y = x + dx, y + dy
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-12, 12)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        tensor = image.astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1))
        heatmap = gaussian_target(x, y, self.heatmap_sigma) if localizable else torch.zeros(HEATMAP_SIZE, HEATMAP_SIZE)
        occupancy = {"empty": 0.0, "occupied": 1.0}.get(record.get("occupancy"), -1.0)
        return {
            "image": tensor,
            "heatmap": heatmap,
            "localizable": torch.tensor(float(localizable)),
            "occupancy": torch.tensor(occupancy),
            "center": torch.tensor([x or 0.0, y or 0.0]),
            "meta": torch.from_numpy(meta),
        }


class DepthwiseBlock(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(),
        )


class TrayCenterNet(nn.Module):
    feature_indices = (1, 3, 8, 12)
    feature_channels = (16, 24, 48, 576)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.lateral = nn.ModuleList([nn.Conv2d(channels, 64, 1) for channels in self.feature_channels])
        self.refine = nn.ModuleList([DepthwiseBlock(64) for _ in range(3)])
        self.heatmap_head = nn.Conv2d(64, 1, 1)
        self.localizable_head = nn.Linear(576, 1)
        self.occupancy_head = nn.Linear(576, 1)

    def forward(self, image):
        features = []
        for index, layer in enumerate(self.backbone):
            image = layer(image)
            if index in self.feature_indices:
                features.append(image)
        pyramid = self.lateral[-1](features[-1])
        for level in range(2, -1, -1):
            pyramid = F.interpolate(pyramid, size=features[level].shape[-2:], mode="bilinear", align_corners=False)
            pyramid = self.refine[level](pyramid + self.lateral[level](features[level]))
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        return {
            "heatmap": self.heatmap_head(pyramid).squeeze(1),
            "localizable": self.localizable_head(pooled).squeeze(1),
            "occupancy": self.occupancy_head(pooled).squeeze(1),
        }


def compute_loss(output, batch):
    localizable = batch["localizable"].bool()
    logits = output["heatmap"].float().flatten(1)
    targets = batch["heatmap"].float().flatten(1)
    if localizable.any():
        center_loss = -(targets[localizable] * F.log_softmax(logits[localizable], dim=1)).sum(1).mean()
    else:
        center_loss = logits.sum() * 0
    localizable_loss = F.binary_cross_entropy_with_logits(output["localizable"].float(), batch["localizable"])
    occupancy_known = batch["occupancy"] >= 0
    if occupancy_known.any():
        occupancy_loss = F.binary_cross_entropy_with_logits(
            output["occupancy"][occupancy_known].float(), batch["occupancy"][occupancy_known]
        )
    else:
        occupancy_loss = output["occupancy"].sum() * 0
    return center_loss + 0.2 * localizable_loss + 0.2 * occupancy_loss


def decode_centers(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits.flatten(1).float(), dim=1).reshape_as(logits)
    centers = []
    for heatmap in probabilities:
        peak = int(heatmap.argmax())
        peak_y, peak_x = divmod(peak, HEATMAP_SIZE)
        x0, x1 = max(0, peak_x - 2), min(HEATMAP_SIZE, peak_x + 3)
        y0, y1 = max(0, peak_y - 2), min(HEATMAP_SIZE, peak_y + 3)
        patch = heatmap[y0:y1, x0:x1]
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=heatmap.device),
            torch.arange(x0, x1, device=heatmap.device),
            indexing="ij",
        )
        weight = patch.sum().clamp_min(1e-12)
        heat_x = (patch * xx).sum() / weight
        heat_y = (patch * yy).sum() / weight
        centers.append(torch.stack(((heat_x + 0.5) * STRIDE - 0.5, (heat_y + 0.5) * STRIDE - 0.5)))
    return torch.stack(centers)


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    if training:
        for module in model.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    losses, errors, occupancy_correct, occupancy_total = [], [], 0, 0
    localizable_correct, sample_total = 0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp = device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            output = model(batch["image"])
        loss = compute_loss(output, batch)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(loss.detach().item())

        predicted = decode_centers(output["heatmap"])
        mask = batch["localizable"].bool()
        if mask.any():
            scale = batch["meta"][mask, :2]
            padding = batch["meta"][mask, 2:]
            predicted_roi = (predicted[mask] - padding + 0.5) / scale - 0.5
            target_roi = (batch["center"][mask] - padding + 0.5) / scale - 0.5
            errors.extend(torch.linalg.vector_norm(predicted_roi - target_roi, dim=1).detach().cpu().tolist())
        known = batch["occupancy"] >= 0
        occupancy_correct += ((output["occupancy"][known] >= 0) == batch["occupancy"][known].bool()).sum().item()
        occupancy_total += known.sum().item()
        localizable_correct += ((output["localizable"] >= 0) == mask).sum().item()
        sample_total += len(mask)

    p95 = float(np.percentile(errors, 95)) if errors else math.inf
    return {
        "loss": float(np.mean(losses)),
        "center_mean_px": float(np.mean(errors)) if errors else math.inf,
        "center_p95_px": p95,
        "occupancy_acc": occupancy_correct / occupancy_total if occupancy_total else math.nan,
        "localizable_acc": localizable_correct / sample_total,
    }


def self_check():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        image = np.full((120, 200, 3), 180, np.uint8)
        cv2.circle(image, (80, 45), 12, (20, 20, 20), 2)
        cv2.imwrite(str(root / "sample.png"), image)
        (root / "labels.jsonl").write_text(
            json.dumps(
                {
                    "image": "sample.png",
                    "split": "train",
                    "center_x": 80.0,
                    "center_y": 45.0,
                    "label_state": "localizable",
                    "occupancy": "empty",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        sample = SlotDataset(root / "labels.jsonl", "train")[0]
        scale, padding = sample["meta"][:2], sample["meta"][2:]
        round_trip = (sample["center"] - padding + 0.5) / scale - 0.5
        assert torch.allclose(round_trip, torch.tensor([80.0, 45.0]), atol=1e-4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TrayCenterNet(pretrained=False).to(device)
        output = model(sample["image"].unsqueeze(0).to(device))
        assert output["heatmap"].shape == (1, HEATMAP_SIZE, HEATMAP_SIZE)
        batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items()}
        loss = compute_loss(output, batch)
        loss.backward()
        assert torch.isfinite(loss)
    print(f"self-check passed on {device}: letterbox, model forward and loss backward")


def format_metrics(name: str, metrics: dict) -> str:
    values = " ".join(
        f"{key}={value:.4f}" if math.isfinite(value) else f"{key}=n/a" for key, value in metrics.items()
    )
    return f"{name} {values}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/v0"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--heatmap-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.labels is None:
        parser.error("--labels is required unless --self-check is used")
    if args.heatmap_sigma <= 0:
        parser.error("--heatmap-sigma must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = SlotDataset(args.labels, "train", augment=not args.overfit, heatmap_sigma=args.heatmap_sigma)
    validation_data = train_data if args.overfit else SlotDataset(args.labels, "val", heatmap_sigma=args.heatmap_sigma)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda"
    )

    model = TrayCenterNet(pretrained=args.pretrained).to(device)
    backbone_parameters = list(model.backbone.parameters())
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_parameters, "lr": 0.0}, {"params": head_parameters, "lr": 1e-3}], weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for parameter in backbone_parameters:
        parameter.requires_grad_(False)

    args.output.mkdir(parents=True, exist_ok=True)
    best = math.inf
    stale = 0
    print(f"device={device} train={len(train_data)} val={len(validation_data)} heatmap_sigma={args.heatmap_sigma}")
    for epoch in range(args.epochs):
        if epoch == min(args.freeze_epochs, args.epochs):
            for parameter in backbone_parameters:
                parameter.requires_grad_(True)
        if epoch < args.freeze_epochs:
            optimizer.param_groups[0]["lr"] = 0.0
            optimizer.param_groups[1]["lr"] = 1e-3
        else:
            progress = (epoch - args.freeze_epochs) / max(1, args.epochs - args.freeze_epochs - 1)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            optimizer.param_groups[0]["lr"] = 1e-6 + (1e-4 - 1e-6) * cosine
            optimizer.param_groups[1]["lr"] = 1e-6 + (3e-4 - 1e-6) * cosine

        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, device)
        print(f"epoch={epoch + 1:03d} {format_metrics('train', train_metrics)}")
        print(f"epoch={epoch + 1:03d} {format_metrics('val', validation_metrics)}")

        score = validation_metrics["center_p95_px"]
        if score < best:
            best, stale = score, 0
            torch.save(
                {"model": model.state_dict(), "input_size": INPUT_SIZE, "heatmap_size": HEATMAP_SIZE},
                args.output / "best.pt",
            )
        else:
            stale += 1
            if not args.overfit and stale >= 20:
                print("early stop: center P95 did not improve for 20 epochs")
                break


if __name__ == "__main__":
    main()
