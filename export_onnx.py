#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx
import torch

from model import HEATMAP_SIZE, INPUT_SIZE, TrayCenterNet


class OnnxModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        output = self.model(images)
        return output["heatmap"], output["localizable"], output["occupancy"]


def main():
    parser = argparse.ArgumentParser(description="Export and verify V5 TrayCenterNet ONNX")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/v5-reviewed/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/v5-reviewed/circle-center.onnx"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("input_size") != INPUT_SIZE or checkpoint.get("heatmap_size") != HEATMAP_SIZE:
        raise ValueError("checkpoint input/heatmap size does not match model.py")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ONNX numerical verification")

    model = OnnxModel(TrayCenterNet(pretrained=False)).cuda().eval()
    model.model.load_state_dict(checkpoint["model"])
    sample = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device="cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = ("heatmap_logits", "localizable_logit", "occupancy_logit")
    torch.onnx.export(
        model,
        sample,
        args.output,
        input_names=["images"],
        output_names=list(names),
        dynamic_axes={name: {0: "batch"} for name in ("images", *names)},
        opset_version=17,
    )

    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    print(f"exported and ONNX-checked: {args.output}")


if __name__ == "__main__":
    main()
