#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch

from train import HEATMAP_SIZE, INPUT_SIZE, TrayCenterNet


class OnnxModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        output = self.model(images)
        return output["heatmap"], output["offset"], output["localizable"], output["occupancy"]


def main():
    parser = argparse.ArgumentParser(description="Export and verify TrayCenterNet ONNX")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/v6/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/v6/circle-center.onnx"))
    parser.add_argument("--trtexec", type=Path, default=shutil.which("trtexec"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("input_size") != INPUT_SIZE or checkpoint.get("heatmap_size") != HEATMAP_SIZE:
        raise ValueError("checkpoint input/heatmap size does not match train.py")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ONNX numerical verification")
    if args.trtexec is None or not args.trtexec.is_file():
        raise FileNotFoundError("trtexec not found; pass --trtexec /path/to/trtexec")

    model = OnnxModel(TrayCenterNet(pretrained=False)).cuda().eval()
    model.model.load_state_dict(checkpoint["model"])
    sample = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device="cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.output,
        input_names=["images"],
        output_names=["heatmap_logits", "offset_xy", "localizable_logit", "occupancy_logit"],
        dynamic_axes={name: {0: "batch"} for name in (
            "images", "heatmap_logits", "offset_xy", "localizable_logit", "occupancy_logit"
        )},
        opset_version=17,
    )

    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    with torch.no_grad():
        expected = tuple(value.cpu().numpy() for value in model(sample))

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        sample.cpu().numpy().tofile(directory / "input.bin")
        output_json = directory / "output.json"
        subprocess.run(
            [
                str(args.trtexec),
                f"--onnx={args.output}",
                f"--optShapes=images:1x3x{INPUT_SIZE}x{INPUT_SIZE}",
                f"--loadInputs=images:{directory / 'input.bin'}",
                f"--exportOutput={output_json}",
                "--warmUp=0",
                "--duration=0",
                "--iterations=1",
                "--noTF32",
            ],
            check=True,
        )
        values = {item["name"]: np.asarray(item["values"], dtype=np.float32) for item in json.loads(output_json.read_text())}
        actual = [values[name].reshape(value.shape) for name, value in zip(
            ("heatmap_logits", "offset_xy", "localizable_logit", "occupancy_logit"), expected
        )]
    for name, torch_value, onnx_value in zip(
        ("heatmap_logits", "offset_xy", "localizable_logit", "occupancy_logit"), expected, actual
    ):
        np.testing.assert_allclose(onnx_value, torch_value, rtol=5e-3, atol=3e-3)
        print(f"{name}: shape={onnx_value.shape} max_abs_diff={np.max(np.abs(onnx_value - torch_value)):.3g}")
    print(f"exported and verified on {torch.cuda.get_device_name()}: {args.output}")


if __name__ == "__main__":
    main()
