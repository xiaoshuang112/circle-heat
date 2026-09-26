# circle-heat

V5 单格位圆心与空满检测。输入 ROI 等比例 letterbox 到 320×320；模型输出 80×80 圆心热图、`localizable` 和 `empty/occupied`。

网络结构及输入尺寸定义在 `model.py`，`train.py` 负责数据、损失和训练，`export_onnx.py` 直接复用模型定义。

## 数据

最新训练清单是 `data/labels-v5-plus-reviewed-new.jsonl`：1033 个 ROI（train 817、val 216）。旧场景与新场景的验证清单分别是 `data/labels-v5.jsonl` 和 `data/labels-new-algorithm-reviewed.jsonl`。同一 `split_group` 不跨 train/val。

清单中的 `image` 路径相对于 `data/`；原图位于 `data/raw/`，因体积较大不纳入 Git。训练前需将原图放到对应路径。LabelMe 标注转换可使用 `prepare_labelme.py`；circle 的第一个 point 是圆心。

## 训练

```bash
uv sync
.venv/bin/python train.py --self-check
.venv/bin/python -u train.py \
  --labels data/labels-v5-plus-reviewed-new.jsonl \
  --output runs/v5-reviewed \
  --epochs 100 --batch-size 16 --heatmap-sigma 1.0
```

训练从 ImageNet 预训练权重开始。最佳模型保存为 `runs/v5-reviewed/best.pt`。`runs/` 不纳入 Git。

## 导出 ONNX

```bash
uv sync --extra onnx
.venv/bin/python export_onnx.py \
  --checkpoint runs/v5-reviewed/best.pt \
  --output runs/v5-reviewed/circle-center.onnx
```

导出脚本执行 ONNX 结构检查。输出名称为 `heatmap_logits`、`localizable_logit`、`occupancy_logit`。
