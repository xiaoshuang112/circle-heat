# circle-heat

V5 单格位圆心与空满检测。输入 ROI 等比例 letterbox 到 320×320；模型输出 80×80 圆心热图、`localizable` 和 `empty/occupied`。

网络结构及输入尺寸定义在 `model.py`，`train.py` 负责数据、损失和训练，`export_onnx.py` 直接复用模型定义。

雷赛项目是整图多目标定位：`train_leisai.py` 复用同一骨干和特征金字塔，输出两个 80×80 热图，分别检测 `luosi`（螺丝）和 `heikong`（黑孔）。它使用独立的训练入口和检查点，不改变上面的单格位任务。

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

## 雷赛项目训练

`data/raw/leisai/train-0927-1.zip` 内的 LabelMe JSON 可直接读取；脚本优先使用 JSON 内嵌图像，所以缺少同名 PNG 的标注也能训练。按是否含黑孔分层留出验证图，默认训练 11 张、验证 2 张。验证集很小，指标只能用于检查训练流程，不能代表现场精度。

```bash
.venv/bin/python train_leisai.py --self-check
.venv/bin/python -u train_leisai.py \
  --archive data/raw/leisai/train-0927-1.zip \
  --init-checkpoint runs/v5-reviewed/best.pt \
  --output runs/leisai
```

`--init-checkpoint` 可省略；省略时使用 ImageNet 预训练骨干。最佳验证损失的模型保存为 `runs/leisai/best.pt`，与原项目的 `best.pt` 分开。推理时对两个热图分别做局部峰值检测；`train_leisai.py` 中的 `decode` 提供了坐标还原示例。

雷赛模型可用同一导出脚本生成 ONNX，输出 `heatmap_logits` 的形状为 `[batch, 2, 80, 80]`；通道顺序是 `luosi`、`heikong`：

```bash
.venv/bin/python export_onnx.py \
  --checkpoint runs/leisai/best.pt \
  --output runs/leisai/leisai-centers.onnx
```
