# circle-heat

首版训练环境：单格位 ROI 输入，经等比例 `letterbox` 到 `320×320`，输出 `80×80` 中心热图、`localizable` 和 `empty/occupied`。

## 数据契约

把当前算法的**原始格位框直接裁成图片，不扩大、不拉伸**。标签写入 `data/labels.jsonl`，一行一个 JSON；图片路径相对标签文件所在目录。

必填字段：

- `image`：ROI 图片路径。
- `split`：`train` 或 `val`。首轮建议约 40/20 格位。
- `label_state`：`localizable`、`reject` 或 `unreviewed`；后者不会参与训练。
- `occupancy`：`empty`、`occupied` 或 `unknown`。
- `center_x`、`center_y`：仅 `localizable` 必填，坐标基于**未 letterbox 的 ROI 原图**。

推荐同时保存 `source_image`、`split_group/session_id`、`tray_id`、`slot_id`。同一整图或同一采集组不能跨 train/val；训练程序会检查这个问题。格式见 [data/labels.example.jsonl](data/labels.example.jsonl)。

## 运行

```bash
.venv/bin/python train.py --self-check

# 先确认 40 个训练格位能够被模型记住（关闭增强，train 同时作为检查集）
.venv/bin/python train.py --labels data/labels.jsonl --overfit --epochs 100 --output runs/overfit

# 再用显式 train/val 划分训练 V0
.venv/bin/python train.py --labels data/labels.jsonl --epochs 100 --output runs/v0
```

当前 `.venv` 复用本机已有的 CUDA PyTorch。需要在其他机器重建时运行 `uv sync`。

LabelMe 圆标注先转换为训练清单；circle 的第一个 point 作为中心，并按文件名中的采集 UUID 隔离 train/val：

```bash
.venv/bin/python prepare_labelme.py data/raw/train-0921 data/labels.jsonl
```

最优权重保存为输出目录下的 `best.pt`。当前只准备训练闭环；自动预标脚本等 V0 和真实标签格式验证后再接。

当前 V2 数据已处理好，可直接启动训练：

```bash
./train_v2.sh
```

需要临时覆盖参数时直接追加，例如 `./train_v2.sh --epochs 150 --batch-size 8`。

V3 只把 heatmap sigma 从 1.5 改为 1.0，用于和 V2 做单变量对照：

```bash
./train_v3.sh
```

V3 训练完成后导出并校验 ONNX：

```bash
python -m venv --system-site-packages .onnx-venv
uv pip install --python .onnx-venv/bin/python "onnx>=1.14"
.onnx-venv/bin/python export_onnx.py --trtexec /path/to/TensorRT/bin/trtexec
```

隔离环境避免改动训练依赖；脚本会同时执行 ONNX 结构校验和 PyTorch CUDA/TensorRT GPU 数值校验。V6 输入保持不变，输出为 `heatmap_logits [B,80,80]`、`offset_xy [B,2,80,80]`、`localizable_logit [B]`、`occupancy_logit [B]`。V6 训练使用 offset head 对 peak cell 内的连续偏移做 SmoothL1 监督：

```bash
./train_v6.sh
.onnx-venv/bin/python export_onnx.py --checkpoint runs/v6/best.pt --output runs/v6/circle-center.onnx --trtexec /path/to/TensorRT/bin/trtexec
```
