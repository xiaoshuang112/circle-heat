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

最优权重保存为输出目录下的 `best.pt`。当前只准备训练闭环；自动预标脚本等 V0 和真实标签格式验证后再接。
