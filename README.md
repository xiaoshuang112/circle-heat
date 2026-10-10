# circle-heat

V5 单格位圆心与空满检测。输入 ROI 等比例 letterbox 到 320×320；模型输出 80×80 圆心热图、`localizable` 和 `empty/occupied`。

网络结构及输入尺寸定义在 `model.py`，`train.py` 负责数据、损失和训练，`export_onnx.py` 直接复用模型定义。

雷赛项目是整图多目标定位：`train_leisai.py` 复用同一骨干和特征金字塔，输出两个 80×80 热图，分别检测 `luosi`（螺丝）和 `heikong`（黑孔）。它使用独立的训练入口和检查点，不改变上面的单格位任务。

## 数据

最新人工复核后的训练清单是 `data/labels-user-checked62-train_and_fixed_val-20261004T155441_652481Z.jsonl`：1418 个 ROI（train 1202、val 216）。新增样本的独立验证清单是 `data/labels-user-checked62-new_heldout82-20261004T155441_652481Z.jsonl`（82 个 ROI），训练样本拟合检查清单是 `data/labels-user-checked62-new_trainfit360-20261004T155441_652481Z.jsonl`（360 个 ROI）。这 82 个独立验证样本没有加入训练。同一 `split_group` 不跨 train/val。

复核后的难例清单 `data/labels-user-checked62-hard-*.jsonl` 保留在原路径，三个子集分别为固定验证难例 10 个、新增独立验证难例 10 个、训练难例 42 个；它们是上述数据的子集，不要重复加入训练。旧场景与新场景的原始验证清单仍为 `data/labels-v5.jsonl` 和 `data/labels-new-algorithm-reviewed.jsonl`。

清单中的 `image` 路径相对于清单所在的 `data/`；原图位于 `data/raw/` 或 `test_inputs/`，因体积较大不纳入 Git。训练前需将原图放到对应路径。LabelMe 标注转换可使用 `prepare_labelme.py`；circle 的第一个 point 是圆心。

2026-10-10 将 8 份旧导入清单及类别修正快照移至 `data/archive/20261010-v5-manifests/`，原始内容未改。归档目录包含清单说明与 `SHA256SUMS`，仅保存在本地。复现历史训练前，按归档目录的 `README.md` 将需要的清单恢复到 `data/`；不要直接用归档路径训练，否则图片的相对路径会失效。最新 3 份数据清单、3 份难例清单及雷赛自动标注清单均保留原路径。

## 训练

```bash
uv sync
.venv/bin/python train.py --self-check
.venv/bin/python -u train.py \
  --labels data/labels-user-checked62-train_and_fixed_val-20261004T155441_652481Z.jsonl \
  --output runs/v5-next \
  --epochs 100 --batch-size 16 --heatmap-sigma 1.0
```

训练从 ImageNet 预训练权重开始。上述命令的最佳模型保存为 `runs/v5-next/best.pt`。当前最新人工复核版权重为 `runs/v5-reviewed-labelme442-userchecked62-20261004T155939_187923Z/best.pt`，对应 ONNX 为同目录下的 `circle-center.onnx`。`runs/` 不纳入 Git。

## 导出 ONNX

```bash
uv sync --extra onnx
.venv/bin/python export_onnx.py \
  --checkpoint runs/v5-reviewed/best.pt \
  --output runs/v5-reviewed/circle-center.onnx
```

导出脚本执行 ONNX 结构检查。输出名称为 `heatmap_logits`、`localizable_logit`、`occupancy_logit`。

## 整图格位预标注

新图片使用当前算法生成整图 LabelMe 标注、固定 50 px 圆半径并拷贝到 97 的流程，见 [整图格位自动标注操作文档](docs/FULL_FRAME_LABELME.md)。以后可直接让 AI 读取该文档并提供图片目录。

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

## 新增数据自动标注

`fine.zip` 含 7 张人工标注图和一批未标注 PNG。自动标注脚本跳过已有人类标注的图片，以及旧包中逐像素重复的那张图；对其余图片生成带类别、圆心和置信度的 `data/leisai-auto-labels.jsonl`。完整的六目标图只有在数量、置信度（最低 0.4）和圆形位置检查都通过时，才标记为 `accepted`。裁切图的候选也保留在清单中，但不会自动进入训练。

```bash
.venv/bin/python auto_label_leisai.py \
  --source data/raw/leisai/fine.zip \
  --exclude-archive data/raw/leisai/train-0927-1.zip \
  --checkpoint runs/leisai-expanded-v2/best.pt \
  --output data/leisai-auto-labels.jsonl
.venv/bin/python -u train_leisai.py \
  --archive data/raw/leisai/train-0927-1.zip \
  --extra-archive data/raw/leisai/fine.zip \
  --auto-labels data/leisai-auto-labels.jsonl \
  --init-checkpoint runs/leisai-expanded-v2/best.pt \
  --output runs/leisai-expanded-v3 \
  --epochs 20 --batch-size 16
```

当前清单有 153 张候选图，其中 142 张通过检查、11 张需要复核。训练时人工标注重复采样 4 次，验证集只用人工标注；这些图像来自相近拍摄场景，验证指标不能代替独立现场测试。

`runs/leisai-cycle/` 的 205 张 PNG 已复制到 `data/raw/leisai/cycle.zip`。这批图有四螺丝和四螺丝加两黑孔两种布局；`--layout cycle` 先检查螺丝的方形位置，再用局部暗圆对比度判断对角黑孔。清单 `data/leisai-cycle-auto-labels.jsonl` 包含全部 205 张的候选点，其中 201 张通过检查并进入训练，4 张因裁切掉一颗螺丝而保留为待复核候选。黑孔的 `confidence` 是图像对比度分数，不是网络概率。

供人工检查的 `runs/leisai-cycle-labeled.zip` 包含 205 张 PNG、201 份同名 LabelMe JSON 和一份待复核名单；每个 JSON 使用 `circle` 标注螺丝或黑孔中心，并通过 `imagePath` 引用同目录图片。

```bash
.venv/bin/python auto_label_leisai.py \
  --source data/raw/leisai/cycle.zip \
  --checkpoint runs/leisai-cycle-198/best.pt \
  --layout cycle \
  --output data/leisai-cycle-auto-labels.jsonl
.venv/bin/python -u train_leisai.py \
  --archive data/raw/leisai/train-0927-1.zip \
  --extra-archive data/raw/leisai/fine.zip \
  --auto-labels data/leisai-auto-labels.jsonl \
  --auto-labels data/leisai-cycle-auto-labels.jsonl \
  --init-checkpoint runs/leisai-cycle-198/best.pt \
  --output runs/leisai-cycle-final \
  --epochs 20 --batch-size 16
```

2026-10-07 的 `center-net-leisai/dataset/crop` 人工修正标注已快照为 `data/raw/leisai/crop-reviewed-20261007.zip`，含 116 张图、623 个圆心。新增 91 张训练图和 25 张验证图，同一数字编号的裁切保持在同一集合。原有 45 张验证图继续保留，使用 `--validation-names` 指定固定验证集合（每行一个图片 stem）。本次使用 `--selection-metric f1` 按检测 F1 选择权重，同分时比较验证损失；省略该参数时继续按验证损失选取。所有划分、输入哈希和训练参数保存在 `runs/leisai-validated-1007/training-config.json`。

```bash
.venv/bin/python -u train_leisai.py \
  --archive data/raw/leisai/train-0927-1.zip \
  --extra-archive data/raw/leisai/fine.zip \
  --extra-archive data/raw/leisai/label-json-0929-oil-fixed-20261008.zip \
  --extra-archive data/raw/leisai/crop-reviewed-20261007.zip \
  --auto-labels data/leisai-auto-labels.jsonl \
  --validation-names runs/leisai-validated-1007/validation-names.txt \
  --selection-metric f1 \
  --init-checkpoint runs/leisai-validated-0929/best.pt \
  --output runs/leisai-validated-1007 \
  --epochs 20 --batch-size 16
.venv/bin/python export_onnx.py \
  --checkpoint runs/leisai-validated-1007/best.pt \
  --output runs/leisai-validated-1007/leisai-centers.onnx
```

2026-10-08 的 `center-net-leisai/dataset/fail2/crop` 校对结果已快照为 `data/raw/leisai/fail2-reviewed-20261008.zip`，含 4 张图、21 个圆心，全部加入训练。继续训练时，在上述训练命令中添加 `--extra-archive data/raw/leisai/fail2-reviewed-20261008.zip`，初始化权重改为 `runs/leisai-validated-1007/best.pt`，输出目录改为 `runs/leisai-validated-1008`。沿用原 70 张固定验证图；这 4 张图的指标单独记为 `fail2_train_fit`，属于训练样本表现。训练配置、划分和输入哈希保存在新输出目录的 `training-config.json`。

经用户确认，`0927_00076` 左上方 `(81.14, 27.70)` 的黑色物体是机油，已删除其 `heikong` 误标。修正版本为 `data/raw/leisai/label-json-0929-oil-fixed-20261008.zip`，该图保留 4 个 `luosi` 和 2 个 `heikong`。最新数据配置已指向修正版本，重新评估后固定 70 张验证图为 0 漏检、4 误检；原训练配置和修正前指标保留供追溯。该图属于固定验证集，训练样本内容未改变。
