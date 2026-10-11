# circle-heat

V5 单格位圆心与空满检测。输入 ROI 等比例 letterbox 到 320×320；模型输出 80×80 圆心热图、`localizable` 和 `empty/occupied`。

网络结构及输入尺寸定义在 `model.py`，`train.py` 负责数据、损失和训练，`export_onnx.py` 直接复用模型定义。

雷赛项目是整图多目标定位：`train_leisai.py` 复用同一骨干和特征金字塔，输出两个 80×80 热图，分别检测 `luosi`（螺丝）和 `heikong`（黑孔）。它使用独立的训练入口和检查点，不改变上面的单格位任务。

## 数据

最新人工复核后的训练清单是 `data/labels-user-checked62-train_and_fixed_val-20261004T155441_652481Z.jsonl`：1418 个 ROI（train 1202、val 216）。新增样本的独立验证清单是 `data/labels-user-checked62-new_heldout82-20261004T155441_652481Z.jsonl`（82 个 ROI），训练样本拟合检查清单是 `data/labels-user-checked62-new_trainfit360-20261004T155441_652481Z.jsonl`（360 个 ROI）。这 82 个独立验证样本没有加入训练。同一 `split_group` 不跨 train/val。

复核后的难例清单 `data/labels-user-checked62-hard-*.jsonl` 保留在原路径，三个子集分别为固定验证难例 10 个、新增独立验证难例 10 个、训练难例 42 个；它们是上述数据的子集，不要重复加入训练。旧场景与新场景的原始验证清单仍为 `data/labels-v5.jsonl` 和 `data/labels-new-algorithm-reviewed.jsonl`。

清单中的 `image` 路径相对于清单所在的 `data/`；原图位于 `data/raw/` 或 `test_inputs/`，因体积较大不纳入 Git。真实标注、`data/` 下的训练 JSONL 清单和导入审计文件也只保存在本地，不纳入 Git；仓库仅保留 `data/labels.example.jsonl` 作为格式示例。训练前需将清单和原图放到对应路径。LabelMe 标注转换可使用 `prepare_labelme.py`；circle 的第一个 point 是圆心。

2026-10-10 将 8 份旧导入清单及类别修正快照移至 `data/archive/20261010-v5-manifests/`，原始内容未改。归档目录包含清单说明与 `SHA256SUMS`，仅保存在本地。复现历史训练前，按归档目录的 `README.md` 将需要的清单恢复到 `data/`；不要直接用归档路径训练，否则图片的相对路径会失效。最新 3 份数据清单、3 份难例清单及雷赛自动标注清单均保留原路径。

## 训练

2026-10-10 的整图人工修正数据已加入 `data/labels-v5-fullframe-reviewed-20261010.jsonl`：训练 1245、固定验证 216。新批次原有 10 张图，其中 app_images/teaching 的同名图像素完全重复，按用户要求以最后保存的 teaching 标注为准；保留 9 张图、43 个格位，全部归入同一个训练组。随后用户确认此前归为分类误判的 8 个格位实际为 occupied，已同步修正 Downloads 原标注、训练快照、复核副本和两份当前清单，随后 `v3-2` 的 `group_id=2` 也经用户确认改为 occupied，当前分布为 empty 27、occupied 16。82 个外部留出样本不变。快照在 `data/raw/fullframe-reviewed-20261010/`，导入与修改记录在同名 `.audit.json`。本轮已经完成的模型仍是用修正前的标签训练；旧清单和标注备份在本轮 runs 目录的 `labels-before-occupied-fix-*` 中，历史 `comparison.json` 不覆盖。

`prepare_fullframe_labelme.py` 使用 `group_id` 严格匹配 bbox 与 circle，并校验图像尺寸、坐标、类别及重复像素；只有显式传入 `--reviewed` 才允许导入。它不会修改 Downloads 中的原始标注。清单含 `bbox` 的新增样本在训练时有 50% 概率保持原裁剪、50% 概率平移 bbox（每轴最多框尺寸的 3%，约 18–19 原图像素），从整图真实像素取样，圆心与逆变换同步。旧 ROI 和验证集不使用此 bbox 抖动；原有轻微平移/亮度增强继续保留。

本轮命令如下；新增 43 个格位的 `-new-trainfit.jsonl` 仅用于拟合检查，不能当独立测试集。

训练增强：沿用亮度偏移 ±12、对比度倍率 0.9–1.1 和画布平移 ±8 px；新增 25% 概率的 3×3 高斯模糊（sigma 0.3–0.8），以及 35% 概率的轻微透视扰动（角点每轴最多移动短边的 1.5%，320 输入约 4.8 px）。圆心与图像同步变换，越界时放弃该变换，训练像素误差先逆透视再还原到原图坐标。固定验证和外部留出均不增强；网络、loss、推理与 ONNX 接口不变。

```bash
.venv/bin/python -u train.py \
  --labels data/labels-v5-fullframe-reviewed-20261010.jsonl \
  --output runs/v5-fullframe-reviewed-20261010 \
  --epochs 100 --batch-size 16 --workers 0 --freeze-epochs 5 \
  --heatmap-sigma 1.0 --seed 7 --pretrained
```

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

## Replicator 合成数据样例

`generate_replicator.py` 在 Isaac Sim 中使用 Replicator 的 RGB annotator 和 orchestrator 渲染简化 3×3 托盘：深色圆盘代表空槽，圆柱代表工件。每帧随机空满、工件轻微偏移、相机位置、光照、灰度与材质粗糙度。随脚本提供的 `replicator.kit` 只加载渲染所需扩展，启用合成数据所需的 3 帧历史缓存；渲染使用 64 次采样的路径追踪和 OptiX 降噪，避免实时模式在本机输出明显噪声。此样例用于验证接入流程；空槽并非真实凹腔，相机安装位置、托盘和工件尺寸是假设值，尚未复现现场外观。

2026-10-10 从 90 只读复制的内参保存在本地 `data/raw/replicator-calibration-20261010/camera_intrinsics_current.yaml`，来源为 `nvidia@192.168.1.90:/home/nvidia/Downloads/remote-camera-intrinsics-20261010/camera_intrinsics_current.yaml`，两端 SHA256 都为 `339a716f65b5f88c7139623ac96fbcfbbe9845469b07fc6b9414b4372b4ffb8a`。内参原始分辨率为 2448×2048；脚本支持这份 OpenCV YAML 的 K 和五参数畸变模型。渲染使用 `OmniLensDistortionOpenCvPinholeAPI`，标签使用同一 K/D 投影；不会把检测框中心当作圆心。缩放分辨率时同步缩放 K，畸变系数不变。内参不包含相机外参，不能确定安装位置。

以下命令须在安装有兼容 Isaac Sim 的 RTX 主机上执行，替换 `ISAAC_SIM` 为其安装目录；项目 `.venv` 没有 Isaac Sim。需要该运行环境中的 NumPy、Pillow 和 OpenCV。不要为此改动现有训练环境。

```bash
"$ISAAC_SIM/python.sh" /home/irobotic/work/circle-heat/generate_replicator.py \
  --intrinsics /home/irobotic/work/circle-heat/data/raw/replicator-calibration-20261010/camera_intrinsics_current.yaml \
  --baseline /home/irobotic/work/circle-heat/data/labels-v5-fullframe-reviewed-20261010.jsonl \
  --output /home/irobotic/work/circle-heat/data/raw/replicator-demo-20261010 \
  --frames 20 --width 768 --seed 7
```

默认高度按内参宽高比计算，此命令生成 768×643 的完整画面、180 个格位样本。输出目录必须不存在；中途失败的目录保留供诊断，重试时换新目录。省略 `--baseline` 可以只生成样例，省略 `--intrinsics` 则使用假设的无畸变针孔相机。

输出包括 `labelme/` 下的完整 PNG 与同名 JSON、仅含合成训练样本的 `synthetic.jsonl`、可选的合并清单 `labels.jsonl`，以及参数与每帧几何记录 `summary.json`。合成标注是几何投影标签，`synthetic=true`、`human_reviewed=false`；LabelMe 的一框一圆共享 `group_id`，圆半径固定 50 px，空槽圆心在深色圆盘中心，工件圆心在圆柱上表面中心。bbox 是三维格位区域投影出的 ROI，尚未模拟生产检测器的框误差。样例限制相机视角并且不加入遮挡，不能用来训练 `reject` 或代表不可定位难例。

合成数据全部归入 train，合并清单保留原有真实数据的划分与其他标签，仅将其图片路径转为绝对路径，不覆盖原清单或原图。固定真实验证集和外部留出集不加入合成图。生成后先检查渲染画面与圆心，再用新的输出目录做小规模训练对比：

```bash
.venv/bin/python -m unittest -v test_replicator test_fullframe_bbox
.venv/bin/python -u train.py \
  --labels data/raw/replicator-demo-20261010/labels.jsonl \
  --output runs/v5-replicator-demo-20261010 \
  --epochs 100 --batch-size 16 --workers 0 --freeze-epochs 5 \
  --heatmap-sigma 1.0 --seed 7 --pretrained
```

2026-10-10 重启后，本机 RTX 3060 12GB 的内核驱动与 NVML 均为 595.99.02，版本不一致的问题已消除。另建的 Python 3.12 渲染环境位于 `/home/irobotic/.local/share/circle-heat-isaacsim/.venv`，使用 Isaac Sim 6.1.0.0 的启动组件和 Replicator 1.13.36，未改动项目训练环境。这是用于此脚本的渲染环境，未验证完整 Isaac Sim 机器人功能。首次运行已经用户确认接受 NVIDIA 许可；扩展和首次编译的着色器保存在本机缓存中。NVIDIA 下载器在本机曾长时间等待，首次下载临时改用标准 HTTPS 后已恢复其原文件；`replicator.kit` 使用内置 urllib 下载选项。本机直连 NVIDIA CDN 明显快于现有代理，以下命令仅对该进程移除代理变量。

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  OMNI_KIT_ACCEPT_EULA=YES NO_PROXY='*' no_proxy='*' \
  /home/irobotic/.local/share/circle-heat-isaacsim/.venv/bin/python -u generate_replicator.py \
  --intrinsics data/raw/replicator-calibration-20261010/camera_intrinsics_current.yaml \
  --baseline data/labels-v5-fullframe-reviewed-20261010.jsonl \
  --output data/raw/replicator-demo-20261010 \
  --frames 20 --width 768 --seed 7
```

实际验证结果保存在 `data/raw/replicator-pathtraced-verified-20261010/`：3 张 768×643 PNG、27 个格位样本（12 empty、15 occupied），缓存齐全后运行约 14 秒，正常退出且无 Error 日志。27 个合成样本均通过现有 `SlotDataset` 读取与坐标还原，合并清单为 1272 train、216 val，原 1461 条真实记录除图片路径转为绝对路径外完全一致。12 个空槽的图像轮廓质心与投影圆心最大偏差约 0.98 px；工件圆心通过标注预览检查，未单独测量其像素偏差。`preview.png` 和每帧 `_overlay.png` 用绿色标空、橙色标满；`validation.json` 和 `render.log` 保存本次检查证据。

上述 5 项单元测试验证投影、畸变与缩放、LabelMe 配对、50 px 半径、训练读取与坐标还原、真实验证集保留。本次未开始混合训练；小样例渲染通过不代表复杂场景的容量或现场效果。真实效果需要对比相同真实验证集和独立现场数据上的圆心误差与空满准确率。

参考 NVIDIA 官方的 [Replicator 入门脚本](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_getting_started.html)、[相机标定与畸变模型](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html)及 [Isaac Sim 硬件要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)。

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
