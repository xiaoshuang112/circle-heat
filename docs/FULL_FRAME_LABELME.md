# 整图格位自动标注：交给 AI 的操作文档

用户以后可以直接说：

> 读取 `/home/irobotic/work/circle-heat/docs/FULL_FRAME_LABELME.md`，用当前算法标注 90 上的 `<图片目录>`，然后拷贝到 97。

执行前读取目标目录适用的 `AGENTS.md`。按用户指定的图片目录操作；本任务是离线预标注，不需要改训练网络、修改生产配置、调用相机或操作机器人。

## 1. 输出要求

- 保留完整原图，不导出单个格位的裁图。网络内部取 ROI 属于推理步骤，不能把这些 ROI 保存为交付图片。
- 每个实际检测到的格位生成两个 shape，使用同一个 `group_id`：

| 内容 | LabelMe `shape_type` | `label` | `points` |
|---|---|---|---|
| 格位外框 | `rectangle` | `bbox` | 左上角、右下角 |
| 格位中心 | `circle` | `occupied` 或 `empty` | 第一个点为圆心；第二个点距圆心 **50 px** |

- 使用原图像素坐标，圆半径统一为 **50 px**。不要使用检测框半径、拟合半径或热图坐标代替。
- `imagePath` 使用同目录图片文件名，`imageData` 为 `null`，宽高与完整图片一致。
- 顶层 flags 设置 `auto_generated=true`、`needs_review=true`。自动标签不是人工真值。
- 不覆盖原图、已有人工标注或以前的输出目录。输出目录存在时创建新的带时间后缀的目录。
- 整盘图、近拍图都可以标注。每张图的格位数量取决于检测结果，不要求每张恰好 9 个或 16 个，也不补造缺失的格位。

## 2. 设备与算法位置

2026-10-10 实际使用的位置如下。以后先检查这些路径是否仍然有效，读取当前配置和对应源代码，不要只根据文件修改时间猜版本。

| 项目 | 位置 |
|---|---|
| 90 | `nvidia@192.168.1.90` |
| 97 | `irobotic@192.168.1.97`，本次执行主机为 `ira-01` |
| 算法源码（90） | `/home/nvidia/work/det-xialiao-nano` |
| 当前算法库（90） | `/home/nvidia/work/robotic_arm_system/dist/qt_robot_app/libai-det.so` |
| 构建工具目录（90） | `/home/nvidia/work/det-xialiao-nano/build-release-mvs` |
| 可用离线运行配置（90） | 上一行目录下的 `tray-ai-det-replay-runtime/` |
| 格位检测模型 | 运行配置的 `models/xialiao.m`；实际名称应读取 `cfg.ini` 的 `model_name` |
| 圆心、空满模型 | 运行配置的 `models/circle-center.m`；实际名称应读取 `cfg.ini` 的 `circle_center_model` |

只用 SSH key 或已有 SSH 认证，不把密码、私钥或 Token 写到文档、脚本或日志。

**模型选择：**默认使用 90 当前运行配置中的模型，并记录实际文件的 SHA256。“当前运行模型”不一定是“最新训练权重”。2026-10-10 这次标注使用的 90 圆心引擎文件来自 9 月 22 日，并没有使用刚导出的最新 ONNX。如果用户明确要求最新模型，应先确认指定的权重/引擎，不能悄悄混用，也不能把旧引擎说成最新模型。离线标注不授权更换生产模型。

## 3. 准备输入与隔离目录

1. 确认主机身份、输入路径、图片数、格式、宽高和文件哈希。
2. 对本次这类目录，原图在 `production/`、`teaching/`、`app_images/`。以后以用户给定的目录结构为准。
3. 同名 PNG 和 JPG 并存时，优先选原始 PNG；已有带框结果图、`preview/`、历史输出目录不进入推理。只有 JPG 原图的批次仍应处理 JPG，不能一律丢弃。无法判定是不是原图时，先检查图像。
4. 在输入目录下创建新的 `labelme_fullframe_<时间>/`，保留原始子目录，避免同名图片冲突：

```text
labelme_fullframe_<时间>/
  labelme/
    production/    完整图片 + 同名 LabelMe JSON
    teaching/
    app_images/
  preview/         完整图片的叠加预览，不是格位裁图
  evidence/        bbox 原始结果、圆心原始结果、工具源码和推理日志
    runtime/      隔离使用的 cfg.ini、param、算法库与模型
  summary.json     数量、类别、版本、哈希、回退情况和失败情况
```

5. 将完整原图复制到 `labelme/`，字节内容不变。把选定的库、配置和两个模型复制到隔离 `runtime/`，所需 `param/` 也复制。模型可以只读使用，但不能把运行日志写到生产目录。
6. 记录库对应的 Git SHA、库和模型哈希、模型来源路径。不能把不同库/模型组合的结果当成同一次推理结果。

## 4. 第一段：用现有检测器取格位 bbox

使用现有工具 `rgb-model-inspector`。它调用当前 YOLO 实现的预处理、推理、后处理和原图坐标恢复，不需要连接相机。

```bash
/home/nvidia/work/det-xialiao-nano/build-release-mvs/rgb-model-inspector \
  <隔离runtime>/models/xialiao.m \
  <输出目录>/labelme \
  <输出目录>/evidence/bbox_results.jsonl
```

工具递归读取图片，所以 `labelme/` 中只放选定的完整原图。检查当前配置中的翻转和缩放规则是否与工具调用一致；本次没有新增翻转、缩放或阈值。

结果每行对应一张图，其中：

- `plate_cell_boxes`：检测器认为空的格位外框。
- `lunzi_boxes`：检测器认为有工件的外框。
- 每个框是 `[x, y, width, height]`，位于完整原图坐标系。
- 其他类别如 `plate_tray`、`plate_hole`、`hole` 不导出为 `bbox`。

合并 `plate_cell_boxes` 和 `lunzi_boxes` 作为格位候选，并保留其检测器类别，用于网络不可用时执行原算法的回退。出现重叠候选、漏检、无候选或读图失败时记录问题；不要为了凑固定数量而补框，也不要临时发明新的去重规则。

## 5. 第二段：调用当前库的原生圆心预测

对每个格位框，在内存中调用当前算法库导出的接口：

```cpp
tray_stack_pixels::setCircleCenterEnginePath(enginePath);
auto predicted = tray_stack_pixels::predictSlotCenter(fullFrameBgr, slotBox);
```

对应源码为 `det-src/CircleCenterNet.hpp`、`det-src/CircleCenterNet.cpp`。传入的是完整 BGR 图与原始格位 bbox。接口内部取 ROI、letterbox 和解码；返回的 `predicted.center` 已经是完整原图坐标，**不要再次加 bbox 偏移或乘缩放系数**。

可以编译文末的轻量调用工具，直接链接复制出的当前 `libai-det.so`。它只输出预测，不初始化完整相机/机器人运行时。模型推理使用现有 TensorRT GPU 引擎，不为标注任务重新部署或更换引擎。

### 圆心和类别选择规则

读取当前 `TrayStackPixels.cpp::buildCandidates`，保持与其规则一致。本次规则如下：

```python
box_center = [x + width / 2, y + height / 2]
max_shift = max(8.0, 0.25 * min(width, height))
accept_network = (
    predicted.valid
    and 圆心坐标有限
    and 圆心在完整原图范围内
    and distance(predicted.center, box_center) <= max_shift
)
if accept_network:
    center = predicted.center
    label = "occupied" if predicted.occupied else "empty"
    source = "circle_center_net"
else:
    center = box_center
    label = "occupied" if detector_class == "lunzi" else "empty"
    source = "ai_slot_bbox"
```

`predicted.occupied` 已由原算法判定，不另设空满阈值。保存占用概率、可定位概率、回退原因及每个候选的来源，便于复核。无效框或无法读取的图片要报告失败，不能用上一张图片的结果填充。

**不要把整盘生产接口作为预标注的前置条件。**它会检查 3×3/2×4 数量、网格几何和整盘有效性。近拍图只有 1～3 个格位时，整盘接口失败是正常的；标注仍可直接使用上面两段的候选输出。保留相关失败证据，也不能把候选标注说成生产整盘校验已通过。

## 6. 生成 LabelMe JSON

下面只演示一个格位的文件结构，坐标是示例，不是真实标签：

```json
{
  "version": "5.5.0",
  "flags": {"auto_generated": true, "needs_review": true},
  "shapes": [
    {
      "label": "bbox",
      "points": [[100.0, 100.0], [700.0, 700.0]],
      "group_id": 1,
      "description": "",
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    },
    {
      "label": "empty",
      "points": [[400.0, 400.0], [450.0, 400.0]],
      "group_id": 1,
      "description": "",
      "shape_type": "circle",
      "flags": {},
      "mask": null
    }
  ],
  "imagePath": "sample.png",
  "imageData": null,
  "imageHeight": 2048,
  "imageWidth": 2448
}
```

外框转换为 `[[x, y], [x + width, y + height]]`，导出时裁到图片范围内并检查面积为正。圆心选择仍按原始检测框与原算法规则计算，不因为导出矩形裁边而改变推理坐标。

圆的第二个点通常为 `[cx + 50, cy]`；靠近右边界时可用 `[cx - 50, cy]`，保证两点距离精确为 50。圆心必须在图内；不要为了把整个圆塞进 bbox 而缩小半径。标注坐标保留浮点精度，只在绘制预览时取整。

## 7. 检查、预览和交付到 97

生成后重新读取实际保存的 JSON，检查：

1. 每张成功标注的原图都有同名 JSON，`imagePath` 在其同目录可解析，宽高正确。
2. shape 只有 `bbox` 矩形和 `empty/occupied` 圆；每个格位的一框一圆使用相同 `group_id`。
3. 所有坐标有限；框面积为正；圆心位于原图内；每个圆两点的欧氏距离为 50 px。
4. 输入原图的前后 SHA256 不变，复制出的完整图片与原图哈希一致，没有导出格位裁图。
5. 汇总图片数、bbox 数、两类圆的数量、网络圆心数、bbox 回退数及失败图片。

绘制完整图片的叠加预览，至少查看整盘和近拍样本：蓝色框为 bbox，绿色圆为 occupied，黄色圆为 empty。预览只用于检查，不能作为原图再进入训练。自动预测有误时保留模型结果并提示人工复核，不擅自改成“人工确认”。

默认交付到 97 的新目录：

```text
/home/irobotic/Downloads/<输入目录名>_labelme/
  labelme/       完整图片与 LabelMe JSON
  preview/       完整图片叠加图
  summary.json
```

若执行主机已是 97，可以直接从 90 使用 `scp -r` 拷贝这三项。目标目录已存在时使用新的时间后缀。拷贝后比对源端与 97 端每个文件的 SHA256，并再次验证 97 端 JSON 的相对 `imagePath`；不能只检查 scp 的退出码。

最终回复用户：97 上可打开的绝对目录、图片及标注数量、50 px 半径检查结果、需要复核的类别/圆心问题。自动标注和人工复核是两个阶段；不要直接把这批自动标签加入训练。

## 8. 已完成批次与参考实现

2026-10-10 这次已执行、检查并交付：

| 项目 | 实际结果 |
|---|---|
| 90 输入目录 | `/home/nvidia/Downloads/remote_test_images_20261010_124128` |
| 90 结果目录 | 输入目录下的 `labelme_fullframe_20261010/` |
| 97 结果目录 | `/home/irobotic/Downloads/remote_test_images_20261010_124128_labelme/` |
| 完整图片 / LabelMe JSON | 10 / 10 |
| bbox / 圆 | 52 / 52，所有圆半径 50 px |
| 模型类别统计 | empty 51、occupied 1，尚未人工确认 |
| 圆心来源 | 网络 50、bbox 中心回退 2 |
| 拷贝检查 | 31 个文件全部 SHA256 一致：10 PNG、10 JSON、10 预览、1 汇总 |
| 算法 Git SHA | `a9ace55af720e49df4a1ee71976908fcadd39556` |
| 算法库 SHA256 | `b44029e06ec5440a7967a5a0255aa31b5944515dfc3504d8d7b117984e309812` |
| xialiao.m SHA256 | `08d85261d1f34d631ea066b02cf237c611bddf987e18644c6597384ffe4dd890` |
| circle-center.m SHA256 | `0f3563da531d2f5d3ccd118244ea0bb7debe735bc94ef6835906333b7485fc55` |

本次近拍工件存在明显空满误判，2 个回退圆心分别出现在 production 的 `v3-3`、`v3-6` 图片中。整盘接口的数量/几何校验失败记录已保留，最后交付的是格位候选预标注。

90 的结果目录 `evidence/` 中保存了本次参考源码：

- `predict_candidates.cpp`：链接当前库，直接执行原生圆心预测，已实际编译、运行。
- `finish_fullframe.py`：选择圆心/类别、输出 JSON、生成预览、检查 50 px 半径及原图哈希。
- `export_fullframe.py`：包含 JSON 生成辅助函数和 self-check；其历史 main 曾尝试整盘接口。

这些脚本有本批次的固定路径，不应直接用于新批次；可复用其辅助函数，并按本文构建新批次的最小入口。**不要重复执行历史 main 的整盘检测流程。**这些参考文件在 90 的数据目录中，不依赖 `/tmp`；若旧批次被归档，仍可使用下面的原生调用源码。

## 附：轻量原生圆心调用工具

以下是本次已实际使用的源码。输入为 inspector 产生的 JSONL，输出为每张图的原生预测 JSONL，不保存任何 ROI 图片。

```cpp
#include "CircleCenterNet.hpp"
#include "json.hpp"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    if (!input || !output) return 3;
    tray_stack_pixels::setCircleCenterEnginePath(argv[1]);
    std::string line;
    while (std::getline(input, line)) {
        const auto boxes = nlohmann::json::parse(line);
        const std::string path = boxes.at("image");
        const auto image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty()) return 4;
        nlohmann::json result = {{"image", path}, {"width", image.cols},
            {"height", image.rows}, {"candidates", nlohmann::json::array()}};
        for (const auto &category : {"plate_cell", "lunzi"}) {
            for (const auto &values : boxes.at(std::string(category) + "_boxes")) {
                const auto v = values.get<std::vector<float>>();
                if (v.size() != 4) return 5;
                const auto p = tray_stack_pixels::predictSlotCenter(image, cv::Rect2f(v[0], v[1], v[2], v[3]));
                result["candidates"].push_back({{"bbox_xywh", values}, {"detector_class", category},
                    {"valid", p.valid}, {"center", {p.center.x, p.center.y}},
                    {"occupied", p.occupied}, {"occupancy_probability", p.occupancyProbability},
                    {"localizable_probability", p.localizableProbability},
                    {"failure_reason", p.failureReason}});
            }
        }
        output << result.dump() << '\n';
        if (!output) return 6;
        std::cout << path << " candidates=" << result["candidates"].size() << '\n';
    }
    return 0;
}
```

将源码写到隔离运行目录的 `predict_candidates.cpp`，在 90 使用对应版本的头文件和库编译。以下命令的 `task_runtime`、`task_evidence` 替换为该批次实际目录：

```bash
g++ -std=c++17 -O2 "$task_evidence/predict_candidates.cpp" \
  -I/home/nvidia/work/det-xialiao-nano/det-src \
  -I/home/nvidia/work/det-xialiao-nano/include \
  $(pkg-config --cflags --libs opencv4) \
  -L"$task_runtime" -Wl,-rpath,"$task_runtime" -lai-det \
  -o "$task_runtime/predict-candidates"

"$task_runtime/predict-candidates" \
  "$task_runtime/models/circle-center.m" \
  "$task_evidence/bbox_results.jsonl" \
  "$task_evidence/candidate_predictions.jsonl"
```

编译或加载失败时记录具体错误，检查头文件、动态库、OpenCV、TensorRT 和模型版本是否对应；不要换用旧库或静默造出预测结果。后续格式转换应留一个小型 self-check，至少覆盖一框一圆、精确 50 px 半径、非法类别拒绝以及候选/JSON数量一致。
