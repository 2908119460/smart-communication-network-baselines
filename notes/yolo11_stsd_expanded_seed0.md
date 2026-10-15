# YOLO11-STSD 扩容实验：17,184 帧，单种子

本轮将训练帧数从 4,296 增加到 17,184（4 倍），继续使用原来 179 个训练视频。增加的是视频内采样密度，独立视频与场景数量没有增加。数据仍为 DroneDetectionThesis / Svanström RGB，不是官方 Anti-UAV。

## 数据与采样

| 项目 | 原正式实验 | 扩容实验 |
| --- | ---: | ---: |
| 训练视频 | 179 | 179 |
| 训练帧 | 4,296 | 17,184 |
| 无人机正帧 | 1,488 | 5,952 |
| 不重叠训练帧对 | 2,148 | 8,592 |
| 每视频训练帧 | 24 | 96 |
| 同一帧对的原始帧间隔 | 13–15 | 1 |
| 每轮 batch 数（batch=8） | 537 | 2,148 |
| 验证帧 | 792 | 792，原样引用 |
| 校准帧 | 480 | 480，原样引用 |
| 测试帧 | 792 | 792，原样引用 |

每段训练视频均匀取 48 个锚点，每个锚点提取 `(a, a+1)` 两张相邻原始帧，共 96 张。30 fps 视频中帧对间隔约为 1/30 秒。一个 epoch 中每张训练图只使用一次。相邻帧之间相关性较高，4 倍帧数不能解释为 4 倍独立样本。

新增图像与标签合计 394,872,426 字节（约 377 MiB）。准备脚本检查了 held-out 图像、标签及 records 保持不变，跨 split 视频交叉与精确像素重复均为 0。数据准备完成后重复执行会直接读取已有协议，不改写数据。

数据目录：`experiments/data/vision/stsd_expanded_v1/`。

- `protocol.json`：来源、拆分、训练配置与哈希。
- `sampling_audit.json`：每视频锚点、帧对、帧间隔与磁盘占用。
- `split_manifest.json`：每帧路径、类别与原始帧号。
- `train.yaml`、`lists/`：Ultralytics 数据入口；held-out lists 引用原图。

## 训练与评估

扩容实验使用独立输出目录 `experiments/runs/vision_yolo11_stsd_expanded_v1/`。队列等待原 4,296 帧的 STSD、匹配对照和评估全部结束后，再依次运行扩容 STSD、扩容 YOLO11n 匹配对照、固定评估，单卡串行。

两种结构均从同一官方 COCO YOLO11n 权重初始化；seed=0，640 输入，batch=8，AMP，AdamW，lr0=0.001，nbs=8，80 epoch 上限，patience=15，使用同一确定性 letterbox 配对训练器。保留原 YOLO Detect head。此轮只使用单种子，不开展多种子扫描。

验证 mAP50–95 用于选择 checkpoint，验证时关闭时序 cache；训练结束后，在原固定 val/test 和 4 段完整测试视频上评估，包括 STSD 时序开/关。时序收益只能根据视频评估判断，不能根据单图验证证明。

扩容同时改变帧数与帧对间隔，因此不能把新旧 STSD 的差异全部归因于“更多数据”。结构效果优先比较扩容 STSD 与同数据、同训练器的扩容 YOLO11n 对照。

网页上的状态是带时间戳的发布快照。以下结果于 2026-10-15 更新，排队和训练阶段均已结束。

## 2026-10-15 已完成结果

北京时间 00:24 开始扩容训练，03:47 完成整条队列，训练、对照、评估退出码均为 0。
STSD 在第 19 轮早停，最佳验证 checkpoint 在第 4 轮；对照在第 17 轮早停，最佳在第 2 轮。
均保留原 80 轮上限和 patience=15，没有手动截短。

| 模型 | 验证 mAP50 | 验证 mAP50–95 | 测试 mAP50 | 测试 mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| 匹配 YOLO11n | 61.88% | 25.84% | 62.46% | 29.81% |
| STSD | 59.81% | 26.22% | 46.13% | 22.96% |

扩容后两个模型均低于其 4,296 帧版本，STSD 也低于扩容匹配对照。目前结果不支持扩容有效或三模块整体提升。
训练损失下降而验证指标恶化，提示训练与泛化问题。帧密度、帧对间隔和每轮更新数同时改变，
每轮 batch 数变为 4 倍而学习率、warmup 与调度仍按 epoch 设置，具体原因尚待单种子针对性诊断。

4 段视频仍为 1,224 帧，阈值 0.25。匹配对照召回 248/305 正帧，STSD 时序开/关均为 290/305，
所有这些方法在 919 张负帧上均为 0 误报。STSD 在该局部视频上的召回较高，但开启时序没有新增召回，
不能据此证明时序模块收益；presence 也不检查框 IoU，不能替代全测试集 detection mAP。
缓存实际在 1,220 帧使用。扩容训练帧间隔为 1，与连续视频推理一致；原始评估 JSON 复用的“间隔不同”限制仅适用于旧 4,296 帧实验。

原始结果：`formal_seed0/evaluation/summary.json`；训练摘要：两模型目录各自的 `training_summary.json`。

## 运行方式

在工作区中使用现有 `experiments/.venv/Scripts/python.exe`：

```powershell
python experiments/scripts/vision_stsd_expand_data.py
python experiments/scripts/vision_stsd_formal.py all --epochs 80 `
  --data-root "experiments/data/vision/stsd_expanded_v1" `
  --run-root "experiments/runs/vision_yolo11_stsd_expanded_v1" `
  --weights "experiments/data/vision/paper_v1/weights/yolo11n.pt" `
  --wait-pipeline "experiments/runs/vision_yolo11_stsd_v2/formal_pipeline_seed0/manifest.json"
```

已有队列时不要重复启动。当前状态见输出目录 `formal_pipeline_seed0/manifest.json` 和 `formal_status.json`；日志位于 `formal_launcher/` 与 `formal_pipeline_seed0/`。
