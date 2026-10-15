## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-14
- Verification Status: RUNNING
- Version Label: yolo11_stsd_v2_formal_seed0

## 正式实验

2026-10-14 22:38（北京时间）开始。使用 v2 直接替代固定 FPN/PAN 的
YOLO11-STSD，原 Detect 保持不变。实际后台进程执行训练，不是 demo 预算。

```powershell
experiments\.venv\Scripts\python.exe experiments\scripts\vision_stsd_formal.py all --epochs 80
```

顺序为完整 STSD → 相同训练数据处理的 YOLO11n control → checkpoint 固定后的
validation/test 和视频评估。每组只用 seed 0；上限 80 轮，patience 15。
不会覆盖旧 probe、原 baseline 或失败记录，不自动重复失败的训练。

## 数据与预算

使用既有 DroneDetectionThesis RGB 视频划分，非官方 Anti-UAV。
train=4,296 帧/179 视频，validation=792/33，test=792/33；另有未参与训练的
calibration=480/20。现有 split 的视频、原视频 SHA 与图片/pixel SHA 无跨集合重叠。

两组都使用已核验的官方 COCO `yolo11n.pt` 初始化，不使用完成无人机训练的
baseline best 或 8 轮 probe 权重。640、batch=8、AdamW、lr0=.001、lrf=.01、
warmup=3、warmup_bias_lr=.01、nbs=8、AMP 与原 baseline 的主要优化配置一致。

`temporal_pair_stride=2` 形成 2,148 对不重叠采样帧，所有 4,296 张训练帧每轮恰好
使用一次，共 537 batches/epoch。batch=8 是 4 对/8 张图；不改模型输入接口。
原帧间隔 13–15，约 0.43–0.50 秒，不是相邻原始视频帧。框和分类损失保留。

配对 Trainer 使用确定的正方形 letterbox，不做 mosaic、HSV 或随机几何增强。
因此额外跑相同配对管线的原 YOLO11n control，以区分结构变化和增强策略变化。
历史 baseline 仅作额外参考。各组自然早停、AMP 溢出跳过更新可能导致总成功更新数
不同，使用 optimizer hook 记录实际次数，不将相同轮数误称为完全相同计算量。

## 真实进度和产物

- `experiments/runs/vision_yolo11_stsd_v2/formal_pipeline_seed0/manifest.json`：
  串行队列、进程、开始/结束时间、退出码；每 30 秒监测，不按固定墙钟时间截断 80 轮预算。
- `formal_seed0/`：STSD 的 console.log、results.csv、heartbeat.json、progress.json、
  initialization_transfer.json、initialization_handoff.json、runtime_setup.json、weights/。
- `formal_pair_baseline_seed0/`：对应的匹配 YOLO11n 对照。
- `formal_seed0/evaluation/`：冻结 best 后的评估计划、逐帧记录和 summary.json。
- `formal_status.json`：供网页读取的带时间戳状态。线上页面为上次提交时的快照，
  不连接本机 GPU；本地状态观察进程每 30 秒更新此 JSON。

正式运行首轮前已实际核对 trainer 的 475 个 tensor 与映射初始化一致；AMP=true、
batch=8、2,148 对、4,296 唯一帧、537 batches 符合配置。配对 stride 的新增测试通过，
模块/集成检查现为 11 项。模型全部模块由真实检测训练参与优化。

## 评估

checkpoint 仅按无状态 validation mAP50–95 选择。训练结束后对三组模型
（STSD、匹配配对 control、历史 baseline）使用同一 validation/test 各 792 帧评估
mAP50、mAP50–95、Precision 和 Recall。

视频使用此前固定的 test 序列 V_DRONE_005、V_BIRD_004、V_HELICOPTER_008、
V_AIRPLANE_007，共 1,224 帧/方法；完整读取每个视频，比较 STSD temporal off/on、
匹配 control 与历史 baseline。置信度固定 .25，不在 test 搜阈值。记录帧级召回、
负帧误报率、分来源类别误报、预测延迟及缓存使用次数。presence 指标不要求框 IoU，
不能替代 detection mAP。开启/关闭时序是相同模型的推理消融，尚非去掉时序训练的结构消融。

## 2026-10-15 已完成结果

北京时间 00:23 完成训练、匹配对照和固定评估，全部阶段退出码为 0。
STSD 在第 28 轮早停，按验证集选择第 13 轮；对照在第 30 轮早停，选择第 15 轮。

| 模型 | 验证 mAP50 | 验证 mAP50–95 | 测试 mAP50 | 测试 mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| 匹配 YOLO11n | 73.15% | 30.78% | 82.24% | 36.64% |
| STSD | 74.09% | 32.03% | 78.23% | 37.45% |
| 历史 YOLO11n（配方不同，仅参考） | 81.07% | 38.72% | 83.10% | 39.14% |

STSD 测试 mAP50–95 比匹配对照高 0.82 个百分点，但 mAP50 低 4.01 个百分点，不能概括为整体提升。
4 段固定视频共 1,224 帧，其中 305 张正帧、919 张负帧。固定阈值 0.25 时，
对照召回 294/305；STSD 时序关闭召回 291/305，开启召回 292/305，均无负帧误报。
缓存确实在 1,220 帧中使用，但时序只多召回 1 帧，尚不足以支撑明确收益或减少鸟类误报的结论。

原始结果：`formal_seed0/evaluation/summary.json`；训练摘要：两模型目录各自的 `training_summary.json`。
尚未完成分别移除 Sky、Dynamic Neck、Temporal 的训练消融，不能将整体差异归因于其中某个模块。
