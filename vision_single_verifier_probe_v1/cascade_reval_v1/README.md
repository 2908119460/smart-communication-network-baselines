# Frozen YOLO verifier cascade re-evaluation

固定已有 YOLO11n 与 verifier 权重，不重训；只读取 paper_v1 train/validation 候选，不读取 calibration/test。
接受规则为 detector score >= 0.25 且 verifier probability >= 0.50；低于 detector 阈值的候选不能被复核器提升。

context 模型是单个扩大约 3 倍的 crop 输入，不是 target/context 双分支网络。

| 方法 | recall | negative-frame FPR | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO | 0.9557 | 0.0392 | 0.9570 | 367 | 16 | 17 | 392 |
| YOLO + EMA0.6 | 0.9375 | 0.0245 | 0.9549 | 360 | 10 | 24 | 398 |
| YOLO + target crop cascade | 0.9531 | 0.0392 | 0.9556 | 366 | 16 | 18 | 392 |
| YOLO + expanded 3x crop cascade | 0.7552 | 0.0221 | 0.8492 | 290 | 9 | 94 | 399 |

类别诊断：飞机、鸟、直升机为负帧 FPR，无人机为 recall。

| 类别 | 帧数 | YOLO | EMA0.6 | target crop cascade | expanded 3x crop cascade |
|---|---:|---:|---:|---:|---:|
| AIRPLANE | 168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| BIRD | 168 | 0.0476 | 0.0179 | 0.0476 | 0.0060 |
| DRONE | 384 | 0.9557 | 0.9375 | 0.9531 | 0.7552 |
| HELICOPTER | 72 | 0.1111 | 0.0972 | 0.1111 | 0.1111 |

结论：target crop 没有减少任何误报，损失 1 个无人机 TP。expanded crop 过滤 7 个鸟类误报，但没有过滤 8 个直升机误报，且损失 77 个无人机 TP（recall 下降约 20.05 个百分点）。当前两种单帧复核器均未达到既定实用门槛。

逐候选预测：validation_predictions.json。完整摘要：summary.json。
重算检查：metric_recompute_audit.json 验证每个 accepted 规则、frame score、级联报警集合包含于 baseline 以及重新计算的指标。
两种复核器实际参数数分别为 19105 和 19105，通过 model.parameters() 统计。
可视化：validation_verifier_montage.jpg；绿色为通过级联的候选，红色为过滤的 baseline 候选。

本次没有重训，没有读取 calibration/test；这只是固定权重的 validation 可行性诊断。EMA作用于每视频24个稀疏抽帧，不能当连续视频结论。
