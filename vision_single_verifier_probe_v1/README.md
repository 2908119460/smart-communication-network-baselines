# Frozen YOLO 二阶段复核器短验证

> 旧评价口径已撤回：下表误把 verifier probability 当 detector 操作分数，且允许 detector score>=0.05 的候选被提升，所以不能解释为 YOLO conf=0.25 的过滤级联。纠正后的固定权重级联结果、逐候选预测和重算审计见 [cascade_reval_v1/README.md](cascade_reval_v1/README.md)。旧表仅作为错误记录保留。context 实际为一个扩大约 3 倍的 crop 的单分支输入，不是 target/context 双分支架构。

仅读取 paper_v1 的 train/validation（不读取 calibration/test），冻结 YOLO11n seed0 best.pt；结果用于决定是否值得继续，不是最终 paper test。

验证帧数：792；train候选样本：4411（正3814、负597）；复核器训练4轮，固定阈值0.50。

| 方法 | recall | negative-frame FPR | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO | 0.956 | 0.039 | 0.957 | 367 | 16 | 17 | 392 |
| YOLO + EMA0.6（稀疏验证诊断） | 0.948 | 0.032 | 0.957 | 364 | 13 | 20 | 395 |
| YOLO + target crop | 0.987 | 0.140 | 0.924 | 379 | 57 | 5 | 351 |
| YOLO + target + 3x context | 0.776 | 0.037 | 0.855 | 298 | 15 | 86 | 393 |

可视化：validation_verifier_montage.jpg。绿色框表示复核器保留的候选，红色框表示目标+上下文复核器过滤掉的高分候选；这是验证集诊断图，不从test挑选。

解释：EMA只处理时间抖动；二阶段复核器尝试利用候选目标与周围背景的视觉证据，可能降低直升机/复杂背景误报，但不能恢复YOLO漏检。

限制：验证集已用于本次可行性判断，不能把该结果当最终泛化；若继续，需预先冻结新方案和阈值，再使用未见测试视频重新评估。
