# 冻结基础视觉教师与显式负类学生初试

只用train/validation；基础模型不微调。部署候选是YOLO+轻量复核器，并非已完成YOLO蒸馏。

| 方法 | Recall | 负帧FPR | F1 | TP | FP |
|---|---:|---:|---:|---:|---:|
| yolo | 0.9557 | 0.0392 | 0.9570 | 367 | 16 |
| frozen_teacher_cascade | 0.6146 | 0.0000 | 0.7613 | 236 | 0 |
| explicit_negative_ce | 0.2682 | 0.0196 | 0.4162 | 103 | 8 |
| teacher_kd | 0.5391 | 0.0147 | 0.6935 | 207 | 6 |

训练crop 5463；两学生均12轮；DINOv2 backbone 22,056,576 参数；学生 19,301 参数。

CE和KD使用相同数据/初值/采样/更新预算。若两者都优于旧复核器，不能归因给蒸馏；必须看KD相对本轮CE的差异。

完整参数、分类别误报及逐候选概率见summary.json和validation_predictions_*.json。没有新增训练种子或最终测试。