# 预训练 MobileNet 学生容量检查

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-12
- Verification Status: 单种子执行完成；无独立复跑或最终test
- Version Label: vision_teacher_student_capacity_v1

使用原有crop和冻结DINOv2教师特征/分类头；ImageNet MobileNetV3-small替换五类输出。
学生1,522,981参数，三支预先固定8轮；同初值、同采样、同优化步数。

## 固定概率阈值0.5

| 方法 | Recall | 负帧FPR | F1 | TP | FP |
|---|---:|---:|---:|---:|---:|
| yolo | 0.9557 | 0.0392 | 0.9570 | 367 | 16 |
| frozen_teacher | 0.6146 | 0.0000 | 0.7613 | 236 | 0 |
| explicit_negative_ce | 0.8750 | 0.0147 | 0.9256 | 336 | 6 |
| teacher_kd | 0.7708 | 0.0123 | 0.8642 | 296 | 5 |
| confidence_gated_kd | 0.8125 | 0.0074 | 0.8927 | 312 | 3 |

## 仅validation诊断：保持Recall至少95%

阈值选择与评价均用validation，是探索性操作点；不是最终test或泛化证据。

| 方法 | P(drone)阈值 | Recall | 负帧FPR | 直升机FP/72 | TP | FP |
|---|---:|---:|---:|---:|---:|---:|
| yolo | — | 0.9557 | 0.0392 | 8 | 367 | 16 |
| frozen_teacher | 0.047619 | 0.9505 | 0.0123 | 0 | 365 | 5 |
| explicit_negative_ce | 0.008696 | 0.9505 | 0.0245 | 7 | 365 | 10 |
| teacher_kd | 0.050315 | 0.9505 | 0.0245 | 8 | 365 | 10 |
| confidence_gated_kd | 0.046381 | 0.9505 | 0.0172 | 5 | 365 | 7 |

## 如何理解

CE与KD是同一强学生上的控制对照；较旧19k从零学生同时改变了模型容量和预训练，因此不能分开归因。
固定0.5点易受概率校准影响，应结合保持95%召回的操作点比较。阈值诊断不改变已训练权重。
学生部署是YOLO候选crop复核器，不能补回YOLO未检出的无人机；仍不是完整YOLO蒸馏。
这里只有一个seed、8轮训练；不声称收敛、最终创新成功或跨场景稳定提升。

## 文件与复现

protocol.json记录来源SHA；summary.json含固定点/分类别误报/阈值诊断；validation_threshold_curve_*.json为完整操作点曲线。
student_*.pt保留模型/优化器；training_*.csv与console.jsonl记录过程；validation_logits_*.pt及预测JSON可复算。

脚本拒绝覆盖已有run：复跑前必须给新输出目录。命令从项目根执行：

`experiments\.venv\Scripts\python.exe experiments\scripts\vision_teacher_student_capacity.py`

本次总耗时121.97秒；三支每支688步。

## 真实validation案例与复核

`validation_cases.html`内嵌全部图片，可直接复制单文件分享；`validation_cases.jpg`是8例总览。
`validation_case_manifest.json`记录每例的原图、视频/帧、完整精度概率、阈值决策与显示框。
`case_assets/`保留逐例卡片和带框原图。案例包括修正误报、残存误报、额外漏检、挽回普通KD漏检、正常保留与YOLO原始漏检。
这些案例在看过validation结果后按预定义类别选取，供理解机制与失败模式，不是随机抽样的性能估计或独立泛化测试。

保存权重已在CPU严格重载；参数有限、最终权重哈希与summary一致。
来源crop、教师特征、教师头与评价引用文件SHA未改变；同初始化、同采样与同688步核查通过。
从保存logits独立复算95%召回工作点，三支均365 TP，CE/KD/Gated KD分别10/10/7 FP，与summary一致。
8例JPG已目视检查，HTML无外链图片依赖。案例生成脚本为`experiments/scripts/vision_teacher_student_capacity_cases.py`。
