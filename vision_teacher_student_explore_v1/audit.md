# 冻结 DINOv2 教师 / CE / KD 学生短测：只读审查

2026-10-12。审查范围是实现、已保存数据/权重元信息、预测指标复算。没有重训，没有增加 seed，没有修改实验脚本或结果；审查不是独立完整训练复现。

## 结论

实现与保存结果一致，未发现 train/validation 混用、按视频名偷取类别、绕过 YOLO gate、固定阈值指标复算错误等问题。**本次可以支持“冻结基础视觉特征具有可利用的近类排序能力；蒸馏对极小学生有部分正向信号”**，还不能支持“已经成功蒸馏 YOLO”“大模型部署创新已经有效”或“低误报且召回不受损”。

固定 `P(drone)≥0.50` 工作点对教师和两学生都过于严格；降低误报伴随明显召回损失。展示时必须同时列 recall/FPR。验证集选择阈值的曲线有助于看潜力，但仅为同一验证集上的探索性工作点，不能当作独立测试成绩。

## 数据和标签

- 实际训练 crop 为 **5,463**：2,376 个原 MAT annotated box + 3,087 个 detector candidate。类别计数为 drone 3,591、helicopter 683、bird 552、airplane 473、background 164。
- 逐条对照原 split manifest，所有训练 crop 均来自 train，所有 438 个 evaluation candidate 均来自 validation；训练与验证视频交集为 **0**。验证共有 **792 帧，384 正帧、408 负帧**。
- crop cache 形状为 train `(5463,96,96,3)`、validation `(438,96,96,3)`；缓存 labels 与 training_samples.json 逐项一致。
- GT labels 来自原 MAT 的 DRONE/HELICOPTER/BIRD/AIRPLANE 框。candidate 与所有类别框按最高 IoU 匹配；IoU<0.30 才标 BACKGROUND。没有用 sequence 名称给候选分类。
- 从五类各均匀抽查 10 个样本，覆盖 48 段源视频，直接重新解码原 MAT，**50/50** 与所存 labels/boxes 一致。这里是抽查，不声称重新逐项审查全体 MAT。
- 一个解释边界：候选 IoU<0.30 的 BACKGROUND 是几何匹配标签；其中可能包含部分截断/错位的小目标，不能把它一概解释成“纯空背景”。GT 框补充和真实候选分布存在差异，应作为下一轮数据改进考虑。

## 教师和学生范围

- 教师是官方 DINOv2 **ViT-S/14，22,056,576 个 backbone 参数**，冻结 backbone + **1,925 参数的五类线性 head**。head 用这些有标签训练 crop 训练 200 次 full-batch update；没有基础模型 fine-tune。
- 这属于基础视觉特征的“小规模教师前测”；不是十亿级视觉/多模态大教师，也不是开放词汇零样本检测。
- 学生是 **19,301 参数**三层 CNN 五类 crop verifier，无预训练初始化；不是修改或蒸馏 YOLO11n 本体。部署候选系统是“冻结 YOLO11n + 小 verifier”。
- crop 外扩 1.25×，先缓存为 96×96；学生用 96，DINO 教师将缓存图再插值到 224。插值不会创建原图不存在的目标细节。
- 教师读取的是 **预测 candidate crop** 来做验证，没有在验证阶段借 GT crop 充当部署输入。

## CE 与 KD 对照

- 两学生共享同一 crop cache、GT labels、class-balanced replacement sampling、batch=64、12 epoch、AdamW 配方、模型初始化。
- 从 seed 和学生类重新构造 CPU 初始模型，参数 SHA256 为 `b5069dacee67405e16cc0bc24620f88d2888a62914f6c70a7f123f9bb37be37f`，与两份 checkpoint 元信息均相同。
- 两学生都记录 **1,032 optimizer steps**，与 `12 × ceil(5463/64)` 一致，epoch history 均 12 条。采样 generator 每支独立以同一 seed 初始化，代码路径表明两支 draw 相同；运行时的 draw 序列未额外保存，所以这里不称逐 step 日志独立重放。
- CE 分支仅 CE；KD 分支 `0.5 CE + 0.5 T² KL`、T=2，教师 soft target 来自已冻结教师 head。两支的计算预算相同，但损失组成不同；这是一项标准蒸馏配方对照，不是 FGD/CrossKD 精确复现。
- 旧 4 轮二分类 verifier 的数据、类别、crop、训练轮数都不同，不能将此次相对旧 verifier 的变化全部归因给教师蒸馏。**能归因的主要比较是本轮 CE 与本轮 KD。**

## 级联与指标复算

实际级联是 `YOLO score≥0.25 AND P(drone)≥0.50`。只要一帧有一个通过的 candidate，就算 alarm；所有其他 candidate 的高 verifier 概率不能绕过 detector gate。对全部保存 candidate 独立复算 acceptance，三个模型均 **0 个不一致**。

| 方法 | TP | FP | FN | TN | Recall | 负帧 FPR |
|---|---:|---:|---:|---:|---:|---:|
| YOLO 0.25 | 367 | 16 | 17 | 392 | 95.57% | 3.92% |
| 冻结教师，p≥0.5 | 236 | 0 | 148 | 408 | 61.46% | 0% |
| CE 学生，p≥0.5 | 103 | 8 | 281 | 400 | 26.82% | 1.96% |
| KD 学生，p≥0.5 | 207 | 6 | 177 | 402 | 53.91% | 1.47% |

这三个级联的 TP/FP/FN/TN 均与 summary.json 一致。KD 相对本轮 CE 在固定0.5工作点有明显正向信号，但其 F1/recall 都明显弱于原 YOLO；不能只拿 FP 从16到6宣称系统全面改善。教师0误报同样伴随61.46% recall，不能忽略。

这些是 **frame-level presence**，不是候选五类准确率、检测 mAP 或 IoU 定位成绩。正帧上任何通过候选都计 presence TP，即使框位置未对上目标；这是现有指标定义的边界。

## 阈值诊断独立复算

重新从每帧符合 YOLO≥0.25 gate 的候选取最大 drone probability，再对全部唯一概率阈值复算，结果与 operating_curves.json/README 一致。

| 方法 | 同验证集选 recall≥95% 的 FPR | 同验证集选 FPR≤1% 的 recall |
|---|---:|---:|
| 教师 | 1.2255%（recall95.0521%，τ0.047619） | 94.7917%（FPR0.9804%，τ0.077440） |
| CE 学生 | 3.4314%（recall95.0521%，τ0.016904） | 9.8958%（FPR0.9804%，τ0.688376） |
| KD 学生 | 3.6765%（recall95.0521%，τ0.010016） | 32.2917%（FPR0.9804%，τ0.593017） |

这说明教师特征的排序能力值得继续；学生确实尚未把教师的能力充分学到。KD 在严格低误报工作点优于 CE，但在保持95% recall时 **KD FPR略高于CE**。因此不能声称蒸馏在所有工作点都胜出。

## 下一步最有价值的实验

当前不需要多 seed 或大规模防御性重复。优先做一项能解释“为什么教师可分、学生没学到”的对照：相同数据下，预训练较强小学生或 teacher feature projection，对比当前从头19k CNN；另以 GT 框和预测框的同类混淆分别诊断定位错位。模型较大并不自动成功，但现有教师曲线给了可继续的依据。

若做正式 demo，先在独立 calibration 冻结工作点，再输出固定整段连续视频，同时展示无人机保留、无人机被误删、误报过滤、误报保留。当前展示四组诊断图已经避免只挑成功帧，但还不是完整时序 demo 或最终 test。
