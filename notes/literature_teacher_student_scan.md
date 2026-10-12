# 教师—学生路线文献扫描与可执行候选

日期：2026-10-12。当前聚焦 YOLO 无人机本体检测与 60 GHz RL。此文是探索性选题扫描，不是系统性综述、学术首创证明或已完成的实验结论。文献支持“机制值得尝试”，不自动支持“在本项目有效”。

## 结论先行

“离线大教师 + 在线小模型”仍然值得保留，但应把核心问题明确为：**教师掌握了学生没有的哪些有效信息，如何只迁移可信部分，并在小模型部署成本不变或可控时解决具体失败？**

YOLO 最值得先验证的是**基础视觉表征能否区分无人机和直升机等混淆类**，其次是**小目标定位分布/浅层特征蒸馏**。先查教师有没有优势，再蒸馏；教师不能区分的对象，增加蒸馏损失通常不会凭空产生可靠监督。仅用教师输出硬框训练学生应称伪标注/自训练。

60 GHz 可尝试**多场景专家教师→一个条件学生策略**或**教师指导后逐步解除约束**，但 Policy Distillation/Kickstarting 已有经典先例，不能把“加入 KL”本身说成新算法。新增贡献需落在不同负载/遮挡下的教师选择、动作结构和积压风险处理。

## 搜索与核验范围

- 搜索词：`knowledge distillation drone detection`、`knowledge distillation small object detection`、`knowledge distillation resource allocation wireless`；对 FGD、MGD、LD、Soft Teacher、Unbiased Teacher、Efficient Teacher、DINOv2/v3、Grounding DINO、Florence-2 与 Policy Distillation/Kickstarting 定向核验。
- 2026-10-12 实际读取 arXiv 官方摘要与元数据、作者官方 GitHub README、JCST 期刊摘要；对少量难以取得正文的论文只核对 Crossref DOI/题名并明确降低证据等级。
- 排除“只因名称含 drone 就当成本项目同任务”的做法。航拍地面目标检测与无人机本体检测不同；VisDrone/xView 上的收益只能作为机制先例。
- 本轮未完整阅读所有 PDF，未复现所引论文的完整数字，未扫描全部数据库；下文只将已读取材料支持的内容作为论文事实。

## 论文与资源矩阵

| 来源 | 论文事实/已核验机制 | 本项目可借鉴之处 | 复现与证据边界 |
|---|---|---|---|
| Wang 等，2019，*Distilling Object Detectors with Fine-grained Feature Imitation*，[原文](https://arxiv.org/abs/1906.03609)、[代码](https://github.com/twangnh/Distilling-Object-Detectors) | 普通检测蒸馏收益有限；重点模仿物体附近位置的特征响应，减少无关区域影响。摘要包含 KITTI/VOC/COCO 实验。 | 做局部/目标邻域蒸馏，而不是整幅图逐像素同权重 MSE。 | 较老检测代码；不是现成 YOLO11 插件。 |
| Yang 等，CVPR 2022，*Focal and Global Knowledge Distillation for Detectors (FGD)*，[原文](https://arxiv.org/abs/2111.11837)、[官方代码](https://github.com/yzd-v/FGD) | 分离前景/背景，强调关键像素与通道；另转移像素关系。COCO RetinaNet-R50 37.4→40.7 mAP 为论文/README 的报告值。 | 当前无人机小目标被大面积背景淹没，FGD 是合理的蒸馏基线。 | README 固定 mmdet 2.11/mmcv 1.2.4；应提取损失原理到现有环境，别把旧环境升级工作当创新。官方另有 YOLOX 分支，不等于 YOLO11 已可直接用。 |
| Yang 等，ECCV 2022，*Masked Generative Distillation (MGD)*，[原文](https://arxiv.org/abs/2205.01529)、[官方代码](https://github.com/yzd-v/MGD) | 随机屏蔽学生特征，用小重建模块恢复教师完整特征；适用于分类、检测、分割。论文摘要报告 COCO RetinaNet-R50 37.4→41.0。 | 易作为“简单特征蒸馏”对照；新增任务机制不能仅是再加一个 MGD。 | 有检测目录；迁移到 YOLO11 仍需 feature hook、通道映射及增强坐标对齐。 |
| Zheng 等，CVPR 2022，*Localization Distillation for Dense Object Detection (LD)*，[原文](https://arxiv.org/abs/2102.12252)、[代码](https://github.com/HikariTJU/LD) | 转移框位置分布而不仅是语义/中间特征，并在 valuable localization regions 选择性蒸馏；论文摘要 GFocal-R50 40.1→42.1，推理速度不变。 | 我们 mAP50 较好而 mAP50–95 弱，定位蒸馏比只提高 presence 更贴近实测瓶颈。 | 原实现基于 MMDetection/GFocal。YOLO11 的 DFL 分布可作为迁移起点，但 head、位置分配和坐标必须另行实现；不能称严格复现 LD。 |
| Ma 等，JCST 2024，*Knowledge Distillation via Hierarchical Matching for Small Object Detection*，[期刊摘要](https://jcst.ict.ac.cn/article/doi/10.1007/s11390-024-4158-5) | 在 P2–P4 层次匹配高层语义与浅层高分辨率小目标特征，另降低背景噪声。期刊摘要报告 COCO Faster R-CNN-R50 41.7 mAP，比其基线高 3.8。 | “给小目标加浅层蒸馏/多尺度匹配”已有直接先例；可研究我们数据的尺寸依赖权重。 | 本轮未确认官方可运行代码；不列为马上整仓复现首选。 |
| Liu 等，2021，*Unbiased Teacher for Semi-Supervised Object Detection*，[原文](https://arxiv.org/abs/2102.09480)、[代码](https://github.com/facebookresearch/unbiased-teacher) | 渐进教师与学生互相训练，处理伪标签偏差；类别平衡损失抑制过于自信的伪标签。 | 伪标签中容易漏掉无人机或把直升机标成无人机，类别偏差应显式检查。 | COCO/VOC 半监督不是无人机专用；EMA 教师本身也不保证消除稳定误报。 |
| Xu 等，ICCV 2021，*End-to-End Semi-Supervised Object Detection with Soft Teacher*，[原文](https://arxiv.org/abs/2106.09018)、[代码](https://github.com/microsoft/SoftTeacher) | 用教师分类分数加权未标注框分类损失；box jittering 选择可靠回归伪框。 | 将“类别可信”与“框位置可信”分开筛选，避免把高置信错类或不稳定小框一视同仁。 | 可靠性筛选已有先例；若做“直升机特异性拒绝 + 视频一致性”，必须分别消融，而不是把 Soft Teacher 换名。 |
| Xu 等，2023，*Efficient Teacher: Semi-Supervised Object Detection for YOLOv5*，[原文](https://arxiv.org/abs/2302.07577)、[官方代码](https://github.com/AlibabaResearch/efficientteacher) | Dense Detector、Pseudo Label Assigner、Epoch Adaptor 处理单阶段检测器伪标签质量与训练一致性。官方提供自定义数据配方和预训练权重。 | 比通用分类蒸馏更贴近 YOLO；半监督路线应至少比较普通同域 YOLO 自训练与质量筛选自训练。 | 论文针对 YOLOv5，当前 README 扩展支持其他 YOLO 系列；没有直接确认 YOLO11 支持。全套 COCO 配方较重，当前只做机制 probe。 |
| Oquab 等，2023/2024，*DINOv2: Learning Robust Visual Features without Supervision*，[原文](https://arxiv.org/abs/2304.07193)、[官方模型与代码](https://github.com/facebookresearch/dinov2) | 训练约 1B 参数教师并蒸馏到较小模型；官方支持冻结表征 + 线性分类。官方表：ViT-S 21M、B 86M、L 300M、g 1,100M。 | 不先做大模型 finetune，也能检查候选 crop 的基础表征是否可分无人机/直升机/鸟/飞机。 | 官方自然图像泛化不等于当前小目标可分性；冻结裁剪分类 probe 与整图检测必须分开报告。 |
| Siméoni 等，2025，*DINOv3*，[原文](https://arxiv.org/abs/2508.10104)、[官方代码/模型](https://github.com/facebookresearch/dinov3) | 通过 Gram anchoring 改善长期训练后密集特征退化，并提供不同规模部署模型。官方模型表包括 ViT-L 300M、H+ 840M、7B 6,716M。 | 后续可用较大的密集教师做 ROI/局部特征监督，比较 DINOv2 小教师与 DINOv3 大教师的任务收益。 | 官方权重入口需要申请获取；本轮未下载/运行。无需从零预训练 7B 模型。用新模型做教师本身不是首创。 |
| Liu 等，2023/2024，*Grounding DINO*，[原文](https://arxiv.org/abs/2303.05499)、[官方代码](https://github.com/IDEA-Research/GroundingDINO) | 用文字类别/描述进行开放集检测，在 COCO/LVIS/ODinW/RefCOCO 等评估。 | 可用 drone/helicopter/bird/airplane 同时提示，试探分类语义和伪框生成能力。 | 官方 zero-shot 指标没有证明能分清几十像素 UAV；prompt 不能用测试样本逐张调到正确。 |
| Xiao 等，2023，*Florence-2*，[原文](https://arxiv.org/abs/2311.06242)、[官方公开模型入口](https://huggingface.co/microsoft/Florence-2-large) | 使用统一 prompt 与序列输出支持 caption、检测、grounding、分割；论文说明 FLD-5B 大规模标注预训练。 | 冻结检测/grounding 可作为另一种教师能力筛查，与同域 YOLO 教师区分。 | 本轮 arXiv 摘要已核验，模型卡网络读取失败；未下载、未验证当前 UAV 能力。 |
| Guo 等，2023，*Global-Local MAV Detection under Challenging Conditions based on Appearance and Motion (GLAD)*，[原文](https://arxiv.org/abs/2312.11008)、[官方数据](https://github.com/WestlakeIntelligentRobotics/GLAD) | 全局搜索后切到自适应局部区域，并协调外观/运动检测；直接面对 UAV 本体小目标、复杂背景和计算限制。 | “全局—局部切换”与“加运动信息”已有强直接先例；可作为外部 UAV 本体数据和时序教师方向的来源。 | 本轮仅核验摘要，未在本机下载或复现；不能用其数字当本项目结果。 |
| Akyon 等，ICIP 2022，*Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI)*，[原文](https://arxiv.org/abs/2202.06934)、[代码](https://github.com/obss/sahi) | 切片推理与微调提高小目标有效分辨率；论文在 VisDrone/xView 验证。 | 给离线教师切片/高分辨率属于现实可行方案；可以让学生部署仍保持整帧低成本。 | VisDrone/xView 检测航拍地面目标。切片本身成熟，不能当新贡献；需证明蒸馏保留了切片教师的有效信息。 |

### 只证明存在、未达到可复现机制证据的相关论文

- Huang 等，2024，*RDR-KD: A Knowledge Distillation Detection Framework for Drone Scenes*，[DOI](https://doi.org/10.1109/LGRS.2024.3398140)。已核验 IEEE/Crossref 题名、作者与年份；未取得摘要/正文与官方代码。本轮不能据标题断言它是无人机本体检测或完全复现其结构。
- Salisu 等，2025，*Enhanced Drone Detection Model for Edge Devices Using Knowledge Distillation and Bayesian Optimization*，[DOI](https://doi.org/10.3390/engproc2025087071)。元数据与公开论文入口存在；本轮正文读取未成功。仅作为“无人机边缘蒸馏已有先例”的查新提醒，不能照搬未读方法。
- *CrossKD: Cross-Head Knowledge Distillation for Object Detection*，CVPR 2024，[DOI](https://doi.org/10.1109/CVPR52733.2024.01563)。本轮核验了正式题名/年份，尚未完成摘要与官方代码读取，不在当前首批实施清单内。

## YOLO：候选具体到可验证机制

### V1：困难负类感知、置信门控的教师蒸馏

**问题。** 当前单类 YOLO 的误报集中于直升机；此前仅二值 crop 复核器并未解决。现在官方数据原本就含 Drone/Helicopter/Bird/Airplane 的目标框，可把混淆类从“泛化背景”变成有监督的具体类别。[官方数据说明](https://github.com/DroneDetectionThesis/Drone-detection-dataset)

**候选机制。** 冻结 DINOv2/v3 提取目标 crop/邻域特征，训练五类线性教师头（四种物体 + 背景）。学生学习五类真标签，并模仿教师软分布；随后只将 drone 概率用作无人机报警。进一步只在教师对 drone 与 confuser 的边距大、或增强视图一致时给予较高蒸馏权重，避免把教师的不确定答案强行传给学生。

**已有方法与我们的改动。** soft/KL 蒸馏与可靠性加权已成熟。可探索的任务改动是把权重显式绑定到无人机—直升机/鸟/飞机的错误类型，且在固定无人机召回下优化 confuser FPR。现在不能宣称学术首创。

**先做的公平对照。** 同一五类数据、相同学生初值/更新步数：CE 真标签学生 vs CE+普通 KD 学生 vs CE+置信门控 KD。必须包含五类真标签的 CE 学生，否则改善可能只来自新增负类标注，不能归功于教师。

**当前 probe。** 主线程已准备 `experiments/scripts/vision_teacher_student_probe.py`：冻结官方 DINOv2 ViT-S/14 + 五类线性头，约 19k 参数 crop 学生；CE/KD 同初值、同采样、同 12 epoch。只读 train/validation，检测候选 score≥0.25 且 drone probability≥0.5 时保留。教师特征缓存后学生训练很轻。本轮不微调 DINOv2、不蒸馏完整 YOLO、不跑新增种子、不把 validation 当最终测试；结果另由运行目录报告。

**重要语义。** 目前输出是 `YOLO + 小 crop 学生复核器`，而不是“已经把大模型蒸馏成 YOLO”。这条 probe 先回答“教师软知识是否比同标签 CE 有用”。如果成立，再用较大 DINOv2-L/g 或 DINOv3-H+/7B 教师，并研究把 ROI 特征/混淆软标签放进检测器训练，减少额外部署模块。

**展示。** 同一完整视频三路：YOLO、CE 学生、KD 学生；每候选展示五类概率；按无人机尺寸分层的 recall、直升机/鸟/飞机 FPR、低 FPR recall、学生推理耗时。先看真实失败，不靠挑片段展示。

**难度。** 当前冻结教师 + 线性头 + crop 学生低—中；完整 YOLO 的 ROI/多头蒸馏中—高。主要成本是监督正确性和教师是否有用，不是 GPU 数量。

### V2：小目标定位分布与尺度选择蒸馏

**问题。** 当前 mAP50≈0.831、mAP50–95≈0.391，presence 召回很高，但框质量相对较弱。这比只追求“有无检测”更适合 LD/细粒度特征蒸馏。[本机结果](../vision_single_seed0_eval_v1/README.md)

**候选机制。** 先训练同域 YOLO11s/m 或较大教师，教师用高分辨率/局部切片获取更稳定小框；学生仍以 640 整图部署。分别蒸馏框分布和目标邻域 P3 浅层特征，对小目标给予权重，对定位不稳定框减少权重。FGD/LD/HMKD/SAHI 都是已有机制出处；新增假设是此选择性迁移能保留小框收益而避免额外切片部署成本。

**立即 probe。** 不先做完整跨尺度蒸馏。先在 train/validation 固定的尺寸分层样本上比较既有 YOLO11n 的 640、960、切片推理，检查教师信息是否可能存在；这只是分辨率能力检查，不能把同模型高分辨率推理称新算法。之后同输入尺寸的较大教师短训，确认定位优势再接 LD loss。

**关键对照。** 学生 CE、学生普通 feature MSE、学生 LD、学生尺寸/可靠性选择 LD；teacher 单独性能；同样学生部署分辨率。不能把学生也改成高分辨率后，将全部收益归到蒸馏。

**展示。** 同一小目标放大窗，GT/学生/教师框叠加；框中心与边长抖动曲线；mAP50–95 与目标像素尺寸、端到端 FPS/峰值显存的关系。

**难度。** 中—高；DFL logit 匹配、正样本分配和增强坐标对齐比普通 crop KD 更难。4 卡可扩大教师规模与分辨率，但不能消除接口工程。

### V3：视频一致性筛选伪标签，学生仍单帧部署

**问题。** 现有训练只抽帧，训练视频内还有大量连续帧；可以在不增加人工逐帧标注时利用教师生成监督。但时间一致性会让持续的直升机误报也变得“稳定”，所以不能仅凭连续存在就标无人机。

**候选机制。** 教师离线跨帧对齐目标，结合类别可信度、框稳定性和短轨迹一致性接受/忽略伪标签；confuser 语义拒绝与时间过滤分开。学生只学习当前帧，部署不用大教师。机制来源为 Soft/Efficient Teacher 的可靠性筛选、GLAD 的运动/局部处理；具体跨帧筛选仍需进一步查新。

**立即 probe。** 先在有原 MAT 标签的训练视频中隐藏一部分帧标签，检查普通教师伪标注 vs 时间筛选 vs 语义+时间筛选的真实框精度/覆盖率。这个离线 probe 无需训练学生，就能否定错误伪监督。

**关键对照。** 固定人工标签预算与未标注训练帧池：监督学生、同域 YOLO 教师自训练、基础教师伪标注、筛选伪标注。测试/validation 帧不进伪标签训练池。

**展示。** 伪标签接受/拒绝视频轨迹、每类错标签计数、标注预算—性能曲线。报告伪标签覆盖率与正确率，不能只显示少量漂亮框。

**难度。** 中；可以先离线生成与缓存，避免先搭完整 EMA 半监督训练框架。

## 60 GHz：教师学生路线与已有先例

| 论文 | 已核验机制 | 对当前环境的启示 |
|---|---|---|
| Rusu 等，2015/ICLR 2016，*Policy Distillation*，[原文](https://arxiv.org/abs/1511.06295) | 小网络模仿专家策略；把多任务专家汇总到单一学生。摘要实验在 Atari。 | 按负载/遮挡训练多个专家，再蒸馏一个带场景/历史条件的策略；教师集成本来就不新。 |
| Schmitt 等，2018，*Kickstarting Deep Reinforcement Learning*，[原文](https://arxiv.org/abs/1803.03835) | 用此前教师加速学生 RL；允许教师/学生架构不同，调节迁移使学生超过教师；支持多个专家。摘要实验为 DMLab-30。 | 学生不必被先验永久绑定；用教师作为训练初期辅助，最后继续 PPO，适合解释此前 full-action KL 过强的失败。 |
| Ross 等，AISTATS 2011，*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*，[原文](https://arxiv.org/abs/1011.0686) | DAgger 聚合学生自己导致的状态分布上的专家标注，处理模仿学习行动改变未来观测导致的分布偏移。 | 教师 rollout 上动作拟合准确不等于学生上线 queue 不积压；要在学生访问状态补标/继续交互。 |

另已核验 2025 WCNC 论文 *Knowledge Distillation-Based MAPPO Approach of Wireless Power and Spectrum Resource Joint Allocation for 6G Networks* 的题名与 DOI：[10.1109/WCNC61545.2025.10978810](https://doi.org/10.1109/WCNC61545.2025.10978810)。未读取全文和官方代码，仅作为“无线 RL 资源分配蒸馏已有先例”的提醒，不据此推导本项目具体结构。

### R1：多场景专家、状态相关的用户级选择性蒸馏

先用现有 PPO、反馈 MAB、validation beam 在训练/validation 轨迹中检验各自擅长的负载/遮挡状态；不是按测试成绩选教师。将完整动作先投影为用户边缘分布，学生只蒸馏有依据的用户优先级，在用户条件下保留 beam/relay/tracking 自由。若某状态专家差异明显，再定义由**可观测**队列、历史服务率、负载估计、阻挡代理驱动的教师权重。

贡献候选是“状态相关、动作结构相关的选择性转移”，不是“策略更大”。普通 continued PPO、单教师全动作 KD、用户级 KD 和多教师用户级 KD 形成紧凑对照。先同一单 seed 30k 额外交互，比较吞吐、平均队列、pending 与动作占比即可。验证阶段不需要新增多随机种子。

不能将读取隐含瞬时 CSI/未来到达的 privileged greedy 与现实可观测学生并列称公平 baseline。可把它作为离线上限/训练教师，但需要单独写清楚学生缺失的信息以及这种信息能否由历史预测。

**展示：** 在同一外生轨迹下网络调度动画、不同教师/学生的用户服务分布、queue 积压和先验权重随场景变换的曲线。**难度：** 中；当前动作/轨迹日志可复用，环境采样和状态解释比 4 卡训练更关键。

### R2：置信/积压触发的 Kickstarting，允许学生解除教师约束

把当前 prior 系数固定退火改为候选规则：教师与学生明显冲突、教师相关用户继续积压时减小监督；训练初期可信状态提供软目标，后期继续 PPO。来源是 Kickstarting，而具体积压/可行性触发规则是待验证任务改动。短试必须包含相同额外 PPO 交互/优化预算的固定退火对照；只看较低 imitation loss 不够。

**展示：** 教师约束系数、策略熵、beam/relay 占比与队列曲线，解释“何时遵循、何时离开教师”。**难度：** 中；比把 RL 策略扩大到亿级参数更契合小型状态任务。

## 4 卡资源应如何用

1. 先冻结小型基础教师完成能力筛查：真正的目标是确认信号存在，不是当前就证明大模型路线失败。
2. 教师有优势后，比较 DINOv2-S/B/L/g 或可取得的 DINOv3-L/H+/7B，优先离线缓存特征/软标签，再训练同一轻量学生。这样比较的是教师信息质量，不把部署端一起扩大。
3. 多卡用于较大教师推理、较高分辨率、完整检测器学生训练和独立机制分支；不把多卡等同于必须多 seed。每卡显存与模型实际峰值需实测，不能把四卡显存简单相加当单卡可用。
4. 在 RL 中可以并行采样多个训练场景或专家，但当前约 100k 参数 PPO 不是显存瓶颈；先验证场景专家是否互补、状态是否足够，再扩大模型。

## 首批实验顺序

| 顺序 | 小实验回答的问题 | 本机可行性 | 何时值得扩大 |
|---|---|---|---|
| 1 | 冻结 DINOv2 五类教师是否比小学生更会分无人机/直升机；KD 是否胜过同标签 CE | 已有脚本，当前正在准备运行，单 seed | 教师至少在关键 confuser 上有优势，学生 KD 能保留可用部分 |
| 2 | 640/960/切片是否确实改善小目标框，而非只增加错类报警 | 只推理，低成本 | 小目标定位质量提升且误报可控，再训练大检测教师和 LD |
| 3 | 普通伪标签 vs 类别/时间筛选有没有提升真实正确率 | 已有视频与 MAT，可离线检查 | 同时保留足够无人机标签覆盖，随后做小预算学生训练 |
| 4 | 多专家是否状态互补；用户级选择是否胜过全部动作模仿 | RL CPU 可短训 | 单 seed 已有改善或清楚可解释权衡，再做更多场景训练 |

当前无需根据这一轮文献立即确定最终选题，也无需强行包装失败方案。保留“问题→机制→短测→决定是否扩大”的节奏。论文事实、我们的候选与本机结果分别放在对应表格/目录，避免后续 PPT 混写。

本记录由 AI 辅助检索与整理；来源核验层级如上，尚未进行发表级系统查新。
