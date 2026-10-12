# YOLO / 无人机小目标检测：论文检索与可执行创新候选

日期：2026-10-12。范围按最新用户指示收窄为视觉；文件名沿用并行任务原名，**本报告不讨论 PA/DPD**。这是选题探索笔记，不是论文稿，也不认定下述组合具有学术首创性。本轮只完成论文与代码来源检索，没有在本报告中新增训练结果。现阶段采用单次短实验，后续 4×4090 可用于扩大教师与数据规模。

## 先明确当前瓶颈

现有 YOLO11n 的测试 presence recall 很高，但 mAP50–95 明显低于 mAP50，且误报高度集中在直升机。19,105 参数、从头训练的小 CNN 候选复核器未改善直升机误报；3×扩大 crop 还牺牲大量无人机召回。因此当前最值得检验的不是继续增加任意模块，而是三个具体问题：

1. **信息是否可分？** 24 px 左右目标的局部图像，在预训练视觉特征或多帧证据下，能否区分无人机和直升机？
2. **监督是否够明确？** 单类 detector 把鸟、飞机、直升机都当背景；现有 MAT 实际包含四类框，能否用这些现成标签提高近类区分？
3. **教师是否可靠？** 教师能否在小目标和负例上提供比学生更好的信号，而不只是把学生的误报再标成伪真值？

本项目证据入口：[连续测试](../vision_single_seed0_eval_v1/single_report_v1/README.md)、[复核器级联重评](../vision_single_verifier_probe_v1/cascade_reval_v1/README.md)、[因果 EMA 对照](../vision_single_temporal_exploration_v1/README.md)。这些已完成结果与本轮提出的新候选分开。

## 已核验的论文与代码

这里的“核验”指读取 arXiv 原始摘要/方法页与作者仓库 README，确认标题、方法和代码入口。除 GLAD、YOLOE、SAM 3 外，本轮未通读全文；不把摘要检索称作精确复现或系统综述。页面缓存保存在 `experiments/literature/vision_scan_20261012/`。

| 文献 | 论文做了什么 | 对当前项目的实际意义 | 代码和限制 |
|---|---|---|---|
| Guo 等，**Global-Local MAV Detection under Challenging Conditions based on Appearance and Motion**，T-ITS 2024；[arXiv 2312.11008](https://arxiv.org/abs/2312.11008) | 全局 YOLOv5s 搜索、局部高分辨率检测、运动辅助、Kalman 自适应区域、失败切换；重点是 air-to-air、复杂背景和小 MAV | “运动+外观”“自适应局部区域”已有明确先例；可以用作有实际无人机任务依据的方案和外部数据入口 | [GLAD](https://github.com/WestlakeIntelligentRobotics/GLAD)。作者 README **明确说只发布 basic code，完整 Kalman/自适应区域代码尚待公开**，不能宣传为完整一键复现；与本数据的 ground-to-air 拍摄分布不同 |
| Akyon 等，**Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection**，ICIP 2022；[2202.06934](https://arxiv.org/abs/2202.06934) | SAHI：切片推理和切片微调，增强小目标的有效分辨率 | 可作为高分辨率强参考；但切片只改善尺度，未保证近类区分，甚至可能放大负类误报 | [SAHI](https://github.com/obss/sahi)。论文评估 VisDrone/xView，主要是航拍地面目标，不能称作无人机本体结果 |
| Yang 等，**Focal and Global Knowledge Distillation for Detectors**，CVPR 2022；[2111.11837](https://arxiv.org/abs/2111.11837) | FGD 区分前景/背景、蒸馏重要像素与通道、传递全局关系，避免全特征均等模仿 | 支持“选择性蒸馏”，比简单套一个 MSE 特征损失更有根据；负例区域也需要处理 | [FGD](https://github.com/yzd-v/FGD)。原仓库依赖旧 MMDetection/MMCV，不是 YOLO11 插件；需移植损失/特征接口 |
| Wang 等，**CrossKD: Cross-Head Knowledge Distillation for Object Detection**，2023 preprint / CVPR 2024；[2306.11369](https://arxiv.org/abs/2306.11369) | 将学生 head 中间特征送入教师 head，使 cross-head prediction 模仿教师，缓解 GT 与教师预测冲突 | 说明检测蒸馏并非“温度 KL 随便加”；可参考预测蒸馏和标签冲突问题 | [CrossKD](https://github.com/jbwang1997/CrossKD)。原实现 GFL/FCOS/RetinaNet/ATSS、MMDetection 3.0rc；直接迁入 YOLO11 难度中等偏高 |
| Xu 等，**End-to-End Semi-Supervised Object Detection with Soft Teacher**，ICCV 2021；[2106.09018](https://arxiv.org/abs/2106.09018) | 教师置信度加权分类损失，box jitter 筛选可靠回归伪框，持续改善伪标签 | 伪框要按可靠性筛选，不能把所有大模型框当真；适合扩展未标注视频 | [SoftTeacher](https://github.com/microsoft/SoftTeacher)。原实现较重；这里只借鉴可靠性机制，不称精确复现 |
| Liu 等，**Unbiased Teacher for Semi-Supervised Object Detection**，ICLR 2021；[2102.09480](https://arxiv.org/abs/2102.09480) | EMA 教师学生和 class-balance，处理半监督伪标签偏差 | 警示“无人机多、直升机少”导致教师伪标签偏差；类别预算要明确 | [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher) |
| Liu 等，**Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**，2023；[2303.05499](https://arxiv.org/abs/2303.05499) | 语言引导开放词汇框检测，可输入类别和描述 | 冻结教师可同时提示 drone/quadcopter、helicopter、bird、airplane；先测分类 margin，而不是只问“drone” | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)。公开 Swin-T 权重；官方还给出 Transformers 入口。自然图像零样本能力不保证小于 24 px 的近类分辨能力 |
| Xiao 等，**Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks**，2023；[2311.06242](https://arxiv.org/abs/2311.06242) | 提示式统一 caption、detection、grounding、segmentation | 可作为另一种教师验证，特别是局部候选描述/分类；不能只因为会语言就假设准确 | [官方模型](https://huggingface.co/microsoft/Florence-2-large)。本轮核验论文摘要；未下载或跑权重 |
| Cheng 等，**YOLO-World: Real-Time Open-Vocabulary Object Detection**，CVPR 2024；[2401.17270](https://arxiv.org/abs/2401.17270) | region-text contrast、RepVL-PAN、开放词汇检测 | 更轻量的开放词汇教师/对照；也可直接小规模适配 | [YOLO-World](https://github.com/AILab-CVC/YOLO-World)。通用 LVIS 零样本结果不能外推为当前无人机准确率 |
| Wang 等，**YOLOE: Real-Time Seeing Anything**，2025；[2503.07465](https://arxiv.org/abs/2503.07465) | 文本提示 RepRTA、视觉示例 SAVPE、prompt-free LRPC；可重参数化到常规 YOLO 架构 | **视觉 exemplar** 可能比“drone”短词更贴近小目标；可用训练视频中的代表性无人机/直升机 crop 作视觉提示；也适合作为“基础模型适配→小模型”的近期路线 | [YOLOE](https://github.com/THU-MIG/yoloe)。公开 11s/m/l 与 v8s/m/l 权重，官方提供仅最后分类层训练的 linear probe 和 full tuning；现有环境已安装 Ultralytics，但需要检查版本/权重兼容，不保证即时可跑 |
| Oquab 等，**DINOv2: Learning Robust Visual Features without Supervision**，2023/2024；[2304.07193](https://arxiv.org/abs/2304.07193) | 1B ViT 教师及较小蒸馏模型，无监督通用视觉特征 | 冻结特征 + 四类 linear probe 是低成本“语义能否分开”的检验；比从头小 CNN 更好的教师能力前测 | [DINOv2](https://github.com/facebookresearch/dinov2)。可用小/中型公开权重；它只给特征，不直接输出四类概率，需要用训练标签训练一个小 head |
| Siméoni 等，**DINOv3**，2025；[2508.10104](https://arxiv.org/abs/2508.10104) | 大规模 SSL、Gram anchoring 维持 dense feature；发布 6,716M ViT-7B 与蒸馏 S/B/L/H 版本 | 后续 4×4090 可以测试大教师与小教师收益；先拿冻结 crop 特征，而非直接 full finetune 7B | [DINOv3](https://github.com/facebookresearch/dinov3)。**权重需申请访问**；已经公开代码和 HF/timm 入口，尚未在本机取得权重 |
| Carion 等，**SAM 3: Segment Anything with Concepts**，2025，2026-03 v2；[2511.16719](https://arxiv.org/abs/2511.16719) | 848M 共享视觉 encoder 的概念检测器与 memory video tracker；presence head 分离 recognition/localization；数据包含 hard negatives | 当前检索中最贴合“语义近类 + 多帧 + 大教师”的新路线；可提示“drone”及正/负 exemplars，再将可靠语义/轨迹知识传给小学生 | [SAM 3](https://github.com/facebookresearch/sam3)。官方要求 Python≥3.12、PyTorch≥2.7、CUDA≥12.6；**HF 权重 gated**。本机现有 Torch cu118 不能直接沿用，应新建环境；本轮仅核验未部署 |
| Gupta 等，**OW-DETR: Open-world Detection Transformer**，CVPR 2022；[2112.01513](https://arxiv.org/abs/2112.01513)；Zohar 等，**PROB: Probabilistic Objectness for Open World Object Detection**，CVPR 2023；[2212.01424](https://arxiv.org/abs/2212.01424) | 区分 known、unknown object、background；伪 unknown 标签或概率 objectness | 可用于将来未知飞行物拒识；**当前有鸟/飞机/直升机标签，首先是已知近类区分，不必直接升级为完整 OWOD** | [OW-DETR](https://github.com/akshitac8/OW-DETR)、[PROB](https://github.com/orrzohar/PROB)。本轮核验论文摘要和论文内代码链接，未跑仓库 |

## 六个具体候选：机制、差异与最小实验

### V1：四类负类监督 + 低误报校准（优先的小实验）

**机制。** 保留 MAT 的 DRONE/HELICOPTER/BIRD/AIRPLANE 四类框，不再仅把前三种非无人机物体视为背景。训练四类 YOLO11n，部署端只发无人机报警；同时保留类别 margin `p(drone) − max p(non-drone)` 作为复核依据。先做普通四类 baseline；若有收益，再测试针对 drone↔helicopter 混淆的分类 margin / contrastive loss。

**与旧方法的差异。** 旧 hard-negative 只是重采样高误报“背景”，旧小 CNN 从候选训练二分类；新方法显式告诉模型“这是直升机”，使用现成类别监督而非仅增加样本难度。

**最小实验。** 用原视频 split 和相同 sparse frame 清单，转换全部四类标注。YOLO11n 4 类与单类从相同 COCO 初始化、同 10–20 epoch 短预算比较；或双方都从已训 backbone 开始、重新初始化 class head，清楚记录 head 重置。以相同视频/帧和单 seed 评价 drone recall、直升机/鸟/飞机 FPR、confusion matrix、mAP50–95。阈值用 calibration，而不以测试误报倒推。

**PPT 展示。** 同一帧显示“原模型：无人机；四类模型：直升机”，再展示类别混淆矩阵与每类 FPR。

**风险。** 这首先是更合理的监督 baseline，本身不应称新算法；小目标的信息可能仍不足；四类 head 的初始化和类别不均衡可能导致短训不公平。若普通四类足够好，可以据此发展“近类错误引导的教师蒸馏”，不必强加模块。

### V2：冻结视觉教师 + 近类 margin 复核（先验证教师，随后蒸馏）

**机制。** 以预训练 DINOv2/YOLOE 为当前教师入口，后续 DINOv3/SAM 3。对 YOLO 候选框提取局部与少量上下文证据，由教师判别四类，学生接收教师四类软分布或 feature loss。目标是只在易混淆候选上增加强语义监督，最终部署仍只有 YOLO11n 或轻量分类头。

**最小实验 A：教师能力前测。** 训练视频产生 1,000–3,000 个按视频、类别和目标大小均衡的 crop；冻结 DINOv2 ViT-S/B 特征，训练线性四类 head；验证视频上与旧 scratch CNN、单帧 YOLO score 比较。目标 crop 和适度上下文 crop 固定两种，不搜索十几种边长。实际级联用 **预测候选框**；GT crop 只作为“信息是否可分”的诊断上限，不计部署成绩。

**最小实验 B：蒸馏。** 教师确实比学生能区分直升机后，冻结教师，YOLO11n 继续训练 5–10 epoch：普通 CE/BCE 对照、普通 KL 蒸馏、近类 margin/不确定性加权蒸馏。教师高 drone 分数但四类 margin 低的候选降权或忽略；监督 GT 永远不被低可靠教师强行覆盖。

**与旧方法的差异。** 不是再训一个 19k 参数 scratch CNN，而是先利用预训练的强视觉表征；也不是“所有大模型框都当 GT”，而是把教师是否真正减少混淆当作前置实验问题。

**PPT 展示。** 离线教师→候选软标签/特征→小学生，部署页只显示小学生；并展示“同样小模型参数量、直升机误报减少”与教师运行成本。

**风险。** 教师大不等于当前任务强；24 px 信息不会由 crop 放大凭空补回。DINO 特征加 linear head 是教师适配，不是零样本能力；只输出硬伪框叫伪标注，用软分布/feature loss 才叫知识蒸馏。当前这些候选尚未验证。

### V3：连续候选轨迹的轻量时序分类（比 EMA 多一步）

**机制。** 把候选框关联成短 tubelet，汇集最近 5 或 9 帧的外观 embedding、box 位移/尺度和帧差，训练小 GRU/1D-TCN 输出四类或 drone probability。使用因果窗口，不用未来帧；仅 track 同一目标，避免全帧最大置信度把不同物体混合。可用冻结教师 embedding 做短测，后续蒸馏到轻量 CNN+TCN。

**文献依据。** GLAD 明确将外观与运动结合；SAM 3 已有语义检测 + memory tracker；FGFA/MEGA 等视频检测已有特征跨帧聚合。因此“加时间”不是首创，课程贡献应是针对近类持续误报的候选级时序区分及部署取舍。

**最小实验。** 选训练/validation 中各类固定连续片段，缓存 YOLO 候选及冻结 embedding。对比单帧 head、EMA、同参数 TCN 使用重复同帧输入、TCN 使用真实 5 帧。训练 3–5 epoch 单 seed；报告 drone recall、类 FPR、误报持续时间、报警切换/分钟、首报延迟。重复同帧对照用于判断收益是否来自真正时序信息而非更大 head。

**PPT 展示。** 单帧误报与连续判别并排视频；下方展示预测曲线、类别证据和报警状态。

**风险。** 直升机也会持续运动，EMA 已无法去除该误报，轨迹连续本身不会保证类别正确。相机平移、背景或单视频 id 可能成为捷径；需 video split，且不同帧来自同一个对象。关联错框会制造新误差。

### V4：高分辨率/局部教师 → 小模型尺度与近类选择性蒸馏

**机制。** 离线教师在原图高分辨率或 SAHI/局部 crop 下输出更好的框、四类软概率和特征；学生仍在 640 输入。FGD 风格对 tiny foreground 与高分近类负候选给予选择性蒸馏，而不是所有背景统一模仿。先区分“高分辨率本身”与“蒸馏权重机制”。

**最小实验。** 先用同一个现有 checkpoint，仅在 validation 小目标帧比较 full-frame 640 与固定切片/局部推理，统计定位 AP 和新增负类误报；不训练即可排查切片是否值得。教师若有可用信号，再用 YOLO11s/m 四类教师生成缓存；学生比较无蒸馏、均匀蒸馏、tiny/近类区域加权蒸馏，各 5–10 epoch。

**后续扩大。** 4×4090 可完整训练 YOLO11l/x 或较大视觉教师，高分辨率批量缓存和学生训练并行；但不能把较大 YOLO 本身等同 foundation model，二者应分列。

**风险。** 更多算力可改善定位但也能放大直升机误报；教师与学生尺度/网络不同需 adapter。蒸馏权重需尽量少，不在测试集搜索。

### V5：稀疏关键帧教师 + 可靠轨迹伪标注

**机制。** 在少量关键帧运行 Grounding DINO/YOLOE/SAM 3 语义教师，在连续未标注训练视频中传播；只保留跨帧一致、类别 margin 足够和重新检测一致的 pseudo track。无人机、直升机的正/负类信号一起进入学生，不只扩大 drone 正例。

**最小实验。** 只在 train 视频中模拟 10% 有标注帧、其余 GT 隐藏用于最终伪标签质量审查。比较同标注预算纯监督、同域 YOLO 自训练、基础教师伪标注、带轨迹/不确定性筛选的教师伪标注。第一阶段只做 100–300 帧伪标签准确率、coverage、每类混淆和人工修正数，不急于完整 student training。

**PPT 展示。** 标一帧→教师轨迹扩展→筛选→轻量 student，展示每分钟人工标注量减少。

**风险。** 已有全标签数据上的“隐藏标签”只是标注预算模拟，不是真正取得新数据；教师传播错误会持续污染整段视频；若教师只是跟踪 GT 框，不能用该轨迹作为部署检测准确率。Soft/Unbiased Teacher 已有半监督框架，单纯自训练不算新算法。

### V6：未知飞行物拒识 / 风险触发第二级（后备）

**机制。** 区分“无人机”“已知非无人机”“未知/不确定”。margin 低或 embedding 远离四类原型时返回 uncertain，不强行给 drone alarm；也可以只在 uncertain 时调用较重时序/复核模块，实现平均计算量控制。

**最小实验。** 用 train 特征建立类原型，validation 决定距离/margin 阈值；用外部未见物类/恶劣背景检验 unknown rejection，同时报告 drone recall 损失、拒识比例、复核调用率。没有外部未知数据时，先做 leave-one-negative-class-out，明确这是模拟而非现实完整开放集。

**风险。** 当前标签已含主要负类，完整 OW-DETR/PROB 工程收益可能不足；拒识阈值降低误报同时降低召回，必须呈现同 recall/FPR 工作点与额外算力，而非只给成功视频。当前优先级低于 V1–V4。

## 现阶段建议的探索次序

| 优先 | 小实验 | 为什么先做 | 继续条件 |
|---|---|---|---|
| 1 | 四类标签恢复 + YOLO11n 短训 | 现成数据、直击核心直升机混淆，成本最低 | 在相近 recall 下 helicopter FPR 有下降，或明确证实四类监督仍不足 |
| 2 | 冻结 DINOv2/YOLOE 教师能力检查 | 判定“大模型教师”是否有真正可传递的信息 | 预测候选 crop 中教师较学生更会分近类；若只在 GT crop 好，先解决定位 |
| 3 | 真实 5 帧与重复单帧的 TCN 对照 | 利用已有连续视频，比 EMA 更明确检验时序是否有用 | helicopter FPR/误报时长改善且首报延迟可接受 |
| 4 | 固定切片/局部高分辨率推理诊断 | 不重训即可检查 tiny 信息瓶颈 | AP 改善且不增加大量负类误报，再做教师蒸馏 |
| 5 | SAM 3 / DINOv3 教师 + 学生完整实验 | 有后续4×4090资源可支持，近期模型也有可讲机制 | 先取得官方权重，完成 teacher能力/伪标签质量检查，再扩大 |

**目前不锁定唯一视觉创新。** 最有潜力的一条综合路线是“近类负样本意识的离线视觉教师 + 候选级时序证据 + 小模型部署”；但应该先分别验证教师、四类监督和时序的作用，不能把三种未经验证的模块叠在一起后称成功。若单一机制已有效，就保留更简单的系统。

## 数据与检索边界

- 当前 Svanström 数据官方 README 明确 Video labels 为 Airplane、Bird、Drone、Helicopter，原 MAT 已能读取四类；优先利用现有标注。来源：[官方数据](https://github.com/DroneDetectionThesis/Drone-detection-dataset)。
- GLAD 的 ARD-MAV 可作为复杂背景/small-target 外部正例诊断。论文给 60 视频、106,665 帧，当前 README 给 107,497 帧，存在版本计数差异，下载后按实际 manifest 核对；它不是专门近类负例数据。此次仅核验公开入口，没有下载全数据。
- Drone-vs-Bird、Anti-UAV 等可进一步取外部验证，但当前不可把未下载或未解析的数据计作已具备资源。VisDrone/UAVDT 的主任务为无人机视角地面目标，不混称无人机本体。
- 这次检索是针对当前瓶颈的候选扫描：arXiv 原始页面、论文方法 HTML、作者 README；未声称覆盖全部2026文献，也未宣布全球首创。旧 API 广泛检索返回大量不相关综述，未作为机制证据。可公开分享的都是论文/作者代码链接，实验私有标签没有外传给模型服务。
- 本报告由 AI 协助整理，论文结论与本项目的新机制假设、既有实测、尚未运行的实验已分别标注。
