# 三方向真实方法清单与选题证据

审查时间：2026-10-11。按用户最新要求，后续优先现有产物验证和 Demo，不再扩展多种子。本记录只读已有代码、论文、协议、报告和状态，不新增实验或审计矩阵。已完成的历史多种子实验保留原有解释，未完成的计划不算结果。本文由 AI 辅助核对，不代表用户已阅读原论文，也不替代最终交付审计。

## 怎样称呼这些 baseline

“原仓库运行”表示直接使用作者的模型/训练流程，但数据、规模或超参数可以与原论文不同。“论文方法适配”表示有明确论文/代码来源，当前实现作了可说明的改动。“自建对照”表示本项目定义的启发式、结构或评价参照。只有数据、实现、训练与评估设置均对齐，才可称“严格复现原论文结果”。目前三条路线均不应整体使用最后一种表述。

## PA 建模 / NN-DPD / MoE

| 实际方法 | 来源类别 | 已有证据及准确名称 |
|---|---|---|
| OpenDPD 原 GRU23 PA → GRU15 DPD | 原仓库流程运行 | `opendpd_gru_full_150ep_seed0` 已完成 PA、DPD 各 150 轮，用官方 APA_200MHz split；PA 按验证 NMSE、DPD 按验证 ACLR 选模。可称“OpenDPD GRU 端到端 DPD 基线”，不能称原论文同设置复现。 |
| 记忆多项式 MP | 经典论文方法族 / 上游 basis 复用 | 原较小 MP 探针和新版全量 MP 均有保存结果；新版在 APA A/B、DPA160 的传统参考已完成。 |
| 因果 GMP | 论文方法适配 | 复用 OpenDPD GMP basis，删除需要未来输入的项，用内存有界 QR/SVD 等价求解；新版三个数据集传统参考已完成。它与上游允许 look-ahead 的 GMP benchmark 不同。 |
| 普通 GRU、三 GRU 软 MoE | 原 GRU backbone + 自建训练/组合对照 | `dpd_candidate_v3` 及 `dpd_candidate_frame200_v1` 已完成源建模和双向适应。MoE 是联合训练的稠密三专家组合，每个专家都执行。 |
| GRU head-only / MoE gate-only | 自建机制对照 | 旧候选中两者均更新 48 个存储标量；只更新 gate 的优势未获支持。 |
| 均匀专家、当前包络 gate、显式历史包络单 GRU | 自建消融 | `pa_source_mechanism_v1` 已完成源建模，不能将其结果直接当成跨采集适应结论。 |
| 约 2,700 参数 GRU/LSTM/因果 TCN/均匀专家/包络 MoE | 方法族 / 改造 / 自建对照的计划矩阵 | `pa_paper_baseline_v1/protocol.json` 固定了设计，但读取时神经状态仍为 `prepared`。不能把原 45 个 source + 96 个 adaptation 计划称为已完成。 |

来源：OpenDPD commit `256f856b2456c126cdd87d0a7ca06fe227b664fb`；[框架论文](https://doi.org/10.1109/ISCAS58744.2024.10558162)；[GMP 论文](https://doi.org/10.1109/TSP.2006.879264)。原 OpenDPD 论文的 §II、§III 和 Table I 已做全文文本核对，其主设置为 DPA_200MHz、约 500 参数、100 epoch、frame50/stride1、5 个种子，并含硬件测量。本机完整 GRU 使用 APA、不同规模和 150 轮，只有 surrogate 评价。后来仓库中约 2,700 参数、300 epoch 的 benchmark 是另一个公开代码 benchmark，同样不能当成本机结果。

最有用的已完成锚点：原 GRU PA 对真实 APA test 的全局 NMSE 为 **−38.8314 dB**；同一冻结 PA surrogate 内，无 DPD / 有 DPD 的线性目标 NMSE 为 **−13.8764 / −38.6094 dB**。后者证明代码级级联效果，不能声称硬件 ACLR/EVM 改善。[完整结果说明](<D:/vs code/智能通信网络大作业/experiments/notes/opendpd_gru_full_baseline.md>) 已保留频谱、波形视频和指标定义。

**创新结论：尚无已证明的 MoE 新机制。** 旧 v3 的 gate-only 在同 48 标量更新对照下弱于 head-only；源消融不支持“15 点历史 gate 必要”。换用 frame200 原 GRU 配方后普通 GRU 明显更强，旧配方的 MoE 差值不能推广为 MoE 优于强 GRU。A/B 输入完全相同、输出不同，且没有温度、偏置、设备等完整条件元数据，只能称跨采集变化。旧适应还用了完整 target validation，不能说总标注成本只有 1,024/4,096。

可用贡献表述：“比较经典多项式、强 GRU 与专家组合，在相同测量数据上的建模和有限适应收益，解释 gate-only 失效及配方敏感性。”如果选择课题 5，正向 PA 建模就已对题；如果选择课题 3，使用已完成的 GRU PA→DPD 级联，所有补偿图必须注明 surrogate 范围。**首次 MoE-PA、首次 gate 微调、硬件线性化成功均不能写。** 已有 [MoE PA 线性化先例](https://doi.org/10.1109/TMTT.2023.3341616)。

证据入口：`notes/opendpd_gru_full_baseline.md`、`notes/dpd_candidate_probe.md`、`notes/pa_source_mechanism_ablation.md`、`runs/dpd_candidate_frame200_v1/summary.json`、`runs/pa_paper_baseline_v1/protocol.json`。

## 60 GHz 定向网络调度

| 实际方法 | 来源类别 | 已有证据及准确名称 |
|---|---|---|
| 作者 DRL 短试 | 原仓库短运行 | 原 `mainDRL.py` 的短 smoke 只说明环境/流程跑通，不是完整作者论文复现。 |
| 标准 PPO | 论文方法适配 / 独立实现 | 作者环境与动作语义保持，上层重写标准 clipped PPO、old logprob/value、GAE 和 time-limit bootstrap；35 维输入、98,613 参数与作者 20 维、float64 网络等设置不同。三个 360k 训练已完成，30k 是预算诊断。 |
| PPO + 全动作先验 KL | 自建任务适配候选 | 与标准 PPO 的同预算 30k 对照已完成；不是 PPO/先验引导的首创。 |
| fixed_beam2、validation_selected_beam3 | 自建可观测启发式 | 按队列 × 历史速率选择用户，direct-only，并有 tracking 启发式。beam3 只在验证场景选择。两者都不是 round-robin/固定 TDMA。 |
| 反馈型 scheme4 TS | 作者方法适配 | 源 `get_MAB_scheme_setting.py` 的 scheme4：MaxWeight 用户、无 relay、码本 TS、tracking。本轮用执行链路 Reff/MCS 反馈更新，隔离策略 RNG，推迟初始探索 tracking；不能称逐行作者复现。 |
| privileged greedy | 自建诊断参考 | 读取真实当前 pathloss/信道，direct-only、myopic；额外信息必须标明，不作公平排名或理论上界。 |

来源：[作者论文 arXiv:2207.03526](https://arxiv.org/abs/2207.03526)、[作者仓库](https://github.com/yzhang417/DeepRL-mmWave-MANET)，commit `f9e5e93caccf53aa3b2f4a252d3ecda6164a059d`；标准 PPO 方法入口为 [PPO 论文](https://arxiv.org/abs/1707.06347)。源码 scheme6 还包含 relay TS，是完整 MAB 方案，但本轮没有严格复现它。冷启动最小计数、relay/codebook 索引、隐藏 outage 信息和预训练统计缺失等问题记录在 `source_mab_feasibility.md`。

当前正式结果已完成 **800 行锁定评价、报告、图和四个回放视频**。PPO 在 topology12 的均值优于现有非特权参考，在 topology13/shift13 则弱于验证选 beam 和反馈 TS。例：test12 PPO360k / feedback TS 吞吐 **0.914682 / 0.810273 Gbps**，test13 为 **0.953174 / 1.003007 Gbps**。这些数字来自三个冻结训练策略的均值，Demo 的单轨迹不能代替均值。反馈 TS 有更丰富的跨时隙后验/服务统计并在线更新，不能称与 PPO 同信息、同离线预算。20 秒仿真不能证明长期队列稳定，pending 不是丢包率，已交付延迟具有删失。

**创新结论：全动作队列先验 KL 未获支持。** 同预算 30k 的两个测试场景中，候选吞吐更低、队列和 pending 更高。训练更久的 PPO 属于完善 baseline，不能算新算法。可用贡献表述：“在锁定外生轨迹下，比较 RL、队列感知启发式和因果反馈 TS 的场景依赖及失效原因。”网络故事最贴题，但算法创新仍需单独提出并验证。

证据入口：[正式 REPORT](<D:/vs code/智能通信网络大作业/experiments/runs/rl_paper_baseline_locked_v1/REPORT.md>)、`notes/rl_candidate_probe.md`、`runs/rl_paper_baseline_locked_v1/source_mab_feasibility.md`、`scripts/rl_candidate_probe.py`、上游 `get_MAB_scheme_setting.py`。

## YOLO 无人机本体检测

| 实际方法 | 来源类别 | 已有证据及准确名称 |
|---|---|---|
| COCO 预训练 YOLO11n | 官方工程基线运行 | 使用 Ultralytics 官方模型/训练器，在 Svanström RGB 数据训练。它不是该数据集原论文的 detector 复现。旧 baseline 已完成，按预设 patience10 在 21 轮结束。 |
| uniform 续训 | 自建等更新预算控制 | 从同一 baseline checkpoint 继续 8 轮，固定正例/负视频配额。 |
| hard-negative mining 续训 | 已有机制的任务适配 | 只在 train 负帧挖难例，与 uniform 同训练更新预算，不是新 HNM 算法。 |
| causal 2-of-3 / 3-of-5、EMA | 自建后处理对照 | 对单帧置信度做因果聚合，仅改变 presence 决策，不改变框或 mAP，也不是 learned temporal detector。旧连续视频诊断已完成。 |
| 新数据 YOLO11n seed0 | 官方工程基线运行 | 265 段此前未用视频、6,360 抽帧；视频互斥 train/val/cal/test。按80轮上限在58轮自然早停，验证集选中43轮。792抽帧test mAP50=.830961，mAP50–95=.391397；10,357连续帧和四类完整demo均已完成。其余五模型未运行。 |

来源：[官方 YOLO11 文档](https://docs.ultralytics.com/models/yolo11/)、[Svanström 数据官方仓库](https://github.com/DroneDetectionThesis/Drone-detection-dataset)，固定 commit `442d0c6750708cc221636f7a1fd31cbd8cb0aeb8`。数据仓库 README 给出 [多传感器数据论文](https://www.sciencedirect.com/science/article/pii/S2352340921007976)、[检测/跟踪论文](https://arxiv.org/abs/2007.07396)和 thesis。课程 PPT 没有指定其中某个 detector 或 YOLO 版本；YOLO 工程 baseline 完全可以满足课程要求。

**创新判断：当前 EMA 是有收益的已有方法对照，新的视觉机制尚未实现。** 新连续test中，固定.25召回99.58%、FPR2.76%；EMA α=.6保持召回，FPR2.44%，切换12.17→5.56次/分钟。原误报147帧中直升机场景占142帧，EMA后仍有128帧，因此持续混淆比单纯报警抖动更值得研究。旧 HNM 的 test mAP50 比 uniform 高 **0.02150**，但低误报校准操作点两者 TP/FP 完全相同（TP6、FP2），F1校准后HNM反而更弱。新视频是video-disjoint；没有外部数据集测试。

可用贡献表述：“构建真实无人机与鸟/飞机/直升机场景的检测及校准流程，在连续视频展示漏检、复杂背景误报和时序后处理边界。”新算法候选可围绕低 FPR 与报警延迟，但当前尚未确立。**换 YOLO 版本、增加数据/种子、COCO 微调均是 baseline 完善，不自动构成算法创新。**

证据入口：[旧候选对照](<D:/vs code/智能通信网络大作业/experiments/notes/vision_candidate_probe.md>)、`runs/vision_video_eval_baseline_v1/REPORT.md`、`data/vision/paper_v1/README.md`、`evaluation_protocol.json`。

## 当前选题结论

以“已有证据能讲清楚、PPT 最重要、剩余四周且主要一人训练”为条件，**视觉方向适合优先验证和 Demo**：真实视频最直观，官方基线维护成本低，数据与误报/漏检问题容易向老师解释。选择前必须看到扩大数据上的实际测试和完整演示，不能仅凭旧成功画面承诺创新。

**PA 是交付风险最低的备选**：强 GRU、传统 MP/GMP、真实 PA 数据和 PA→DPD 视频已可用，实验闭环比 RL 可控。若做 MoE，当前负结果要求把贡献收窄为条件与消融，而不能继续宣传 gate-only 优势。若老师重视算法正收益，目前不能凭现有资料宣布 PA/MoE 创新已成功。

**RL 现有 baseline 最完整，也最贴通信网络**：报告和四种场景视频已可展示，且 PPO/TS/固定码本之间的场景反转是清楚的研究现象。其代价是信息公平性、随机训练波动和有限仿真范围更难讲，完整作者 scheme6 仍未复现。继续做新 RL 算法比使用已有结果做严谨课程报告风险更大。

三方向目前都有可展示的课程工作，**都没有已验证且可声称学术新颖性的算法贡献**。课程 contribution 可由可重复实现、严格比较、失败机制分析和可视化构成；若用户要求“自己的方法取得正收益”，还需先在验证集确定一个小而明确的机制，再做独立评估。大模型 finetune 本轮没有执行，也不列为已完成创新。

