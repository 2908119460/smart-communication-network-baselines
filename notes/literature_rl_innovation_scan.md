# 60 GHz / mmWave RL：新候选机制扫描

更新：2026-10-12。目的为课程项目的机制探索与 Demo，不是系统综述，也不将“找到可借鉴论文”视为我们的创新已有效。本文使用 academic-research-suite 辅助检索、核对与整理；用户尚未声明已阅读这些论文。

## 本轮范围与来源

优先考虑能够接到现有作者仿真环境、35 维状态、180 个合法联合动作和标准 PPO 上的机制。保留来源为 arXiv 官方摘要/HTML正文、论文作者仓库、正式 DOI/官方页面。排除需要全新物理层仿真、雷达数据或一开始就做多智能体环境的路线。检索窗口主要为 2022—2025，经典算法保留作机制依据。

2026-10-12 的 arXiv 首次检索 `millimeter wave resource allocation reinforcement learning` 显示 11 条结果，覆盖 2018—2025。`mmWave reinforcement learning scheduling` 起初超时，但后来的公开页下载成功，缓存显示13条（2018—2026），补入2025 POMDP跨层调度/世界模型以及2024版MARL-IAB；两篇2026多模态ISAC跟踪论文因偏感知/物理层相位设计，不作为本轮主候选。`wireless Lyapunov reinforcement learning` 和 `wireless knowledge distillation resource allocation` 遇到搜索超时/429；Bing 返回词典和泛毫米波页面，未作为研究证据。因此这份报告是一轮可核验的有界扫描，不声称完整覆盖近期文献，也不据此声称首次提出。通过已获结果和全文参考链取得相关论文的官方页面，并核对其标题、作者、摘要；两篇关键近年论文读取了 HTML 方法章节。

原始来源缓存：`experiments/literature/rl_scan_20261012/`。通过抓取脚本保存的每份成功来源都有 `.html` 与记录 URL、内容 SHA256、HTTP 状态和可读文本的 `.json`；调度检索页 `search_mmwave_sched.html` 是单独下载的原始页。本报告的来源支持范围见下表；未核对正式出版页的条目，按 arXiv/作者自报出版状态写，不自行补 DOI。

## 核验文献与可借鉴内容

| 来源 | 核验范围 | 论文做了什么 | 对当前项目的含义 |
|---|---|---|---|
| [Zhang & Heath：Joint User Scheduling and Link Configuration，2022](https://arxiv.org/abs/2207.03526) | 官方摘要、作者仓库；arXiv Journal reference 为 IEEE TWC 2022 | PPO 联合调度用户、relay、codebook、tracking；MAB 方法把动作拆成子动作并用 MaxWeight/Thompson sampling | 我们的环境来源；“分解动作”本身已有先例，用户先验不能宣称首次分层调度 |
| [Wu 等：URLLC-Awared Resource Allocation for Heterogeneous Vehicular Edge Computing，2023](https://arxiv.org/abs/2311.18352) | 官方摘要 + [HTML §III-G、§IV-A/IV-B](https://arxiv.org/html/2311.18352v1) | DSRC/mmWave/C-V2I 异构卸载；MGF 随机网络演算描述时延要求；Lyapunov 将长期队列问题转为短期问题后交给 DRL | 支持“队列漂移/约束意识”的设计动机，但我们不直接复现其卸载网络或稳定性证明 |
| [Badnava 等：Neural-Enhanced Rate Adaptation and Computation Distribution，2025](https://arxiv.org/abs/2505.13337) | 官方摘要 + [HTML §V-A/§V-B](https://arxiv.org/html/2505.13337v1)；摘要 Comments 自报 TMM accepted | mmWave 多用户 VR；PPO/PPG 式多阶段学习；R1C2/C1R2 级联用先决动作指导后续动作；动态 Lagrangian 系数处理 QoE 约束 | 明确提醒：条件动作级联已有近年先例；可把“用户→链路”作适配并验证分解顺序，而不能只给双头网络改名 |
| [Guo 等：Joint Design of Access and Backhaul，2023](https://arxiv.org/abs/2308.08707) | 官方摘要 | MADRL 联合 user association 与 wireless backhaul 资源，适应动态 access 状态 | 多智能体有应用先例，但当前仅一个集中决策控制器，改为 MADRL 需重定义职责/信息/奖励，暂不优先 |
| [Yao 等：DRL Resource Allocation and Beam Management under Location Uncertainty，2022](https://arxiv.org/abs/2204.10984) | 官方摘要；Comments 自报 ISCC 2022 accepted | UK-means 减少定位不确定性，再用 DRL 动态资源分配 | 不确定观测是真问题；当前环境的全用户遮挡开始信息是作者假设，适合做因果历史/观测退化测试 |
| [Achiam 等：Constrained Policy Optimization，ICML 2017](https://arxiv.org/abs/1705.10528) | 官方摘要及 accepted metadata | 显式 reward 与 cost constraint；带近约束满足保证的 policy optimization | 简单加队列 penalty 不等于 CPO；若以后做显式 tail constraint，可借鉴 CMDP/dual 形式，先不搬完整二阶更新 |
| [Rusu 等：Policy Distillation，2015/2016](https://arxiv.org/abs/1511.06295) | 官方摘要 | 将强 RL 策略蒸馏成更小网络；也可聚合多个任务教师 | “大教师+小策略”是已有通用机制；我们应研究老师来源、条件动作知识或跨拓扑能力，而非仅做压缩 |
| [Pinto 等：Asymmetric Actor Critic，2017](https://arxiv.org/abs/1710.06542) | 官方摘要 | critic 训练时读模拟器全状态，actor 只读部署时部分观测；domain randomization 做 sim-to-real | 训练阶段 privileged 信息可合法用于 critic；必须严格限制推理 actor，避免将 oracle 策略当公平 baseline |
| [Kumar 等：RMA，RSS 2021](https://arxiv.org/abs/2107.04034) | 官方摘要与 RSS metadata | base policy + adaptation module，模拟训练后在线适应未知动力学 | 跨负载/遮挡适应可借鉴“历史→隐条件”模块；它来自机器人，不直接证明无线环境收益 |
| [Wang 等：Cross-Layer Design for Near-Field mmWave Beam Management and Scheduling，2025](https://arxiv.org/abs/2511.12470) | 官方摘要；Comments 为向 NeurIPS2025 AI4NextG Workshop 投稿，不写成已录用 | POMDP；动态决定何时及多大力度beam训练，联合发射功率，权衡pilot开销、数据服务和能耗 | 可以借鉴“tracking是否值得”的队列/观测置信度门控；当前环境不是near-field，不复制其85.5%数字 |
| [Wang 等：World Model-Based Learning for Long-Term AoI Minimization，2025](https://arxiv.org/abs/2505.01712) | 官方摘要 + [HTML §III、Table I](https://arxiv.org/html/2505.01712v2)，v2=2025-08-01；未显示正式期刊录用 | mmWave V2X；RSSM学习动态模型，在可微想象轨迹中学link scheduling；Sionna物理/射线追踪仿真 | 是符合4卡资源的较大路线；先做短期队列/服务预测，确认模拟模型有用，再考虑imagined rollout |
| [Gahtan 等：Using DRL for mmWave Real-Time Scheduling，2022/2023](https://arxiv.org/abs/2210.01423) | 官方摘要，v2=2023-02-18 | AARL在10/48/96条link mesh中决定激活集合与功率，强调严格timeslot内决策速度 | 老师/学生路线应同时展示推理时延与闭环质量；当前5用户小网络本来就快，不能只靠压缩参数讲贡献 |
| [Zhang & Filippini：Mobility-Aware Resource Allocation for mmWave IAB，2022/2024](https://arxiv.org/abs/2205.06011) | 官方摘要，v2=2024-04-23；Comments 自报即将发表于IEEE/ACM ToN | MARL流路由/link scheduling，应对用户移动、障碍阻断、interference、relay buffer；full/half duplex及agents协调 | 比“加多agent”具体得多；若后续没有真正多控制节点，先不复现 |

## 候选机制：先验证问题，再验证结构

### A. 队列漂移 + 尾部积压意识（本机优先）

问题：当前 baseline 的训练 reward 实际主要奖励已交付吞吐，出现“交付延迟下降但 pending 上升”的删失现象；平均吞吐也可能掩盖某个用户积压。参考 Wu 等的 Lyapunov-DRL 和 Achiam 等的显式约束思想，先用两种小改动探测方向。

1. 漂移奖励：保留原 throughput reward，加入 `η·clip(L(q_t)-L(q_{t+1}),-1,1)`，其中 `L(q)=0.5·mean((q/Qref)^2)`。
2. 尾部积压代价：`r'=r-η·mean(clip(q_{t+1}/Qref-1,0,5))`，只对超过阈值的队列施加代价。

本机已完成：Qref=2,000 包/UE，η=0.02 固定，不做参数网格；从同一 360k PPO 与 Adam 状态 warm-start，各再训练 30k；单训练 seed=0，使用完全相同的训练场景/RNG重启规则与优化预算；三个独立 eval 场景各一个 environment seed=41000，保留 NPZ 队列/动作。这里只测“奖励中加入积压信号是否有希望”，若毫无改善，不必马上上 CVaR/复杂分布式 critic。

结果已保存于`experiments/runs/rl_queue_risk_explore_v1/`，三支全部exit0，9行评价，总用时525.5秒。两种固定形式没有整体支持：高负载topology12的continued PPO为0.9822 Gbps、平均队列13,276；drift为0.9291/19,520，tail penalty为0.9403/18,767。tail在topology13最差用户均队列下降22.11%、队列P95下降25.89%，但平均队列上升5.79%，是局部trade-off。奖励附加项绝对均值仅相当于raw reward均值的0.0843%/0.0238%；tail阈值只在10.11%训练时隙触发。下一轮应优先验证历史/条件动作或强教师，而不把本轮固定reward称为已成功创新。此负结果不否定经过风险训练分布、自适应dual设计的完整约束RL。

核心差异：漂移鼓励减少积压变化；tail penalty 直接处罚持续大队列。它们都不是新理论算法，不是 CVaR，不是 CPO，也不保证队列稳定。继承的 critic 原来预测 throughput return，新 reward 的价值尺度/语义不完全一致，短续训需看 value loss 与奖励分量，并明确这轮 warm-start 的边界。

演示：节点队列颜色、最差用户队列曲线、pending、已交付 P95 与末尾 pending age 同屏；显示吞吐换取尾部积压改善的 trade-off。

后续 4 卡：若方向支持，可做自适应 dual 队列预算而非任意固定权重，并比较“固定 penalty—自适应 dual—纯吞吐 PPO”；或者用分位数 critic 学到对尾部成本的响应。GPU主要帮助并行环境和更多场景，小 MLP 本身不需要4卡。

### B. 历史置信度驱动的用户先验（本机优先）

沿用已有用户边缘 KL，但不要永久信任单次历史服务率或固定强度。新候选为 `β(s)=β0·confidence(s)`：当历史服务估计不可靠、近期反馈突变时降低先验；队列风险上升、反馈稳定时提高先验。参考环境作者的动作分解与 2025 cascade 文献，关键是只引导用户服务，不锁死 beam/relay/tracking。

最小对照：continued PPO、固定用户先验、置信度自适应用户先验；同起点同30k；记录β、用户边缘KL、beam/relay多样性与平均/最差队列。所需状态优先限于已有35维历史量，若新增服务计数，必须也提供给所有公平对照或单独标“信息增强”。

难度：低到中。风险：用户先验本就只有 topology12 局部改善；自适应 β 可能只是又一组启发式权重，必须通过“恒定β同均值”消融排除只是弱先验效果。当前未验证，不将它写成已完成创新。

### C. 条件动作策略：用户→relay/beam/tracking（结构候选）

将 `π(a|s)` 改为 `π(u|s)·π(link|s,u)`，明确第二阶段读第一阶段用户；再比较顺序反转或平行独立双头。2025 R1C2/C1R2 已说明级联顺序可能重要，2022 作者 MAB 已拆过子动作。因此我们可以研究“mask下的调度分解与用户级先验为何不同”，不能声称首次层次RL。

本机最小探针：先离线拟合现有策略的联合概率，确认结构表达/合法mask不损失；再短训 factorized actor vs 参数量接近的 joint actor。更换网络后不能直接加载原 actor 并称等 warm-start，需说明初始化/蒸馏。

难度：中。风险：当前只有5用户/180动作，巨大动作空间优势可能不明显；tracking和D2D约束使条件mask实现容易错。展现方式清楚：先点用户、后点配置、展示每步概率与最终链路。

### D. 因果历史记忆 + 观测退化（值得探索）

问题不是加 LSTM 就算创新，而是测试现有全用户遮挡概率观测假设是否过强。保留队列与实际服务反馈，去掉/延迟/加噪遮挡开始信息，用短历史 GRU/TCN 推断链路可用性，再给 actor。Yao 等证明定位不确定性被研究过，Pinto/RMA 提供部分观测与适应机制依据。

先做不重训的观测退化敏感性回放；如果 baseline 对退化不敏感，不优先训练记忆模块。若敏感，比较 feedforward、堆帧、GRU，历史长度3/8任选固定一个，不做网格。记录突发遮挡后恢复时隙、最差队列和用户饿死时长。

难度：中。风险：原仿真器遮挡有具体生成假设，记忆模型可能只记住生成分布；必须使用真实因果历史，不能把未来CSI塞入历史。需要新增固定观测适配器，原环境保持不动。

### E. 强教师→小部署策略 / privileged critic（中长期）

用户提出的“大模型教师+小模型部署”在RL也可保留，建议先拆成两种：

- 同观测较大教师：多个拓扑/负载训练的大MLP、Transformer历史教师，推理输入仍是合法历史；蒸馏其soft action distribution给约100k参数学生。先问老师是否比小 PPO 强；无教师优势就不值得蒸馏。
- 训练期特权：critic或教师读当前模拟器信道/阻挡，学生只读35维或因果历史。先做 asymmetric critic，小actor不变，改动比完整教师小。若做教师蒸馏，学生仅看部分状态，教师动作可能不可辨识，必须量化残差而不承诺复制oracle。

依据是 Policy Distillation 与 Asymmetric Actor Critic，机制已有。可形成我们的贡献的是“训练期信息如何通过用户边缘/条件动作软知识转给仅历史反馈的小策略，以及其跨拓扑/遮挡恢复收益”；需要实测，未证明首次。

本机可做：用现有强策略/TS当老师，收集短轨迹，拟合轻量行为克隆并检查action KL与闭环性能。行为克隆失败可能由状态分布漂移，后续加学生轨迹再标注（DAgger类机制需再补正式来源）而非只提升模型大小。

4卡可做：多环境域随机化强教师、多任务教师聚合、历史Transformer；测教师/学生同观测闭环吞吐/队列与CPU推理时延。当前部署学生已很小，单纯进一步压缩不够吸引，应优先泛化/因果观测收益。

### F. 多智能体/真实trace迁移（暂低优先级）

Guo等MADRL面向access/backhaul多个职责。当前环境只有一个集中控制器，硬改多个agent会引入credit assignment与通信协议，增加难度；只有后续扩为多AP/多个独立控制节点才合理。2025 VR论文有真实mmWave traces，但其任务是视频处理，不能把换数据直接当MAC仿真复现。暂保留文献线索，不挤占本轮单机机制验证。

### G. 历史世界模型：短期预测→辅助调度→大教师（4卡路线）

2025 World Model-Based Learning已经在mmWave V2X研究了学动态模型与imagined scheduling，因此“加world model”也不是首次。我们可以先在当前环境验证更小的机制：从最近历史观测和已执行action预测下一时隙服务量/各UE队列，用预测误差或不确定度决定是否信任用户先验，或作为critic辅助任务。此时实际actor仍只看部署可得历史。

已读其§III：用CAoI、ray-tracing data、vehicle locations作observation；RSSM含256维确定性和256维随机latent，sequence length64、imagination horizon30（Table I），仿真为26GHz、8 vehicles，不是现有60GHz MANET同任务。它对即时ray-tracing/几何的观测假设也不能无条件移到可部署系统。我们的“仅服务反馈历史”版本会更难，但信息边界更明确。

本机先做数据探针：收集现有PPO轨迹；训练小MLP/GRU预测下一slot服务/队列；与“保持上一服务率+已知queue更新”基线比较。必须按整条轨迹划分，避免邻近时隙泄漏。若预测不超过简单持久性，不开展imagined PPO。

后续4卡可做：更长历史Transformer世界模型/多域teacher，在imagined rollout里得到强策略后蒸馏给小actor。风险比reward probe高：模型误差长时滚动放大、策略利用错误模型、原环境5用户规模太小不需要大型Transformer。不能把4卡都占满当研究目标；优先验证“缺观测时的计划能力/跨域恢复”是否有益。

## 探索顺序与判停

| 优先级 | 候选 | 本轮资源 | 什么结果值得继续 | 什么结果应暂停 |
|---|---|---|---|---|
| 1 | drift 与 tail queue奖励 | CPU三支30k、单seed | 高负载pending/最差队列明显改善且吞吐损失可接受 | 只改善delivered delay但pending更高；或奖励分量根本未触发 |
| 1 | confidence-adaptive用户先验 | CPU短续训 | 稳定保留link自由度，缓解topology13退化 | 效果等同减小固定β；仍锁死某配置 |
| 2 | 历史/观测退化 | 先回放后训练 | 无特权观测时明确失效，记忆能恢复 | 退化不影响结果，新增GRU没有必要 |
| 2 | conditional actor | CPU可运行、实现中等 | 更快样本效率/清楚可解释变化 | 参数/初始化差异解释全部收益 |
| 3 | 强教师或privileged critic | 先小probe，后4卡域随机化 | 老师真强，学生闭环有益且仅合法历史推理 | 教师不强；或只在特权信息泄漏时好 |
| 3 | 历史世界模型 | 本机短期预测，后4卡teacher | 超过持久性预测基线，闭环或观测缺失恢复有益 | 预测只拟合queue算术；长rollout误差爆炸 |
| 4 | MADRL/新真实trace环境 | 新环境工作多 | 有明确多个自治节点需求 | 仅把一个控制器拆开，任务无法解释 |

当前阶段每项最多一个固定方案短测、一个训练seed，不扩成论文级多种子矩阵。失败也形成可展示的证据。报告先保留候选并行状态，不在第一次局部好结果后立即定最终课题。

## 与当前结果的关系

已完成的用户边缘先验在topology12有局部正向信号，topology13有trade-off；这属于现有探索证据。A的固定reward短测也已完成，但没有整体支持；B—G仍是候选或具体后续实验，不把文献里的收益移植成我们的数字。没有把大模型教师路线取消；只是先检查教师能力、现有信息瓶颈与小机制能否成立，再决定是否利用4卡扩大。

## 扫描局限

本次来源集中在arXiv官方页面，部分正式出版状态仅是作者在arXiv自报；没有用付费IEEE全文逐项核对。完成三篇关键方法的HTML正文检查（Lyapunov、级联、world model），其余按摘要层次支持动机，不推断未读的超参数或具体实现。检索接口遇限流，2024—2026仅有有界扫描；后续最终选定机制时，需要补一轮相邻论文、已有代码及新颖性比对。该局限不妨碍做可逆的课程实验，但妨碍“首次提出”的声明。
