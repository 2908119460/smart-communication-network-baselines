# 单种子30k：队列奖励机制探索

## Material Passport

Origin Skill: academic-research-suite / experiment-agent；模式：run；日期：2026-10-12。
状态：三个分支均完成；只表示本机训练/评价完成，不表示算法新颖性或泛化已证明。

三支从同一360k PPO模型与Adam恢复；collector和RNG重启，训练seed=0，每支额外30,000环境交互、472次Adam优化。每个场景仅一个environment seed=41000。没有增加训练种子，没有改正式locked800行。

原reward为每时隙已交付包数除以2Gbps对应包数（原作者变量名npkt_10gbps，但计算实际为2e9）。Qref=2,000包/UE，η=0.02固定，无参数网格。

2026-10-12 文档校正：执行时脚本误把归一分母标签写为10Gbps；实际一直调用原环境的2Gbps reward，训练数值与权重没有变化。protocol/summary保留script_sha256为实际执行时哈希，documentation_correction记录当前说明版本。checkpoint内嵌protocol仍包含执行时旧标签，不改写权重。

- continued PPO：保留原reward。
- queue drift：r'=r+η·clip(L(q_t)-L(q_{t+1}),-1,1)，L(q)=0.5·mean((q/Qref)^2)。
- tail penalty：r'=r-η·mean(clip(q_{t+1}/Qref-1,0,5))。

queue drift是有界队列变化奖励；tail penalty是超过固定阈值的积压代价。它们不是CVaR/CPO，不提供队列稳定保证。新reward改变critic的价值语义，继承critic最初尚未适配；30k结果需要结合value loss解释。

| 场景 | 方法 | 吞吐 Gbps | 平均队列 | 最差用户均队列 | pending % | queue>2000 % |
|---|---|---:|---:|---:|---:|---:|
| explore11_load09_block10 | continued_ppo | 0.899443 | 518.6 | 598.3 | 0.350 | 1.27 |
| explore12_load11_block12 | continued_ppo | 0.982183 | 13275.7 | 35343.3 | 10.512 | 78.40 |
| explore13_load10_block10 | continued_ppo | 0.982224 | 2373.3 | 6318.4 | 2.943 | 35.53 |
| explore11_load09_block10 | queue_drift | 0.895964 | 616.0 | 784.8 | 0.541 | 1.73 |
| explore12_load11_block12 | queue_drift | 0.929096 | 19519.9 | 35731.8 | 15.171 | 84.07 |
| explore13_load10_block10 | queue_drift | 0.987831 | 3734.2 | 9092.2 | 3.364 | 49.24 |
| explore11_load09_block10 | tail_queue_penalty | 0.900480 | 584.8 | 818.5 | 0.255 | 2.49 |
| explore12_load11_block12 | tail_queue_penalty | 0.940338 | 18767.5 | 52908.8 | 14.608 | 76.95 |
| explore13_load10_block10 | tail_queue_penalty | 0.983838 | 2510.8 | 4921.7 | 2.859 | 46.48 |

与同预算continued PPO相比：

| 场景 | 候选 | 吞吐变化 % | 队列变化 % | 最差用户队列变化 % | pending变化 百分点 |
|---|---|---:|---:|---:|---:|
| explore11_load09_block10 | queue_drift | -0.39 | +18.79 | +31.17 | +0.191 |
| explore11_load09_block10 | tail_queue_penalty | +0.12 | +12.77 | +36.80 | -0.094 |
| explore12_load11_block12 | queue_drift | -5.41 | +47.03 | +1.10 | +4.659 |
| explore12_load11_block12 | tail_queue_penalty | -4.26 | +41.37 | +49.70 | +4.096 |
| explore13_load10_block10 | queue_drift | +0.57 | +57.34 | +43.90 | +0.421 |
| explore13_load10_block10 | tail_queue_penalty | +0.16 | +5.79 | -22.11 | -0.084 |

限制：一个训练seed与每场景一次轨迹；topology12/13此前已探索，这些结果是机制筛选而非新锁定test。不挑最佳checkpoint；评估只使用最终390k策略。已交付时延只对完成包计数，pending不是丢包率。

固定强度惩罚只是用于探测方向，不自动构成新的学术算法。此轮改变reward与warm-start价值适配同时发生，负结果不能证明所有Lyapunov/CMDP方法不可行。

缓存：每支3个NPZ保存actions/queues/arrivals/departures/delay/pending，training_rewards.npz保存30k步奖励分量；没有渲染视频。

总耗时：525.5秒。source环境、原collector/evaluator脚本与locked800文件hash均保持一致。

## 本轮结果解释

这两种固定奖励形式没有得到整体支持。queue drift在三个场景的平均队列都增加，topology12吞吐下降5.41%、平均队列增加47.03%。tail penalty在topology12吞吐下降4.26%、平均队列增加41.37%；topology13则出现局部trade-off：队列P95从9,698降至7,187包、最差用户均队列下降22.11%，但平均队列仍上升5.79%。因此不把某个尾部指标改善写成“已验证创新有效”。

奖励信号审查直接读取保存的30,000步training_rewards.npz：原reward均值约0.458，drift附加项绝对均值0.0003857，仅为原reward均值的0.0843%；tail附加项绝对均值0.0001090，仅为0.0238%，并且只有10.11%训练时隙真正触发阈值。这说明本轮在原有较轻负载训练分布中探测信号很弱，并没有充分训练“高负载避积压”。这个数值审查是可能解释，不证明失败的唯一原因。

三支value loss（首rollout→末rollout）依次为0.3499→0.1237、0.3500→0.1532、0.3499→0.1252，没有明显数值爆炸。继承critic与新reward语义不完全一致仍是设计边界，不能仅凭value loss认定它导致了负结果。

下一步优先探索条件动作与因果历史/观测退化，或先证明强教师比当前PPO更强；不继续堆这两种固定penalty并宣称成功。若以后重做约束调度，应先把训练分布中的队列风险真正激活，再考虑自适应dual，而不是盲目扩大网络。

交互图：queue_trajectory_topology12.html。两行分别显示五用户平均队列与每时隙最大用户队列；三方法均为缓存的真实轨迹，20秒完整曲线，前5秒灰区不纳入指标统计。图中queue P95指“时隙×用户队列”分位，pending指未交付比例。它展示为什么本轮固定奖励没有改善积压，而不是展示成功的创新模型。
