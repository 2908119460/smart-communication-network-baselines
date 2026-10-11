## Material Passport

模式：单训练seed、固定beta的warm-start探索验证。没有改动正式locked800行。

三支均从同一360k模型和Adam状态继续30k，重启collector/RNG，额外472个优化step；并非无缝继承旧环境轨迹。beta=0.02恒定。用户先验是旧联合先验的5维用户边缘，不约束同用户下beam/relay/tracking条件概率。

| 场景 | 方法 | 吞吐 Gbps | 平均队列 包/UE | pending % | beam2 % | relay % |
|---|---|---:|---:|---:|---:|---:|
| validation_demo12_load11_block12 | anchor_360k | 1.005443 | 14604.1 | 9.876 | 29.93 | 38.20 |
| validation_demo12_load11_block12 | continued_ppo | 1.033871 | 8357.8 | 6.578 | 17.67 | 38.56 |
| validation_demo12_load11_block12 | joint_prior | 1.001050 | 14318.7 | 9.793 | 35.31 | 36.67 |
| validation_demo12_load11_block12 | user_marginal_prior | 1.053455 | 8133.6 | 5.533 | 24.44 | 39.29 |
| validation_demo13_load10_block10 | anchor_360k | 1.009490 | 3981.1 | 0.727 | 21.04 | 44.58 |
| validation_demo13_load10_block10 | continued_ppo | 1.012990 | 3232.7 | 0.554 | 16.67 | 43.51 |
| validation_demo13_load10_block10 | joint_prior | 1.012881 | 4077.4 | 0.507 | 32.67 | 36.73 |
| validation_demo13_load10_block10 | user_marginal_prior | 1.008969 | 3070.7 | 0.707 | 20.24 | 40.91 |

数值为预定三个环境draw的均值；一个训练seed，不是三个独立训练模型。不以该验证集挑checkpoint或beta，不作显著性/普遍鲁棒性/新颖性声明。

缓存：24个NPZ保留actions/queues/arrivals/departures/delivered/pending数组，可用于录制动作回放。没有新视频；真实geometry须用未修改源环境按记录动作重放，不能直接把旧视频替换成新方法。

中断恢复：anchor与continued PPO的12行结果/权重hash已核对并复用；旧joint的10240步未进入结果，新joint_prior_attempt2从同一360k模型+Adam重跑30k。原中断status/protocol及hash保持原样，完成状态见continuation_status.json；额外失败尝试的计算成本另计。

用户级先验相对同预算continued PPO：topology12吞吐+1.89%、平均队列−2.68%、期末pending占比从6.578%降至5.533%（−1.045个百分点），三个配对环境draw的吞吐和队列方向均一致。topology13吞吐−0.40%、平均队列−5.01%，但pending从0.554%升至0.707%（+0.154个百分点）；平均队列下降主要来自seed40001，另两个draw的队列反而略升，不能称为该场景稳定改善。

用户级先验相对joint prior：topology12吞吐+5.23%、队列−43.20%，topology13吞吐−0.39%、队列−24.69%。joint的beam2占比分别35.31%/32.67%，用户先验降至24.44%/20.24%，与不直接惩罚同用户条件配置的机制相容；不能从这些比例单独推出因果。相对anchor的改善不能全部归因于先验，因为普通PPO继续训练30k也改善了。

结论：支持继续研究“仅在用户边缘施加队列先验、保留条件动作探索”的候选机制，尚不足以支持普遍优于PPO或已证明创新。训练仅一个seed，环境draw只有三个，场景在此前探索中已观察过；这是单seed固定beta的验证，不是新的正式test。

直观对照图：figures/user_prior_mechanism_comparison_v2.png（同名SVG/PDF可用于论文与PPT，PNG已目视检查）。派生均值变化与配对逐draw差值见exploratory_analysis.json。V1图保留，V2修正标题拥挤并使用从零开始的线性条形轴，增加beam2配置份额。
