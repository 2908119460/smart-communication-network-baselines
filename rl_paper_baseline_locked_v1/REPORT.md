## Material Passport

产物：60 GHz 调度完整基线评估；来源为作者开源仿真器与改写标准 PPO。

记录范围：三个独立训练种子、两个既有场景参数、两个预先锁定负载/遮挡偏移，各20个新环境种子，共800行。有限20秒仿真，首5秒warmup，测量15秒。

## 结果

PPO 并未在所有场景稳定胜出。请按下表与每训练种子结果展示，不能只选择seed0。特权贪心的额外瞬时信息不参与公平算法排名。

| 场景 | 方法 | 吞吐 Gbps | 平均队列 包/UE | 末尾未交付 / 全部到达 % |
|---|---|---:|---:|---:|
| test12_load11_block12 | PPO 30k（三训练种子，预算诊断） | 0.895519 | 29677.7 | 19.423 |
| test12_load11_block12 | PPO 360k（三训练种子） | 0.914682 | 28195.2 | 18.233 |
| test12_load11_block12 | 固定码本2启发式 | 0.733298 | 52478.2 | 34.508 |
| test12_load11_block12 | 验证选码本3启发式 | 0.757597 | 48767.3 | 32.012 |
| test12_load11_block12 | 反馈型作者scheme4 TS（在线） | 0.810273 | 40944.1 | 26.955 |
| test12_load11_block12 | 特权信息逐槽贪心参考 | 1.078196 | 5578.3 | 3.007 |
| test13_load10_block10 | PPO 30k（三训练种子，预算诊断） | 0.918284 | 12831.0 | 8.788 |
| test13_load10_block10 | PPO 360k（三训练种子） | 0.953174 | 8468.9 | 5.350 |
| test13_load10_block10 | 固定码本2启发式 | 0.977174 | 5703.3 | 3.104 |
| test13_load10_block10 | 验证选码本3启发式 | 0.999974 | 1264.0 | 0.500 |
| test13_load10_block10 | 反馈型作者scheme4 TS（在线） | 1.003007 | 1353.4 | 0.441 |
| test13_load10_block10 | 特权信息逐槽贪心参考 | 1.000374 | 359.3 | 0.139 |
| shift12_load115_block13 | PPO 30k（三训练种子，预算诊断） | 0.919429 | 34105.9 | 20.932 |
| shift12_load115_block13 | PPO 360k（三训练种子） | 0.915950 | 35437.0 | 21.639 |
| shift12_load115_block13 | 固定码本2启发式 | 0.745576 | 57141.2 | 36.289 |
| shift12_load115_block13 | 验证选码本3启发式 | 0.772304 | 53977.8 | 33.974 |
| shift12_load115_block13 | 反馈型作者scheme4 TS（在线） | 0.803228 | 47847.0 | 30.471 |
| shift12_load115_block13 | 特权信息逐槽贪心参考 | 1.118886 | 7656.1 | 3.723 |
| shift13_load105_block11 | PPO 30k（三训练种子，预算诊断） | 0.956764 | 14428.5 | 9.533 |
| shift13_load105_block11 | PPO 360k（三训练种子） | 0.993235 | 9660.9 | 6.119 |
| shift13_load105_block11 | 固定码本2启发式 | 1.023523 | 5311.3 | 3.153 |
| shift13_load105_block11 | 验证选码本3启发式 | 1.047251 | 1133.4 | 0.579 |
| shift13_load105_block11 | 反馈型作者scheme4 TS（在线） | 1.050603 | 1265.0 | 0.529 |
| shift13_load105_block11 | 特权信息逐槽贪心参考 | 1.049183 | 373.9 | 0.192 |

PPO 数字先对每个训练种子的20条轨迹求均值，再对三个训练种子求均值。图上的三个黑点与范围展示训练随机性；20条环境轨迹不是20个独立训练模型。

## 配方、信息与预算边界

标准 PPO 使用98,613参数的actor/critic、clipped objective、GAE、固定360k训练预算和final checkpoint；它与原作者DRL的batch、学习率、gamma、lambda、entropy等不同，不属于精确作者配方复现。队列在训练episode重置，无法从这里证明无限时域稳定性。

固定启发式和验证选码本复用历史队列/速率，direct-only。六档码本只在独立validation种子30000–30009上选择，beam3胜出；新测试seed20000–20019不参与调参。

反馈TS复用作者scheme4的MaxWeight用户选择、码本TS和tracking启发式；策略随机流与环境隔离，后验更新系数只由执行链路Reff/MCS反馈推导。其服务计数、est_depart、后验历史多于PPO当前35维输入；零离线训练但每测试轨迹在线更新2000槽，因此不是同一离线预算/相同stateless特征的比较。原scheme6缺checkpoint且有relay索引、选/更新用户索引与隐藏outage问题，详见source_mab_feasibility.md。

PPO自带遮挡估计来自作者仿真状态，其真实部署可观测性没有验证。特权greedy读取真pathloss/当前信道，direct-only且myopic，评分省略next-step outage，没有性能上界保证。

## 失败与恢复

seed2_v1因系统OOM在约238k退出，原status与日志保留。seed2_v2同seed同配方重构轨迹；在30k、60k、120k分别64个模型/Adam张量逐位一致。最终360k checkpoint、355 rollout更新、5636 Adam steps与18条计划验证均完成。

v2尾部旧诊断评价达到base runner的实际2400秒wall cap而timeout；training_completion.json与optimizer_step_audit.json另行核验训练已完成，原timeout状态未改写。新锁定80条测试单独完成。原配置字段4200没有覆盖base内部min(...,2400)，此限制已记录。失败尝试和重放前缀消耗额外计算，不宣称总wall资源相同。

## 检查与解释限制

全部800行覆盖与整数包数/float64吞吐重算通过，80个场景×seed外生轨迹hash跨所有方法完全配对。报告11项统计误用检查，paired bootstrap区间只条件于单份冻结policy，不作为总体训练算法显著性证据。

原两场景家族已经观察过；两参数偏移在评价前锁定但拓扑家族仍相同。测量窗口可能交付warmup到达的积压，所以短窗口吞吐超过名义负载不等于容量提高。未交付比例不是loss概率，delivered-only delay有删失，应同时展示队列/未交付。

统计图：locked_throughput、locked_mean_queue、locked_pending、validation_three_seeds，均PNG/SVG/PDF。新动画使用预定envseed20000的单轨迹，非multi-seed均值。
