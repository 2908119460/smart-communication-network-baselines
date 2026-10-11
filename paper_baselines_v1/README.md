# 三方向课程论文基线补强

> 当前范围更新（2026-10-11）：用户要求优先验证与 demo，不再扩展多种子。YOLO 仅保留 YOLO11n seed0，80 轮上限与自然早停；PA 原多种子等待队列已取消，改为单seed3源模型/8适应验证；RL 原800行结果保留，新增先验机制仅做单seed探索。下方原计划表用于追踪历史，不能把计划矩阵当作已完成结果。

**最新选题结论：[单种子验证与选题结论](../../notes/单种子验证与选题结论.md)。** 方法来源：[真实方法清单](../../notes/baseline_methods_verified_v1.md)。结果与demo：[YOLO](../vision_single_seed0_eval_v1/README.md)、[时序诊断](../vision_single_temporal_exploration_v1/README.md)、[PA](../pa_single_demo_v1/README.md)、[RL用户先验24行对照](../rl_user_prior_demo_v1/REPORT.md)。旧RL中断状态保留，完成状态以continuation_status.json为准。

更新：2026-10-11。此目录是统一入口，训练、数据和检查记录保留在各方向独立目录。本轮单seed训练与测试已经完成；交付核验见delivery_audit.json。

## 交付范围

用户要求将 PA/MoE、60 GHz 调度、YOLO 无人机检测三个课题的基线补到可以用于课程论文的程度，并生成可视化。基线方法与候选创新分开记录；完成训练不等同于复现原论文全部设置，不等同于证明创新或达到发表要求。

本阶段完成条件：可查的数据/方法来源、固定训练和选模方式、必要对照、保留原始指标/权重/日志、汇总图与可播放演示。单seed验证支持当前demo和初步选择，不要求额外训练重复。失败实验保留。

## 本轮任务

| 方向 | 正式协议与结果目录 | 本轮补强内容 | 当前状态 |
|---|---|---|---|
| PA/MoE | [传统参考](../pa_paper_baseline_v1/)、[单seed](../pa_single_demo_v1/) | 单seed约2700参数GRU/均匀专家/包络MoE各150轮；A→B 1024/4096总标签full/partial适应，75%拟合/25%选模；MP/因果GMP传统参考 | 传统6条与神经14条完成；2图与15秒残差回放；gate-only收益未获支持。原45+96矩阵取消 |
| 60 GHz | [rl_paper_baseline_locked_v1](../rl_paper_baseline_locked_v1/) | 360k标准PPO 3 seed、固定码本与验证选择码本、反馈型TS基线；4场景×20新环境seed；特权信息greedy单独记录 | 三个360k训练 seed 已完成；800/800 锁定测试行、45项 post integrity 审计、报告、4图和4个回放视频已完成；seed2 尾部旧诊断超时单独保留 |
| YOLO | [数据与协议](../../data/vision/paper_v1/)、[单模型结果](../vision_single_seed0_eval_v1/README.md) | 265新视频/6360抽帧，train179/val33/cal20/test33；YOLO11n seed0，80轮上限、58轮自然早停 | test mAP50=.8310、mAP50–95=.3914；10357连续帧召回99.58%、FPR2.76%；4检测视频+4时序视频；其余5模型未运行 |

## 已完成的补充诊断

- [同配方frame200 PA/MoE](../dpd_candidate_frame200_v1/summary.json)：8次源拟合、32次适应与8条冻结结果；full target validation仍使用19,662标签，因此不属于新严格预算实验。两seed描述统计不是置信区间。
- [20段连续视频与时序规则](../vision_video_eval_baseline_v1/REPORT.md)：6,281连续帧；复用旧YOLO11n，时序后处理不是新模型训练或新独立测试。
- [特权信息greedy参考](../rl_greedy_oracle_v1/result.json)：读取真实瞬时信道，不能与历史观测PPO公平排名，也不能保证是理论上界。

## 文件组织和核验

- 原始数据保留在 `experiments/data` 或上游自带datasets，来源/hash在各协议内。
- 上游仓库只读；自有兼容性适配和训练入口在 `experiments/scripts`。
- 新结果使用独立run-id；旧初试、失败或作废结果不覆盖。
- 静态科学图输出PNG/SVG/PDF；视频保留来源、模型、阈值、采样和编码记录。
- 最终报告会区分文献中的方法、当前代码配方、实际运行结果和未验证的候选贡献。

旧演示入口：[baseline_videos_v1](../baseline_videos_v1/README.md)。本轮完成后另建统一入口，并保留旧入口及其证据边界。

## 长实验与资源记录

本机为单RTX4060 Laptop 8GB、16GB RAM。视觉单模型实际训练202.4分钟，第58轮自然早停；用户保留80轮上限，没有人为改成40轮。PA单seed和RL候选串行执行，已释放训练资源。

Windows提交内存曾接近耗尽：视觉attempt0/1尚未发生优化就失败；第三PPO在237,760步失败，不能计作360k完成。第三seed采用独立目录精确重放，在旧30k/60k/120k模型与Adam状态逐张量核验后继续；不把仅存权重的checkpoint假称完整恢复。恢复脚本实际仍受底层2400秒安全上限约束，4200秒只是外层配置，不能当作实际可用时限。

PA全矩阵SVD已操作员中断（exit1），已完成两条APA传统参考保留。新的 [工程修订](../pa_paper_baseline_v1/polynomial_streaming_amendment.json) 采用4096行分块QR后SVD，没有下采样、没有正规方程、没有修改结构/阈值或验证选模规则。随机序列的分块basis逐值相同，真实APA大GMP验证NMSE与原全矩阵相差4.263×10⁻¹⁴dB。原protocol和脚本未修改，续算结果单独保存。

Windows资源以 `scripts/machine_resources.py` 的GetPerformanceInfo为准；自动页面文件变化后psutil.swap_memory可能显示过期limit。没有手动改动页面文件或结束用户应用。

旧持久队列：[PA queue](pa_queue/status.json)，已按用户最新范围取消。当前单seed任务在本回合显式串行完成，不会由旧队列继续启动原多种子矩阵。

传统参考已完成：APA A的MP/GMP为−36.735/−37.826dB，APA B为−32.980/−37.644dB，DPA160为−38.303/−39.279dB。六个保存系数重建全部完整test输出，逐值与保存预测一致；验证最优结构/阈值、数据hash和startup敏感性均通过核验。见 [审计](../pa_paper_baseline_v1/polynomial_delivery_audit.json) 与 [传统参考图](../pa_paper_baseline_v1/traditional_references.png)。这些是本轮传统强参考，不代表待跑神经矩阵已完成。

RL 对照中的 `fixed_beam2` 与 `validation_selected_beam` 是队列感知的 MaxWeight-like 动态启发式，不是 round-robin/TDMA。严格 direct-only round-robin 已单独完成动作定义可行性审查，但未加入锁定 800 行主结果；若补跑，应使用独立 supplemental 目录，不改写本轮协议。
