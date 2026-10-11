# 60 GHz single-seed supplemental comparison

本补充只在两个已观察场景各取一个环境seed=40000，复用用户级先验run中的四条策略轨迹，并在同一外生trace上新增固定码本2、验证选择码本3、严格固定波束2的RR/TDMA以及在线反馈MAB。没有修改locked 800行。

RR/TDMA定义：每个slot按 `u = slot mod 5` 轮流给一个UE，固定beam=2，direct-only，relay=UE，tracking bit=0。它是可解释的零训练下界式参考，不是优化后的TDMA。

| 场景 | 方法 | 吞吐 Gbps | 平均队列 包/UE | pending % |
|---|---|---:|---:|---:|
| validation_demo12_load11_block12 | anchor_360k | 1.056131 | 5903.0 | 4.705 |
| validation_demo12_load11_block12 | continued_ppo | 1.066789 | 3890.8 | 3.508 |
| validation_demo12_load11_block12 | joint_prior | 1.061964 | 6619.4 | 4.026 |
| validation_demo12_load11_block12 | user_marginal_prior | 1.087737 | 3495.5 | 1.747 |
| validation_demo12_load11_block12 | fixed_beam2 | 0.706869 | 50559.1 | 35.057 |
| validation_demo12_load11_block12 | validation_beam3 | 0.430106 | 83915.4 | 58.675 |
| validation_demo12_load11_block12 | rr_tdma_beam2 | 0.699598 | 51838.7 | 35.645 |
| validation_demo12_load11_block12 | feedback_mab | 0.751045 | 41552.1 | 29.917 |
| validation_demo13_load10_block10 | anchor_360k | 1.000671 | 1566.6 | 1.008 |
| validation_demo13_load10_block10 | continued_ppo | 1.005759 | 1048.5 | 0.626 |
| validation_demo13_load10_block10 | joint_prior | 1.005840 | 1123.8 | 0.598 |
| validation_demo13_load10_block10 | user_marginal_prior | 1.000974 | 1197.5 | 0.490 |
| validation_demo13_load10_block10 | fixed_beam2 | 1.005335 | 787.8 | 0.210 |
| validation_demo13_load10_block10 | validation_beam3 | 1.000881 | 496.2 | 0.255 |
| validation_demo13_load10_block10 | rr_tdma_beam2 | 0.943533 | 9078.1 | 6.531 |
| validation_demo13_load10_block10 | feedback_mab | 1.002588 | 611.8 | 0.154 |

相对同一draw的continued PPO，用户级先验在Topology 12的吞吐提高1.96%、平均队列降低10.2%、pending降低1.76个百分点；Topology 13的吞吐降低0.48%、平均队列增加14.2%，但pending降低0.136个百分点。这个单draw结果支持“用户边缘约束保留条件动作自由度”作为可展示的候选机制，不支持稳定优于PPO的结论。

RR/TDMA在Topology 12达到0.700 Gbps、平均队列51839，明显低于PPO和用户级先验；Topology 13达到0.944 Gbps、平均队列9078，也低于同场景的学习方法和在线反馈MAB。它适合在PPT中解释调度自由度与拥塞的关系，不适合作为优化算法排名的强基线。

解释边界：学习方法只有一个训练seed；每个场景只有一个环境draw。feedback MAB包含2000槽在线后验更新，与PPO离线预算和35维无记忆状态不等价。短窗口结果不能证明稳定性或跨拓扑泛化。

图：figures/short_supplement_comparison.png（同名SVG/PDF）。
