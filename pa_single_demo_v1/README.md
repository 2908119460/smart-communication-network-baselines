# PA 单种子验证与 Demo

本次训练、测试、图表、视频已完成，GPU 训练进程退出码为 0。只运行了
3 次源模型训练和 8 次适应训练，没有启动旧矩阵或增加种子。目录独立于
冻结的 `../pa_paper_baseline_v1`，其原脚本和传统参考结果保持不变。

## 实验方法

源数据为 OpenDPD 的 `APA_200MHz`（A），目标为 `APA_200MHz_b`（B），
采用官方时间划分。三支源模型使用相近的约 2,700 参数预算、seed 0、
frame200/stride1、batch64、AdamW、lr0.005、ReduceLROnPlateau，以及
完整 150 epochs；按验证集最低全局 NMSE 选择 checkpoint。

GRU 是 OpenDPD 原生 GRU-H28（2,746 参数）。均匀专家是三支原生
GRU-H15 的平均（2,661 参数）。Envelope MoE 是同样的三支专家加
15 点因果包络历史线性 softmax 门控（2,709 参数）。后两者是本项目的
模型族对照，不是已验证的创新算法。

只进行 A→B 适应。预算 1,024 和 4,096 都指总目标标签数：分别
768/256 和 3,072/1,024 点用于拟合/选择，来自 B train 前缀的按时间
75%/25% 切分。未用 B 官方 val 选择适应 checkpoint。适应 lr0.0005，
完整 20 epochs。GRU partial 更新 58 个输出头参数；MoE partial
更新 48 个门控参数，两者适应参数数目不同。

本轮最终测试值在 `choices_frozen_before_test.json` 写出后计算，
每个模型从 test 起点重置状态，以一次连续推理评估完整 19,662 点。
准备阶段为数据清单计算过 test 原文件行数和 SHA-256；冻结标记中的
`test_files_opened:false` 指本轮训练期间尚未加载 test 数组用于模型评价，
不应理解为准备阶段从未打开过 test 文件。此前探索实验也已观察过 APA
test，本次不是一套从未接触的新数据。主指标为 float64 全局 I/Q NMSE，
裁掉前 150 点的敏感性指标只作次要参考。

## 14 行真实全测试结果

NMSE 越低越好；A 表中的参数是源训练参数，B 表中的参数是目标适应
更新数。零适应行没有更新任何参数，其 raw result 中 epochs/seconds 等
继承源 checkpoint 的训练记录，已通过 `training_metadata_scope` 明确说明。

| 测试域 | 方法 | 总目标标签 | 目标适应参数 | 全测试 NMSE/dB |
|---|---|---:|---:|---:|
| A | GRU-H28 源模型 | — | — | -38.9030 |
| A | Uniform experts 源模型 | — | — | -38.9841 |
| A | Envelope MoE 源模型 | — | — | -38.9875 |
| B | GRU-H28 full | 1,024 | 2,746 | -35.9229 |
| B | GRU-H28 output head | 1,024 | 58 | -30.2494 |
| B | GRU-H28 full | 4,096 | 2,746 | -38.2083 |
| B | GRU-H28 output head | 4,096 | 58 | -33.5558 |
| B | Envelope MoE full | 1,024 | 2,709 | -36.0113 |
| B | Envelope MoE gate | 1,024 | 48 | -24.1836 |
| B | Envelope MoE full | 4,096 | 2,709 | -38.0225 |
| B | Envelope MoE gate | 4,096 | 48 | -24.4448 |
| B | GRU-H28 零适应 | 0 | 0 | -23.8040 |
| B | Uniform experts 零适应 | 0 | 0 | -23.8138 |
| B | Envelope MoE 零适应 | 0 | 0 | -23.8076 |

3 次源训练耗时分别为 88.83、144.06、147.34 秒；8 次适应训练共
4.89 秒，fit 计时总计约 385.12 秒。所有源模型跑满 150 epochs，
所有适应跑满 20 epochs。保存的 14 组有限值预测各覆盖 19,662 点；
从保存的预测重新计算全局 NMSE，与 `neural_results.json` 的最大绝对
差为 0.0 dB。

既有验证选模后的传统参考为 A MP -36.7351、Causal GMP -37.8261，
B MP -32.9801、Causal GMP -37.6437 dB。B 的传统参考使用完整目标
训练/验证集，不是相同 1,024/4,096 标签预算的对照，图中用空心方框
及 `full target fit` 明示。

## 对创新点的判断

当前没有证据支持“包络历史门控是主要贡献”：A 源域 MoE 比均匀专家
仅好 0.0034 dB；B 上 gate-only 只改善约 0.38/0.64 dB，而 GRU
输出头改善约 6.45/9.75 dB。完整微调有效，但 MoE 对 GRU 的方向
随预算改变：1,024 标签略好约 0.088 dB，4,096 标签弱约 0.186 dB。
单种子无法解释这些小差异的稳定性。

有可展示的现象是“跨采集偏差 + 小样本适应”：完整微调能将 B 的
约 -23.8 dB 恢复至约 -38 dB。但完整微调和输出头校准本身都不能
作为新的算法贡献。若选择该课题，PPT 应如实展示门控假设未被支持，
而不是宣称 MoE 获得稳定优势。

APA A/B 输入完全相同，输出不同，且没有设备、温度、偏置或老化元数据，
只能称为 cross-capture。这里是正向 PA 建模，不能宣称硬件 DPD
线性化。本次使用原生 OpenDPD 模型和训练配方家族，不复现原论文的
DPA_200MHz、约 500 参数、100 epochs 和硬件测量设置。

## 图表、视频与目视 QA

`figures/` 中两张图均导出 PNG（300 dpi）、SVG、PDF。
`videos/apa_b_residual_demo.mp4` 是固定 GRU/MoE、4,096 总标签、partial
适应的 15 秒残差 Demo，180 帧、12 fps、1600×900、H.264 yuv420p。
它按时间扫描整个 B test 的 512 点展示窗，复用同一次连续测试推理，
不为动画重新重置状态；动画是慢放，不是 RF 实时播放。

已实际查看两张最终图及视频第 0/90/179 帧。修复了图例覆盖 MoE
曲线、以及视频底部指标与坐标标签靠得过近的问题。最终图表能够区分
full/partial、58/48 更新参数和传统完整目标训练参考；视频能够显示
GRU head 残差明显小于 MoE gate。180 帧完整解码且均不同。
详见 `visual_qa.json` 和 `videos/manifest.json`。

## 复现命令

在 `experiments/` 目录运行：

```powershell
.venv/Scripts/python.exe -B -u scripts/pa_single_demo.py estimate
.venv/Scripts/python.exe -B -u scripts/pa_single_demo.py run
.venv/Scripts/python.exe -B -u scripts/pa_single_demo.py render
```

`run` 会重用已完整完成的 fit，不需要再次训练；如果只修改图表，可用
`render --figures-only` 重用已验证的视频。启动任何新的 GPU 实验前先
确认其它训练已退出。`prepare/estimate` 只导入标准库；`render` 使用 CPU。
