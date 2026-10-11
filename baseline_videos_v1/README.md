# Baseline 视频与完整训练入口

2026-10-11。覆盖此前重点比较的三个方向：PA/MoE（课题5，另补课题3的真实 NN-DPD 训练）、60 GHz 调度（课题4）、无人机检测（课题6）。其余课题已有文献/资源调研，本轮没有训练它们的 baseline。大模型 finetune 未做。

## 观看

已打开的本地画廊为 <http://127.0.0.1:8765/>。它只提供本目录的文件，可以暂停、拖动进度或全屏；关闭服务后，MP4 文件仍可直接用本机播放器打开。需要重启时在工作目录执行：

```powershell
& .\experiments\.venv\Scripts\python.exe .\experiments\scripts\serve_baseline_videos.py --port 8765
```

页面源码是 [index.html](index.html)，视频索引是 [gallery.json](gallery.json)。直接打开 HTML 文件可能被浏览器禁止读取相邻 JSON，建议使用上面的本地服务。

| 方向 | 已完成训练 | 视频展示 |
|---|---|---|
| YOLO11n | 当前 480 帧数据；264 train / 96 calibration / 120 test；50轮上限、第21轮验证早停，最佳第11轮 | 原图与预测并排的无人机段、鸟类背景段和合辑；原速连续帧，保留漏检/误报 |
| 60 GHz 标准 PPO | seed0，360,000交互、355 rollout、5,636优化更新；全程24.90分钟 | 两个预定test场景，固定环境seed10000；30k PPO / 360k PPO / 固定码本启发式同外部轨迹并排 |
| OpenDPD GRU PA→DPD | 官方APA_200MHz完整split；原API默认配方，两个阶段各150轮；共314.25秒 | 真实PA测试波形/预测/残差；实际学习的DPD输入变换与共同PA surrogate下的无/有补偿输出 |
| PA/MoE与传统MP参考 | 既有8次源模型、32次适应、12次源结构消融、MP选型/拟合 | 数值与静态机制对照见[候选报告](../../notes/候选机制对照结果.md)；视频采用新增完整原GRU baseline，未将不同配方混作架构对照 |

## 具体视频

- [YOLO 连续合辑](vision/baseline_drone_then_background.mp4)，20.8秒。
- [无人机单段](vision/baseline_V_DRONE_001.mp4)，10.03秒。
- [鸟类/背景单段](vision/baseline_V_BIRD_021.mp4)，10.77秒。
- [调度 test12](rl/rl_test12_load11_block12_seed10000.mp4)，20秒。
- [调度 test13](rl/rl_test13_load10_block10_seed10000.mp4)，20秒。
- [PA 真实波形拟合](pa/pa_forward_modeling.mp4)，20秒。
- DPD最终展示文件为 `pa/dpd_common_surrogate_v2.mp4`；排版修订前的 `dpd_common_surrogate.mp4` 保留，训练结果未修改。来源与解码凭据以 [gallery.json](gallery.json) 与 [PA视频manifest](pa/manifest.json) 为准。

## 如何解释效果

YOLO：绿框是真实无人机标注，红框是预测，conf=.25是固定展示阈值。原数据20段视频此前已经过smoke审阅；连续视频的presence计数不能替代正式IoU/mAP，也不是新独立测试。背景中的很多误框落在树枝/纹理，不等于鸟本体被识别为无人机。

RL：20秒视频慢放warmup后15秒仿真，抽样200个画面；每个画面只展示选中slot的动作，跨过的slot仍在仿真里实际执行。位置/遮挡取服务所用slot-start值，队列取服务后值；同场景三种方法的到达、几何、遮挡与pathloss配对。训练预算收益不是创新，单条轨迹不能替代十个环境seed统计；标准PPO不是原作者算法全部设置的精确复现。

PA/DPD：PA test相对真实输出NMSE为−38.8314dB。共同冻结PA surrogate下，无/有DPD相对线性目标NMSE为−13.8764/−38.6094dB，legacy ACLR均值−27.6505/−49.7355dB。补偿输出都是模型预测，没有RF硬件重采集。原frame200配方比此前v3的frame50/custom GRU拟合更强，但这不是公平架构对照；此前MoE提升需在更强共同配方下重新确认。

## 来源与核验

三个方向的[视觉manifest](vision/manifest.json)、[调度manifest](rl/manifest.json)、[PA manifest](pa/manifest.json)记录输入/模型哈希、选择规则、显示范围、编码与逐帧解码结果。固定首/中/末帧均保存并目视检查；最终画廊还检查浏览器实际加载、播放与拖动。

训练记录仍在原run，视频不包含重新训练的候选机制或额外大模型：

- [原GRU PA→DPD完整训练](../opendpd_gru_full_150ep_seed0/result.json)
- [PA完整基线说明](../../notes/opendpd_gru_full_baseline.md)
- [PPO 360k训练与预算审计](../rl_ppo_360k_seed0_v1/budget_diagnostic.json)
- [YOLO完整训练摘要](../vision_candidate_baseline_seed42/training_summary.json)

本机只在workspace隔离环境添加视频编码依赖，版本见 `experiments/requirements-video-local.txt`，没有更改全局Python环境。
