# YOLO11-STSD：Ultralytics 兼容实现与验证说明

更新：2026-10-14。目标是保持 YOLO11n 的检测头和训练接口，在特征网络中加入空间注意力、动态尺度融合与可选时序融合。代码和原 baseline 分开存放；工程可运行不等于检测性能已经提升。

当前数据来自 [DroneDetectionThesis / Svanström](https://github.com/DroneDetectionThesis/Drone-detection-dataset)，不是官方 Anti-UAV 数据集。YOLO11-STSD 面向无人机检测任务；是否能用于官方 Anti-UAV，需要转换对应标注并另行训练验证。

## 当前实现验证结果

工程检查已通过：模块 shape、反传、YAML 构建、原 Detect stride、有限检测 loss、训练优化和权重重新加载有效；原 baseline 安装包未修改。STSD 参数量 4,759,077，原单类 YOLO11n 为 2,590,035，增加约 83.75%。因此当前实现不是零额外计算的轻量替换。

当前有效短训练已完成：128 张训练采样帧、32 张验证采样帧、120 对同视频采样帧，来自 8 个训练视频；8 epochs、每轮 60 个 batch，共 480 次优化更新，约 186 秒。模型从正确映射的原基线初始化，训练器接收初始化已检查；checkpoint 保存再加载有效。相邻采样帧在原始视频中相隔 13–14 帧，不等于连续原视频帧。

| 同一 32 张验证帧 | mAP50 | mAP50–95 | Precision | Recall | 预算 |
|---|---:|---:|---:|---:|---|
| 已完成原 YOLO11n baseline | 0.9950 | 0.4047 | 0.9632 | 1.0000 | 原完整训练 |
| 当前 STSD 有效短训练 | 0.4983 | 0.2908 | 0.6667 | 0.2500 | 8 轮、128 train 帧 |

STSD 已产生非零检测结果，工程可运行性通过，但当前短训练弱于已完成原基线，尚未验证性能收益。预算和训练数据量不同，这些数字也不是最终同预算消融。Precision / Recall 是 Ultralytics 在其工作点报告的检测指标，不是此前表格中的“是否有无人机”的帧级召回。后续需要同划分、匹配预算的单 seed 对照，再分别验证三个模块。

视频工程回放完成 64 帧；首帧无历史，后 63 帧使用缓存，换流重置通过检查。GIF 用于检查视频输入、特征状态与检测输出流程，本身不能证明三个模块改善了检测。

历史保留两项记录：正确初始化后的 2 轮小预算试跑 mAP 为 0，归档为 `two_epoch_summary.json`；更早一项训练器未保留映射初始化的试跑归档为 `probe_train_uninitialized_attempt`，属于无效试跑，不作性能证据。当前网页主结果读取 `summary.json` 中最新有效的 8 轮实验。

## 代码位置与注册方式

原基线使用 `experiments/.venv/Lib/site-packages/ultralytics/` 中的 Ultralytics 8.3.253。STSD 独立源码在 `experiments/sources/ultralytics_stsd/ultralytics/`，运行入口仅为 STSD 任务加载此目录，原 baseline 的代码、模型和结果保留。

| 文件 | 修改用途 | 本地代码 |
|---|---|---|
| `nn/modules/SkyPriorModule.py` | shape 不变的空间注意力 | [完整代码](../vision_yolo11_stsd_v1/src/SkyPriorModule.py) |
| `nn/modules/DynamicScaleNeck.py` | 通道映射、尺度对齐和动态权重 | [完整代码](../vision_yolo11_stsd_v1/src/DynamicScaleNeck.py) |
| `nn/modules/TemporalFeatureModule.py` | 成对训练和视频缓存 | [完整代码](../vision_yolo11_stsd_v1/src/TemporalFeatureModule.py) |
| `nn/modules/__init__.py` | 导出三个类 | [完整代码](../vision_yolo11_stsd_v1/src/modules_init.py) |
| `nn/tasks.py` | 导入模块，`parse_model` 推导通道 | [完整代码](../vision_yolo11_stsd_v1/src/tasks.py) |
| `cfg/models/11/yolo11n-stsd.yaml` | 模型结构和基线层号映射 | [完整配置](../vision_yolo11_stsd_v1/src/yolo11n-stsd.yaml) |
| `experiments/scripts/vision_yolo11_stsd.py` | 检查、短训练、正式训练和视频入口 | [完整代码](../vision_yolo11_stsd_v1/src/vision_yolo11_stsd.py) |

公共网页的 [独立实施报告](../vision_yolo11_stsd_v1/index.html) 提供可查看的完整源码副本与便携代码包。权重、数据和训练缓存不放入公共网页。

Ultralytics YAML 中的模块名由 `tasks.py` 的全局名称空间解析。只修改 `modules/__init__.py` 不够：`tasks.py` 也必须导入新类，并在 `parse_model` 为单输入模块补充实际输入通道，为三输入动态模块传入三个实际通道及目标尺度。

## 结构与兼容方案

```text
Input [B,3,640,640]
  → 原 YOLO11n backbone
  → SkyPriorModule(P3), SkyPriorModule(P4), SkyPriorModule(P5)
  → 原 YOLO11 FPN/PAN
  → DynamicScaleNeck(target=0/1/2) 三个输出分支
  → TemporalFeatureModule(P3/P4/P5)
  → 原 Detect([P3,P4,P5])
```

两个接口问题需要显式处理。首先，直接把三尺度都 resize 到 P3 后只返回一个 tensor，无法接入原 Detect 的三输入接口；同时 P3/P4/P5 的通道数不同，不能直接相加。实现先用 1×1 Conv 映射通道，再分别对齐到每个目标尺度，用三个分支形成完整金字塔。其次，完全删除原 FPN/PAN 会改变大量预训练参数的用途；当前保留原 FPN/PAN，再插入动态融合，优先保证训练和预训练迁移兼容。

新增层会改变原 YOLO 层号。YAML 中 `stsd_base_layer_map` 显式记录原层号到新层号的映射，入口据此迁移已有基线参数；不能直接假设 `model.11` 在新旧模型中表示相同模块。新模块单独初始化，迁移范围与缺失项由实际检查结果记录。

640 输入、nano 宽度下的 shape 如下；B 是 batch 大小。

| 位置 | P3 | P4 | P5 |
|---|---|---|---|
| Backbone 输出 | `[B,128,80,80]` | `[B,128,40,40]` | `[B,256,20,20]` |
| Sky 输出 | `[B,128,80,80]` | `[B,128,40,40]` | `[B,256,20,20]` |
| 原 FPN/PAN 输出 | `[B,64,80,80]` | `[B,128,40,40]` | `[B,256,20,20]` |
| Dynamic 输出 | `[B,64,80,80]` | `[B,128,40,40]` | `[B,256,20,20]` |
| Temporal / Detect 输入 | `[B,64,80,80]` | `[B,128,40,40]` | `[B,256,20,20]` |

检测头 stride 保持 8/16/32；类别数由数据 YAML 决定，当前单类数据采用 `nc=1`。原 YOLO loss、框回归和后处理格式沿用。

## 模块 1：Sky Spatial Prior

结构是 `Conv3×3 → BN → SiLU → Conv1×1 → Sigmoid`，从 `[B,C,H,W]` 生成 `[B,1,H,W]` 注意力 A，再执行：

```python
output = feature * (1.0 + alpha * attention_map)
```

`alpha` 默认以 0.1 初始化并可学习；隐藏通道比例默认 0.25。输出仍是 `[B,C,H,W]`。三个 backbone 输出分别插入模块，P3 尤其对应小目标的较高空间分辨率。

示例 YAML 参数为 `[hidden_ratio, alpha, learnable_alpha]`，实际通道由 parser 自动补入：

```yaml
- [4, 1, SkyPriorModule, [0.25, 0.1, True]]
```

这里没有天空像素标签或天空分割损失，注意力只由检测损失训练。因此 `attention_map` 不能未经验证就叫准确天空概率。采用 `1+alpha×A` 的残差式调制；当 alpha 为正时主要增强高注意力区域，不会硬删除非天空区域。需要通过注意力可视化、复杂背景误报与非天空无人机召回确认它是否学到了期望先验。

测试检查：输入输出 shape 相同、A 的范围为 [0,1]、损失反传后注意力卷积和 alpha 的梯度存在且有限。

## 模块 2：Dynamic Multi-scale Feature Fusion

输入是三个 feature map 的列表。每个分支的 `target` 为 0/1/2，分别对应 P3/P4/P5。

```python
# 以 P3 分支为例：C_out=64，H_target=W_target=80
aligned = [resize(project(Pi), target_size) for Pi in (P3, P4, P5)]
global_feature = torch.cat(aligned, dim=1)       # [B,192,80,80]
weights = softmax(MLP(GAP(global_feature)))     # [B,3]
output = sum(weights[:, i] * aligned[i] for i in range(3))
```

具体实现采用 1×1 Conv 映射通道、nearest resize、GAP、Linear、ReLU、Linear、Softmax。权重是每张图一组，每个目标尺度独立生成；三个权重和为 1。当前是样本级尺度选择，不是逐像素尺度权重。最后一层线性层零初始化，使三权重开始为 1/3。

```yaml
- [[19, 22, 25], 1, DynamicScaleNeck, [0, 4]]
- [[19, 22, 25], 1, DynamicScaleNeck, [1, 4]]
- [[19, 22, 25], 1, DynamicScaleNeck, [2, 4]]
```

测试检查：三个分支输出 shape 与对应原尺度相同，权重非负且每张图和为 1，全部投影分支和 MLP 的梯度可回传。模块增加计算和参数；只有实测延迟与消融才能说明其代价是否合理。

## 模块 3：Temporal Motion Feature

基本融合把 `[B,C,H,W]` 的前帧与当前帧 concat 成 `[B,2C,H,W]`，经 3×3 Conv 映射回 `[B,C,H,W]`。采用可学习 gamma 的残差融合，减小随机初始化对当前特征的扰动：

```python
fused = fusion(torch.cat((previous, current), dim=1))
output = current + gamma * (fused - current)
```

单图推理默认 stateless，输出当前特征；视频显式 `enable_temporal(True)` 后使用缓存。首帧无缓存直接返回当前帧。缓存只保存原始当前特征，采用 detach，不跨帧保留计算图；缓存不进入 state_dict，并在模式/设备/精度变更时清空。形状、设备或 dtype 不匹配时也按首帧处理。

普通 Ultralytics 随机 batch 中，上一 batch 与下一 batch 通常来自不同视频，mosaic 还会混合图像；把它们当连续帧会学习错误时序。当前采用显式成对训练：batch 顺序为 `[previous_0,current_0,previous_1,current_1,...]`，B 必须为偶数。前帧输出原 feature，当前帧与对应前帧融合。训练入口选择同一视频的相邻采样帧，并保持成对几何变换一致；无需改变模型 `[B,3,H,W]` 输入接口。原视频帧间隔在训练记录中列出，不能默认是连续两帧。

```python
for module in model.modules():
    if isinstance(module, TemporalFeatureModule):
        module.set_pair_training(True)  # 仅用于已正确排序的相邻帧 batch
```

普通 shuffled 训练保持兼容并走 identity，不会训练 temporal 分支。不能把这种训练称为完成时序学习。真正的时序训练需要关闭不适合连续帧的 mosaic/mixup 等增强，并确保 sampler 不把不同视频配成对。

视频缓存以 batch lane 为单位，连续调用时每个 lane 必须对应同一视频流。换视频、跳转、改变帧顺序或 stream ID 时调用 `reset_memory()`。普通单图、导出和 tracing 使用 stateless 路径；ONNX 的单图导出不会自动包含 Python 视频缓存。缓存拼接也没有光流或运动补偿，目前只能称轻量因果特征融合，不能预设已经解决鸟/无人机混淆。

## 如何检查与训练

从项目根目录使用本项目虚拟环境：

```powershell
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py check
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py probe --epochs 2 --device 0
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py probe --epochs 8 --probe-frames-per-sequence 16 --name probe_train_8ep --device 0
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py train --epochs 80 --batch 8 --device 0
```

默认加载现有数据 YAML 与基线 best 权重，通过层映射初始化新模型；默认训练走成对时序训练入口。8 轮阶段将每个训练视频的采样帧数从 8 扩为 16，当前输出权重为 `experiments/runs/vision_yolo11_stsd_v1/probe_train_8ep/weights/best.pt`。重复运行应使用新的 `--name`，保留旧日志和权重。`probe` 是短训练可行性检查；历史 baseline 跑满/早停训练的成绩不能与少量短训练 probe 直接当作同预算胜负对照。最终对照需相同划分、初始化来源和训练预算；现阶段按用户要求不新增多随机种子矩阵。

视频入口示例：

```powershell
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py video --weights "experiments\runs\vision_yolo11_stsd_v1\probe_train_8ep\weights\best.pt" --source "路径\视频.mp4"
```

省略 video 的 `--weights` 时使用 summary 中最新有效的 best 权重，省略 `--source` 时使用预设视频；当前默认最多 64 个采样帧、stride=2。完整参数以入口 `--help` 为准。多个视频应分别运行或在每个新流开始清缓存，不把上一视频特征延续到下一视频。单图预测则沿用 YOLO 常规接口并保持 temporal 默认关闭。

工程检查应覆盖模块 shape 和反传、YAML 注册、Detect stride、真实检测 loss 有限、优化一步、参数迁移、序列缓存首帧/次帧/清空/变尺寸、pair batch 合法性、checkpoint 保存再加载，以及原 YOLO11n 仍能由原包加载。检查记录与短训练实际数字保存在 STSD `summary.json`，公共实施页展示同一份记录。

## 结果如何读

新增模块是三项待检验假设：空间权重可能减少复杂背景干扰；动态尺度融合可能适应目标尺度变化；真实历史特征可能减少偶发误报。工程测试通过支持模型可训练和接口兼容，不能代替检测效果证据。

短训练看 loss 能否下降、验证 mAP 是否有限，以及视频输出和注意力/尺度权重是否可读。后续单 seed 消融依次比较原模型、+Sky、+Dynamic、+Temporal 与完整模型，关注 mAP50–95、小目标指标、相近无人机召回下鸟/直升机误报和推理延迟。当前不宣称这三种基础机制本身在学术上首次提出，也不预设组合一定有效。
