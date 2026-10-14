# YOLO11-STSD v2：直接动态 Neck 实现与验证

更新：2026-10-14。当前默认模型以动态尺度融合直接替代原固定 FPN/PAN，保留 YOLO11n Backbone、三尺度 Detect、原检测损失、标准图像输入与结果格式。

v1 在固定 FPN/PAN 后叠加 Dynamic 模块，未满足用户最终要求的直接替代结构。v2 修正该结构；v1 的独立报告、权重和成绩保留为历史。**v2 的实际检查、短训练和视频结果读取 `experiments/runs/vision_yolo11_stsd_v2/summary.json`，不复用 v1 的成绩。**

当前数据来自 [DroneDetectionThesis / Svanström](https://github.com/DroneDetectionThesis/Drone-detection-dataset)，不是官方 Anti-UAV benchmark。模型面向无人机检测；官方 Anti-UAV 数据上的效果尚未验证。新增机制是待验证的贡献候选，工程可运行不等于性能提升或学术首次提出。

## 当前 v2 实际结果

直接动态 Neck 版本通过 10 项工程检查。模型参数量 3,994,053，原单类 YOLO11n 为 2,590,035；新增模块与直接 Neck 的合计参数仍高于原基线。128 张训练采样帧、32 张验证采样帧、8 轮单 seed 短训练已完成，120 对采样帧来自 8 个视频，原始帧间隔 13–14。每轮 60 个 batch，共 480 次优化更新，约 69.29 秒。361 个兼容基线 tensor 已迁移；训练器接收初始化验证通过（检查 475 个模型 tensor）。

| 同一 32 张验证帧 | mAP50 | mAP50–95 | Precision | Recall | 训练预算 |
|---|---:|---:|---:|---:|---|
| 已完成原 YOLO11n | 0.9950 | 0.4047 | 0.9632 | 1.0000 | 原完整训练 |
| STSD v2 直接动态 Neck | 0.4583 | 0.2187 | 0.6667 | 0.2500 | 8 轮、128 train 帧 |

当前 v2 能训练并产生非零检测，短训练表现弱于已完成原基线，不支持性能提升的结论。删除原固定 Neck 后，新 Dynamic 随机层需要训练适应；这不是完整同预算对照。普通 validation 关闭时序缓存，表中指标只评估单帧检测。视频回放的实际帧数和缓存状态由 summary.json 的 video 部分记录，公共页面读取同一份数据。

## 代码结构与修改文件

原 baseline 使用 `experiments/.venv/Lib/site-packages/ultralytics/` 中的 Ultralytics 8.3.253；STSD 使用独立源码 `experiments/sources/ultralytics_stsd/`。不改原 baseline 安装包、YAML、已训权重和结果。

| 文件 | 用途 | 完整文件 |
|---|---|---|
| `ultralytics/nn/modules/SkyPriorModule.py` | soft 空间注意力和残差调制 | [代码](../vision_yolo11_stsd_v2/src/SkyPriorModule.py) |
| `ultralytics/nn/modules/DynamicScaleNeck.py` | 三输入通道投影、resize、动态尺度权重 | [代码](../vision_yolo11_stsd_v2/src/DynamicScaleNeck.py) |
| `ultralytics/nn/modules/TemporalFeatureModule.py` | paired batch 融合与视频缓存 | [代码](../vision_yolo11_stsd_v2/src/TemporalFeatureModule.py) |
| `ultralytics/nn/modules/__init__.py` | 导出三个模块 | [代码](../vision_yolo11_stsd_v2/src/modules_init.py) |
| `ultralytics/nn/tasks.py` | 导入模块；parse_model 推导通道 | [代码](../vision_yolo11_stsd_v2/src/tasks.py) |
| `ultralytics/cfg/models/11/yolo11n-stsd.yaml` | 默认 v2 直接动态 Neck | [YAML](../vision_yolo11_stsd_v2/src/yolo11n-stsd.yaml) |
| `ultralytics/cfg/models/11/yolo11n-stsd-pan.yaml` | 历史 v1 叠加 PAN 的对照配置 | [YAML](../vision_yolo11_stsd_v2/src/yolo11n-stsd-pan.yaml) |
| `stsd_tools/training.py` | 同视频采样帧配对训练 | [代码](../vision_yolo11_stsd_v2/src/training.py) |
| `stsd_tools/transfer.py` | 显式 baseline 层映射 | [代码](../vision_yolo11_stsd_v2/src/transfer.py) |
| `tests/test_stsd.py` | 模块、网络、缓存、训练和基线不变检查 | [测试](../vision_yolo11_stsd_v2/src/test_stsd.py) |
| `experiments/scripts/vision_yolo11_stsd.py` | check / probe / train / video，默认 v2 | [入口](../vision_yolo11_stsd_v2/src/vision_yolo11_stsd.py) |

Ultralytics 的 YAML 模块名通过 `tasks.py` 全局名称空间解析。因此只在 `modules/__init__.py` 导出不足：`tasks.py` 还需导入新类，在 `parse_model` 处理单输入模块和三输入动态模块。Sky 和 Temporal 自动接收 `ch[f]`，输出通道不变；Dynamic 自动接收三个输入的实际通道，并按 nano width=0.25 缩放 YAML 中指定的输出通道。类名与 YAML 名称一致，原 Detect 仍按标准列表输入处理。

## 直接替代结构与接口冲突

```text
Input [B,3,640,640]
  → YOLO11n Backbone 0–10
  → Sky 11/12/13，分别读取 Backbone P3/P4/P5
  → Dynamic 14/15/16，全部直接读取 [11,12,13]
  → Temporal 17/18/19
  → 原 Detect 20，输入 [17,18,19]
```

用户原公式将三尺度 resize 到 P3 后加权求和，可得到一个 P3 输出；原 Detect 则需要 P3/P4/P5 三个输入，而且三个 Backbone 通道不同。这是需要兼容处理的实际冲突。实现使用三个 DynamicScaleNeck 分支，target=0/1/2，分别对齐到对应目标分辨率，每个源先用 1×1 Conv 投影到目标输出通道。它们直接融合 Sky 后的 Backbone 特征，原固定 FPN/PAN 不再参与默认网络。

| 640 输入、nano 宽度 | P3 / stride 8 | P4 / stride 16 | P5 / stride 32 |
|---|---|---|---|
| Backbone 输出 | `[B,128,80,80]` | `[B,128,40,40]` | `[B,256,20,20]` |
| Sky 输出 | `[B,128,80,80]` | `[B,128,40,40]` | `[B,256,20,20]` |
| Dynamic 输出 | `[B,64,80,80]` | `[B,128,40,40]` | `[B,256,20,20]` |
| Temporal / Detect 输入 | `[B,64,80,80]` | `[B,128,40,40]` | `[B,256,20,20]` |

默认 YAML 有 21 个层级，Detect 层号为 20。基线层映射复用 Backbone 0–10 和原 Detect 23 → 新 20 的兼容 tensor。原固定 Neck 被删除，它的参数不会迁入随机新 Dynamic 层；新增模块独立初始化。Detect 的输入通道和 stride 与原 nano 检测头兼容；实际可迁入 tensor 数由运行记录给出。

## Step 1：SkyPriorModule

```text
[B,C,H,W] → Conv3×3 → BN → SiLU → Conv1×1 → Sigmoid → [B,1,H,W]
output = feature × (1 + alpha × attention_map) → [B,C,H,W]
```

alpha 默认 0.1，可配置为可学习或固定；隐藏通道比例默认 0.25。广播的单通道 attention 保持输入输出 shape 相同，不硬裁掉任何区域。

```yaml
- [4, 1, SkyPriorModule, [0.25, 0.1, True]]   # P3
- [6, 1, SkyPriorModule, [0.25, 0.1, True]]   # P4
- [10, 1, SkyPriorModule, [0.25, 0.1, True]]  # P5
```

当前没有天空像素标签或 sky segmentation loss；输出应称空间注意力，不能未经验证称准确 sky probability。检测损失是否使它学到天空先验，需可视化 attention 和统计天空/非天空无人机表现。测试覆盖 shape 不变、Sigmoid 范围、卷积和 alpha 的有限梯度。

## Step 2–3：模块注册和 YAML

三个类在 `nn/modules/__init__.py` 导出，并由 `nn/tasks.py` 导入。parse_model 补入输入通道；新模块可直接通过 YAML 类名构建。默认 YAML 保留原 Backbone、替换原 Neck、保留原 Detect；原 `yolo11.yaml` 保持不变。

公共 [v2 实现网页](../vision_yolo11_stsd_v2/)提供模块完整代码、完整 `tasks.py`、YAML 和 `registration.patch`，可逐文件审查。便携代码包包含独立 fork、训练工具、测试与入口，按工作目录相对路径组织，不包含权重和数据。

## Step 4：DynamicScaleNeck

构造参数是 `DynamicScaleNeck(channels, target=0, reduction=4, out_channels=None)`。out_channels 省略时沿用目标源通道，用于兼容历史 PAN 配置；默认 v2 明确指定 64/128/256。

```python
aligned = [resize(project(source), target_size) for source in (P3, P4, P5)]
global_feature = torch.cat(aligned, dim=1)
weights = softmax(MLP(GAP(global_feature)))  # [B,3]
output = sum(weights[:, i, None, None, None] * aligned[i] for i in range(3))
```

P3 分支的投影输出均为 `[B,64,80,80]`，concat 后为 `[B,192,80,80]`，GAP/MLP 输出 `[B,3]`，最后 weighted sum 恢复 `[B,64,80,80]`。P4/P5 分支相同，concat 通道分别为 384/768，空间尺寸分别 40×40/20×20。各 target 分支独立计算每图尺度权重；权重非负且每张图和为 1。最后 Linear 零初始化，使权重起始为 1/3。

```yaml
# target / reduction / 输出通道（nano ×0.25 后为 64/128/256）
- [[11, 12, 13], 1, DynamicScaleNeck, [0, 4, 256]]
- [[11, 12, 13], 1, DynamicScaleNeck, [1, 4, 512]]
- [[11, 12, 13], 1, DynamicScaleNeck, [2, 4, 1024]]
```

测试覆盖显式输出通道、默认目标通道兼容、三目标尺度输出、权重和、所有投影的有限梯度与非法输入检查。新动态模块需要训练适应；删除已训固定 Neck 后，不能期待随机初始化立即继承原完整基线精度。

## Step 5：TemporalFeatureModule

```python
# previous/current 都是 [B,C,H,W]
fused = fusion(torch.cat((previous, current), dim=1))  # [B,2C,H,W] → [B,C,H,W]
output = current + gamma * (fused - current)
```

融合用 3×3 Conv，通过可学习 gamma 残差混合减少新模块初始扰动。首帧无 previous 直接返回 current。视频用简单缓存保存原始当前 feature 的 detach 版本，不递归缓存增强后特征，不跨帧保留计算图，也不把缓存写入 state_dict。

单图默认 temporal 关闭，标准 `YOLO(...).predict(image)` 无状态。视频入口显式开启缓存，换视频、跳转、换序列需 `reset_memory()`；形状、dtype、设备不一致和模式切换也清空缓存。多流 batch 每个 lane 必须稳定对应同一视频；单图 ONNX 导出不包含 Python 缓存。

随机 batch 中的图像不能冒充连续帧。实际训练入口从同视频取相邻采样帧，batch 顺序为 `[previous₀,current₀,previous₁,current₁,…]`，前帧走当前特征，后帧与对应前帧融合。batch image 数为偶数；当前 paired Trainer 关闭随机增强，使用确定性 letterbox 保持成对坐标一致，mosaic/mixup 也关闭。它与原基线增强不同，是额外的比较边界。输入仍为标准 `[B,3,H,W]`。

```python
for module in model.modules():
    if isinstance(module, TemporalFeatureModule):
        module.set_pair_training(True)  # 仅在已正确配对的 batch 中开启
```

普通 shuffled Trainer 可用但 temporal 走 identity，不会训练该分支。`--spatial-only` 可显式选择此路径。训练的采样帧间隔由实际记录给出，不默认连续原始视频帧。当前没有光流或相机运动对齐，不能预设轻量历史融合已经解决 bird/drone 误报。测试覆盖首帧、后续帧、reset、变尺寸、batch 配对、非法奇数 batch、detach 与缓存不保存。

## 检查、短训练与正式训练

从项目根目录使用当前虚拟环境运行，默认输出目录为 v2：

```powershell
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py check
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py probe --epochs 8 --probe-frames-per-sequence 16 --name probe_train_8ep --device 0
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py train --epochs 80 --batch 8 --device 0
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py video --weights "experiments\runs\vision_yolo11_stsd_v2\probe_train_8ep\weights\best.pt" --source "路径\视频.mp4" --max-frames 64
```

入口默认加载已有数据配置与原 baseline best，映射 Backbone/Detect 参数后保存/重新加载初始化 checkpoint，并验证 Trainer 接收相同初始化。`probe` 为短训练工程检查；`train` 使用完整现有划分。`--run-dir` 可选择独立结果目录，`--model-yaml` 可选择结构，`--name` 为每个预算保留独立日志和权重。

Anti-UAV 需要先转换成 YOLO 检测格式，训练/验证按视频划分，避免相邻帧跨集合。随后通过 `--data` 指定其 YAML。paired 工具支持 `<sequence>_<frame-number>.jpg` 或每序列文件夹中的数字帧名，按父路径区分序列。如果数据采用另一命名规则，需传显式 pair 索引或适配序列解析，不能用错误配对训练 Temporal。现有默认数据已符合配对工具规则。

使用原 PAN 历史配置时应另设输出目录，避免覆盖 v2：

```powershell
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py probe --model-yaml "experiments\sources\ultralytics_stsd\ultralytics\cfg\models\11\yolo11n-stsd-pan.yaml" --run-dir "experiments\runs\vision_yolo11_stsd_pan_comparison" --epochs 8 --name probe_train_8ep --device 0
```

当前不安排多随机种子矩阵。下一步只做相同划分、初始化来源和训练预算的单 seed 对照，再分别加/去 Sky、Dynamic、Temporal 验证贡献；关注 mAP、小目标、相近召回下近类误报与实际推理延迟。

## 结果记录与阅读边界

v2 工程检查、模型参数、短训练 mAP 与 Precision/Recall、迁移检查、采样帧间隔、视频缓存和尺度权重轨迹保存在 [summary.json](../vision_yolo11_stsd_v2/summary.json)及对应日志。公共网页由同一份实际记录显示数字，完整实验尚未记录时展示待完成。普通 validation 关闭 temporal 缓存，报告单帧检测表现；这些 mAP 不能用于证明时序分支的效果，时序贡献需要单独的连续视频对照。

短训练成功证明网络可构建、计算 loss、优化、保存再加载及处理视频。短预算和已完成基线的预算不同，即使在同一验证子集比较，也不是最终同预算胜负。视频 demo 用于展示检测输出与缓存/权重状态，单个片段不能证明误报减少。三种模块的效果和计算代价仍需消融验证。
