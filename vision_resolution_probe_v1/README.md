# YOLO11n 640 / 960 验证集分辨率探针

## Material Passport

Origin Skill: academic-research-suite / experiment-agent; Mode: run; Verification Status: ANALYZED; Date: 2026-10-12.

同一冻结模型、同792张validation帧、conf=0.25；无重训、无test/calibration、无阈值网格。

| 输入 | Presence recall | 负帧FPR | IoU≥.50框召回 | IoU≥.75框召回 | TP / FP |
|---|---:|---:|---:|---:|---:|
| 640 | 0.9557 | 0.0392 | 0.8854 | 0.4115 | 367 / 16 |
| 960 | 0.9089 | 0.0123 | 0.8281 | 0.4115 | 349 / 5 |

GT共384个框；640与旧候选缓存presence计数一致：True。

每个GT与预测框做一对一匹配。IoU框召回是固定阈值诊断，不是mAP；高分辨率是否值得作为教师，要同时看负类误报与定位。

叠框例由改善、退化、新增/去除误报规则选择，完整候选预测及分类别指标保留。
- [定位改善 IoU50：V_DRONE_009 f13](examples/comparison_01.jpg)
- [定位退化 IoU50：V_DRONE_009 f143](examples/comparison_02.jpg)
- [误报被去除：V_BIRD_040 f197](examples/comparison_03.jpg)
- [新增误报：V_AIRPLANE_035 f305](examples/comparison_04.jpg)
- [精定位改善 IoU75：V_DRONE_009 f0](examples/comparison_05.jpg)
- [精定位退化 IoU75：V_DRONE_009 f91](examples/comparison_06.jpg)