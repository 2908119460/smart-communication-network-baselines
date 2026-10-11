# 单模型固定时序规则探索

纯CPU读取已冻结缓存；没有新增神经网络推理、训练或test调参。普通多数投票和EMA作为已有方法基线，不声称新算法。

| Fixed method | Recall | Negative-frame FPR | F1 | Alarm switches/min | Episode recall | Median detected delay (s) |
|---|---:|---:|---:|---:|---:|---:|
| Frame >= 0.25 | 0.9958 | 0.0276 | 0.9835 | 12.1657 | 1.0000 | 0.0000 |
| Causal 2 of 3 | 0.9932 | 0.0255 | 0.9832 | 7.6470 | 1.0000 | 0.0333 |
| Causal 3 of 5 | 0.9904 | 0.0240 | 0.9826 | 6.2566 | 1.0000 | 0.0667 |
| EMA alpha=0.6 | 0.9958 | 0.0244 | 0.9851 | 5.5615 | 1.0000 | 0.0000 |

延迟仅统计检测到的GT连续正例段，漏掉的段另列；GT出现前已响起的报警可能使首报警延迟为0，summary披露这类段数。报警切换跨视频重置，不把不同视频拼成一条时间线。

- Classical majority voting and EMA are established methods; improvement alone is not algorithmic novelty.
- One model checkpoint and one fixed video split; no multi-seed uncertainty or session-disjoint claim.
- Calibration has 24 sparse samples per video: only fixed frame baseline is reported, no temporal calibration.
- All temporal constants are fixed; this is an exploratory test-cache reanalysis, not a preregistered independent test.
- Episode latency is conditional on detection; missed episodes are counted separately.
- Alarm already active at GT onset yields zero first-alarm delay; preexisting alarm counts are disclosed.
- Presence/alarms do not measure full-video box localization or tracking accuracy.
- Timelines draw entire cached recordings for explanation; online decisions use only current/past frames.
