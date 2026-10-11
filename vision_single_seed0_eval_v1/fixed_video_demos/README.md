# YOLO11n seed0 固定新测试视频

只使用已冻结的单模型测试推理缓存框，CPU渲染四段类别预选的完整测试视频。绿色是DRONE标注，红色是conf=0.25预测。没有YOLO11s或其它训练seed。输出经OpenCV逐帧和FFmpeg完整解码验证；visual_qa等待人工查看快照。
