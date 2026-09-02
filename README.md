# robot-object-detection

桌面物体目标检测（ROS2 + Jetson）。实验一：目标检测与识别。

- 检测框架：Ultralytics YOLOv8
- 部署平台：Jetson（USB 摄像头）
- 结果发布：ROS2 标准消息 `vision_msgs/Detection2DArray`

## 目录结构

```
robot-object-detection/
├── desk.yaml            # 数据集配置（类别名 + 路径）
├── datasets/desk/       # 数据集（图片 + YOLO 标注）
├── train.py             # 训练脚本
├── detect.py            # 实时检测（显示类别/框/置信度 + FPS）
├── ros2_detect.py       # ROS2 节点（发布 Detection2DArray）
├── test.py              # 测试脚本（识别率 + 错误案例）
├── detector.py          # 共享检测器（TensorRT/ONNX/PyTorch 自动选择）
└── requirements.txt
```

## 完整流程

### 1. 环境准备

```bash
# PC（训练用）或 Jetson 都装：
pip install -r requirements.txt
```

### 2. 采集与标注数据

1. 用摄像头拍桌面物体照片，放进 `datasets/desk/images/{train,val,test}/`
2. 用 LabelImg / Roboflow 画框标注，生成同名 `.txt` 放进 `datasets/desk/labels/` 对应目录
   - 标注格式：每行 `类别编号 cx cy w h`（0~1 归一化）
3. 确认 `desk.yaml` 里的 `names` 和你的类别一致

### 3. 训练

```bash
python train.py                      # yolov8n, 100 epochs, 640
python train.py --epochs 200 --imgsz 800 --device 0
```

训练完成后权重在 `runs/train/desk/weights/best.pt`，并自动导出 `best.onnx`。

### 4. 在 Jetson 上导出 TensorRT（提速，达到 5 FPS 关键）

```bash
# 把 best.pt 拷到 Jetson，然后在 Jetson 上执行：
yolo export model=runs/train/desk/weights/best.pt format=engine imgsz=640
```

生成 `best.engine`。之后所有脚本会**自动优先加载 engine → onnx → pt**。

### 5. 实时检测（显示类别/框/置信度）

```bash
python detect.py                # USB 摄像头 0
python detect.py --model best.engine
```

按键：`q` 退出，`s` 保存截图。

### 6. ROS2 发布结果

```bash
source /opt/ros/humble/setup.bash
sudo apt install ros-humble-vision-msgs   # 首次
python3 ros2_detect.py                    # 发布到 /detections

# 另开终端查看：
ros2 topic echo /detections
ros2 node list                            # 应看到 /desk_detection
```

消息类型 `vision_msgs/Detection2DArray`，每项含 `bbox`（中心点+宽高）和 `results`（`class_id`=类别名、`score`=置信度）。

### 7. 测试与评估（测 20 物体，识别率 ≥80%）

```bash
python test.py
```

输出到 `runs/test/`：
- `results.json` — 逐图逐物体结果
- `summary.txt` — 汇总（正确识别率、逐类结果、是否达标）
- `errors/` — 错误案例图（绿=真值，红=预测）

## 验收要求对照

| 要求 | 实现 |
|------|------|
| 识别 ≥2 类物体 | `desk.yaml` 默认 3 类，可改 |
| 实时显示类别/框/置信度 | `detect.py` |
| ROS2 发布识别结果 | `ros2_detect.py`（`vision_msgs/Detection2DArray`）|
| 20 物体识别率 ≥80% | `test.py` 自动统计 |
| ≥5 FPS | Jetson 上导出 TensorRT（`best.engine`）|
| 保存测试结果 + 错误案例 | `runs/test/` |
