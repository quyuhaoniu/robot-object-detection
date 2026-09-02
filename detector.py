"""
共享检测器：负责加载模型（自动选 TensorRT > ONNX > PyTorch）并执行推理。
detect.py / test.py / ros2_detect.py 都复用这个类，避免重复代码。

权重查找优先级（速度从快到慢）:
    best.engine  (TensorRT，Jetson 上最快，需在 Jetson 上导出)
    best.onnx    (ONNX Runtime，跨平台)
    best.pt      (PyTorch 原始权重，最慢但最简单)
"""

from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

BACKEND_ORDER = ["engine", "onnx", "pt"]

# 画框颜色（BGR），按类别编号循环取色
COLORS = [
    (0, 255, 0),    # 绿
    (255, 128, 0),  # 橙
    (255, 0, 0),    # 蓝
    (0, 255, 255),  # 黄
    (255, 0, 255),  # 品红
    (0, 165, 255),  # 橙黄
]


class Detector:
    def __init__(self, model_path=None, conf=0.25, device=None):
        self.conf = conf
        self.device = device
        self.model_path = str(self._resolve_model(model_path))
        self.model = YOLO(self.model_path)
        self.backend = Path(self.model_path).suffix.lstrip(".")
        self.names = self._load_names()
        print(f"[Detector] 模型: {self.model_path} | backend={self.backend} | 类别数={len(self.names)}")

    # ---- 模型加载 ---------------------------------------------------------
    def _resolve_model(self, model_path):
        """未指定权重时，自动在训练产物目录里按优先级找。"""
        if model_path:
            return Path(model_path)
        weights_dir = Path("runs/train/desk/weights")
        for ext in BACKEND_ORDER:
            candidates = sorted(weights_dir.glob(f"best.{ext}"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(
            "找不到训练好的权重。请先运行 train.py，或用 --model 指定 .engine/.onnx/.pt 路径。"
        )

    def _load_names(self):
        """类别名：优先取模型元数据，取不到则回退读 desk.yaml。"""
        names = getattr(self.model, "names", None)
        if names:
            return {int(k): str(v) for k, v in names.items()}
        cfg = Path("desk.yaml")
        if cfg.exists():
            data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            if "names" in data:
                return {int(k): str(v) for k, v in data["names"].items()}
        return {}

    # ---- 推理 -------------------------------------------------------------
    def detect(self, frame):
        """对一帧 BGR 图像推理，返回检测结果列表。

        每个结果: {"cls": 类别编号, "name": 类别名, "conf": 置信度,
                   "xyxy": [x1, y1, x2, y2] (像素坐标)}
        """
        kwargs = {"conf": self.conf, "verbose": False}
        if self.device is not None and self.backend == "pt":
            kwargs["device"] = self.device

        results = self.model(frame, **kwargs)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                dets.append({
                    "cls": cls,
                    "name": self.names.get(cls, str(cls)),
                    "conf": float(box.conf[0]),
                    "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                })
        return dets


def draw_detections(frame, dets, fps=None):
    """在帧上画出检测框（类别 + 置信度），可选显示 FPS。"""
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        color = COLORS[det["cls"] % len(COLORS)]
        label = f"{det['name']} {det['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame
