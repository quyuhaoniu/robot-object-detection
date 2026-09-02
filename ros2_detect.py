import argparse
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from ultralytics import YOLO
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

CLASS_NAMES = {0: "keyboard", 1: "laptop"}

BACKEND_ORDER = ("engine", "onnx", "pt")

COLORS = [
    (0, 255, 0),    # 绿
    (255, 128, 0),  # 橙
    (255, 0, 0),    # 蓝
    (0, 255, 255),  # 黄
]


class Detector:

    def __init__(self, model_path=None, conf=0.25):
        self.conf = conf
        self.model_path = str(self._resolve_model(model_path))
        self.model = YOLO(self.model_path)
        self.backend = Path(self.model_path).suffix.lstrip(".").lower()
        self.names = self._load_names()
        print(f"[Detector] {self.model_path} | backend={self.backend} | 类别数={len(self.names)}")

    def _resolve_model(self, model_path):
        if model_path:
            return Path(model_path)
        here = Path(__file__).parent
        for ext in BACKEND_ORDER:
            candidates = sorted(here.glob(f"best.{ext}"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(
        )

    def _load_names(self):
        names = getattr(self.model, "names", None)
        if isinstance(names, dict) and names:
            return {int(k): str(v) for k, v in names.items()}
        return dict(CLASS_NAMES)

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
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


class DetectionNode(Node):
    def __init__(self, model, conf, source, topic, rate, show, save_video=None, video_fps=20.0):
        super().__init__("desk_detection")
        self.detector = Detector(model, conf=conf)
        self.pub = self.create_publisher(Detection2DArray, topic, 10)
        self.show = show
        self._save_video = save_video
        self._video_fps = video_fps
        self._writer = None

        src = source
        self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._fps = 0.0
        self._fps_log_time = time.time()

        self.create_timer(1.0 / rate, self.callback)
        self.get_logger().info(f"节点已启动，发布到 {topic}（{rate} Hz），Ctrl+C 退出")

    def callback(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        t0 = time.time()
        dets = self.detector.detect
        (frame)
        dt = time.time() - t0
        self._fps = 0.9 * self._fps + 0.1 / (dt + 1e-6)

        now = time.time()
        if now - self._fps_log_time >= 5.0:
            self.get_logger().info(f"推理 FPS: {self._fps:.1f}")
            self._fps_log_time = now

        self.pub.publish(self._build_msg(dets))

        if self.show or self._save_video:
            drawn = draw_detections(frame, dets, self._fps)
            if self._save_video:
                if self._writer is None:
                    self._init_writer(drawn)
                if self._writer is not None:
                    self._writer.write(drawn)
            if self.show:
                cv2.imshow("desk-detect", drawn)
                cv2.waitKey(1)

    def _init_writer(self, frame):
        h, w = frame.shape[:2]
        Path(self._save_video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            self._save_video, cv2.VideoWriter_fourcc(*"mp4v"), self._video_fps, (w, h))
        if writer.isOpened():
            self._writer = writer
            self.get_logger().info(f"开始保存视频: {self._save_video}")
        else:
            self.get_logger().warn(f"无法创建视频写入器: {self._save_video}，请改用 .avi 输出")

    def _build_msg(self, dets):
        msg = Detection2DArray()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="camera")
        for det in dets:
            x1, y1, x2, y2 = det["xyxy"]
            d = Detection2D()
            d.bbox = BoundingBox2D()
            center = d.bbox.center
            if hasattr(center, "position"):
                center.position.x = (x1 + x2) / 2.0
                center.position.y = (y1 + y2) / 2.0
            else:
                center.x = (x1 + x2) / 2.0
                center.y = (y1 + y2) / 2.0
            d.bbox.size_x = x2 - x1
            d.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det["name"]  # 类别名，如 "keyboard" / "laptop"
            hyp.hypothesis.score = det["conf"]
            d.results.append(hyp)
            msg.detections.append(d)
        return msg

    def destroy_node(self):
        if self._writer is not None:
            self._writer.release()
            self.get_logger().info(f"视频已保存: {self._save_video}")
        self.cap.release()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="ROS2 目标检测节点")
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--source", default="0")
    parser.add_argument("--topic", default="/detections")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--no-show", action="store_true", help="不显示窗口")
    parser.add_argument("--save-video", nargs="?", const="detect.mp4", default=None,
                        help="保存检测视频（可指定输出路径，默认 detect.mp4）")
    parser.add_argument("--video-fps", type=float, default=20.0, help="保存视频的播放帧率")
    args = parser.parse_args()

    rclpy.init()
    node = DetectionNode(args.model, args.conf, args.source, args.topic, args.rate,
                         show=not args.no_show, save_video=args.save_video, video_fps=args.video_fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
