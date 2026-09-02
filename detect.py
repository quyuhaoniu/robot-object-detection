"""
实时目标检测：USB 摄像头实时识别桌面物体，实时显示类别、检测框和置信度。

用法:
    python detect.py                      # 自动加载 best 权重，打开摄像头 0
    python detect.py --model best.engine  # 指定 TensorRT 模型
    python detect.py --source 0 --conf 0.5
    python detect.py --source video.mp4   # 也可识别视频文件
    python detect.py --save-video         # 同时保存检测视频 runs/detect/detect.mp4

按键:
    q = 退出
    s = 保存当前帧截图到 runs/detect/
"""

import argparse
import time
from pathlib import Path

import cv2

from detector import Detector, draw_detections


def main():
    parser = argparse.ArgumentParser(description="实时目标检测")
    parser.add_argument("--model", default=None, help="权重路径(.engine/.onnx/.pt)，默认自动查找")
    parser.add_argument("--source", default="0", help="摄像头索引(0)或视频/图片路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", default=None, help="设备，默认自动(0=cuda/cpu)")
    parser.add_argument("--save-dir", default="runs/detect", help="截图保存目录")
    parser.add_argument("--save-video", nargs="?", const="runs/detect/detect.mp4", default=None,
                        help="保存检测视频（可指定输出路径，默认 runs/detect/detect.mp4）")
    parser.add_argument("--video-fps", type=float, default=20.0, help="保存视频的播放帧率")
    args = parser.parse_args()

    det = Detector(args.model, conf=args.conf, device=args.device)

    # 打开摄像头 / 视频
    src = args.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {src}")

    # 固定分辨率：Jetson 上默认分辨率过高会掉帧，640x480 更稳
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    snap_id = 0
    writer = None

    print("按 q 退出，按 s 保存截图")
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = det.detect(frame)
        frame = draw_detections(frame, dets, fps)

        if args.save_video:
            if writer is None:
                Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (w, h))
                if writer.isOpened():
                    print(f"开始保存视频: {args.save_video}（按 q 停止）")
                else:
                    print("警告: 无法创建视频写入器，请改用 .avi 输出")
                    writer = None
            if writer is not None:
                writer.write(frame)

        cv2.imshow("desk-detect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            snap_id += 1
            path = save_dir / f"snap_{snap_id}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"已保存截图: {path}")

        # 平滑 FPS（指数移动平均）
        now = time.time()
        fps = 0.9 * fps + 0.1 / (now - prev + 1e-6)
        prev = now

    if writer is not None:
        writer.release()
        print(f"视频已保存: {args.save_video}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
