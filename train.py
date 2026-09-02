"""
桌面物体检测模型训练脚本
用法:
    python train.py                          # 使用默认参数训练
    python train.py --epochs 200 --imgsz 800 # 自定义参数
    python train.py --device 0               # 指定 GPU

依赖: pip install ultralytics
数据集配置见 desk.yaml，训练前请先把图片和标签放进 datasets/desk/ 对应目录。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练 YOLO 桌面物体检测模型")
    parser.add_argument("--data", default="desk.yaml", help="数据集配置文件路径")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练权重（yolov8n/s/m/l/x 等）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", default=None, help="设备：0=GPU, cpu=CPU，默认自动选择")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--patience", type=int, default=50, help="早停 patience（多少轮无提升则停止）")
    parser.add_argument("--project", default="runs/train", help="结果保存目录")
    parser.add_argument("--name", default="desk", help="本次训练的名称")
    parser.add_argument("--resume", nargs="?", const=True, default=None,
                        help="从上次中断处继续训练（可选：直接指定 last.pt 路径）")
    args = parser.parse_args()

    # 加载预训练模型
    model = YOLO(args.model)

    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        patience=args.patience,
        project=str(Path(args.project).resolve()),  # 绝对路径，避免 ultralytics 拼出 runs/detect/runs/train
        name=args.name,
        resume=args.resume,
    )

    # 训练完成后在验证集上评估
    metrics = model.val()
    print("验证结果:", metrics.box.map)

    # 导出为 ONNX，方便部署到机器人端
    model.export(format="onnx", imgsz=args.imgsz)
    print(f"训练完成，权重和 ONNX 已保存到 {args.project}/desk/weights/ 目录。")


if __name__ == "__main__":
    main()
