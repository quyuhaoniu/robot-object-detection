"""
续训脚本：从上次中断的 last.pt 继续训练（接续到原定的 100 epochs）。

用法:
    python resume.py

说明:
    train.py 的 --resume 会去 runs/train/desk/ 找 last.pt（旧目录），
    而本次训练在 runs/train/desk-2/，所以这里直接加载 desk-2 的 last.pt 续训。
"""
from ultralytics import YOLO

model = YOLO("runs/train/desk-2/weights/last.pt")
model.train(resume=True)
