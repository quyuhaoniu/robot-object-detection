"""
把 labelme 的 JSON 标注转成 YOLO 格式（test.py 需要的格式）。

用法:
    python labelme2yolo.py --json-dir <JSON目录> --out-dir datasets/desk/labels/test

每张图对应一个 .json，转出同名 .txt，每行: class_id cx cy w h（归一化 0~1）。
"""

import argparse
import json
from pathlib import Path

# 类别顺序必须和 desk.yaml 一致（0=keyboard 1=laptop）
CLASSES = ["keyboard", "laptop"]


def shape_to_bbox(points):
    """从任意形状（矩形/多边形）的点列表算出外接框 (x1, y1, x2, y2)。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def convert(json_dir, out_dir):
    json_dir = Path(json_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsons = sorted(json_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"{json_dir} 下没有 .json 文件，请先检查路径。")

    for jf in jsons:
        data = json.loads(jf.read_text(encoding="utf-8"))
        w = data.get("imageWidth")
        h = data.get("imageHeight")

        # 万一 JSON 里没有宽高，回退读同名图片
        if not w or not h:
            import cv2
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                img = json_dir / (jf.stem + ext)
                if img.exists():
                    im = cv2.imread(str(img))
                    if im is not None:
                        h, w = im.shape[:2]
                        break
        if not w or not h:
            print(f"  [跳过] {jf.name}: 拿不到图片宽高")
            continue

        lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            if label not in CLASSES:
                print(f"  [警告] {jf.name} 里有未知类别 '{label}'，已跳过")
                continue
            x1, y1, x2, y2 = shape_to_bbox(shape["points"])
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{CLASSES.index(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out = out_dir / (jf.stem + ".txt")
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"  {jf.name} -> {out.name} ({len(lines)} 个框)")

    print(f"\n完成：共 {len(jsons)} 张，输出到 {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="labelme JSON -> YOLO txt")
    parser.add_argument("--json-dir", default="labelme_json", help="labelme 保存的 JSON 目录")
    parser.add_argument("--out-dir", default="datasets/desk/labels/test", help="输出的 YOLO txt 目录")
    args = parser.parse_args()
    convert(args.json_dir, args.out_dir)
