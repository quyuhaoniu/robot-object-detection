"""
对 train 里的自拍图做光线/颜色增强，扩充训练数据（只改 train，不碰 test，无泄漏）。

对每张自拍图生成 6 个变体：
    bright        加亮（模拟强光）
    dark          变暗（模拟弱光）
    contrast_high 高对比度
    contrast_low  低对比度（发灰）
    hue           色相偏移（模拟暖光/不同色温）
    flip          水平翻转（标签 cx 也同步翻转）

用法:
    python augment.py
"""

import cv2
import numpy as np
from pathlib import Path

TRAIN_IMG = Path("datasets/desk/images/train")
TRAIN_LBL = Path("datasets/desk/labels/train")

# 已经生成的变体后缀（再次运行时会跳过这些）
NEW_SUFFIXES = ("bright", "dark", "contrast_high", "contrast_low", "hue", "flip")


def imread_unicode(path):
    """用 Unicode 安全方式读图（中文文件名在 Windows 上 cv2.imread 可能失败）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


def flip_label(text):
    """水平翻转标注：cx -> 1-cx，其余不变。"""
    out = []
    for line in text.strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx = 1.0 - float(p[1])
        out.append(f"{p[0]} {cx:.6f} {p[2]} {p[3]} {p[4]}")
    return "\n".join(out) + ("\n" if out else "")


def hsv_shift(img, dh=0.0, ds=1.0, dv=1.0):
    """色相/饱和度/亮度调整（模拟不同光线色温）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * ds, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * dv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def main():
    # 只处理自拍图原图，跳过已生成的变体副本
    imgs = [
        p for p in sorted(TRAIN_IMG.glob("微信图片*.jpg"))
        if p.stem.rsplit("_", 1)[-1] not in NEW_SUFFIXES and "_aug" not in p.stem
    ]
    print(f"待增强的自拍原图: {len(imgs)} 张")

    total = 0
    for img_path in imgs:
        img = imread_unicode(img_path)
        if img is None:
            print(f"  [跳过] {img_path.name}: 读图失败")
            continue
        base = img_path.stem
        lbl = TRAIN_LBL / f"{base}.txt"
        if not lbl.exists():
            print(f"  [跳过] {base}: 无标注")
            continue
        label_text = lbl.read_text(encoding="utf-8")
        flipped = flip_label(label_text)

        variants = {
            "bright":        cv2.convertScaleAbs(img, alpha=1.0, beta=45),
            "dark":          cv2.convertScaleAbs(img, alpha=1.0, beta=-45),
            "contrast_high": cv2.convertScaleAbs(img, alpha=1.3, beta=0),
            "contrast_low":  cv2.convertScaleAbs(img, alpha=0.7, beta=0),
            "hue":           hsv_shift(img, dh=12, ds=1.1, dv=1.0),
            "flip":          cv2.flip(img, 1),
        }
        for name, aug in variants.items():
            out_img = TRAIN_IMG / f"{base}_{name}.jpg"
            out_lbl = TRAIN_LBL / f"{base}_{name}.txt"
            if out_img.exists():
                continue
            imwrite_unicode(out_img, aug)
            text = flipped if name == "flip" else label_text
            out_lbl.write_text(text, encoding="utf-8")
            total += 1
        print(f"  {base} -> 6 个变体")

    print(f"\n完成：新增 {total} 张增强图")


if __name__ == "__main__":
    main()
