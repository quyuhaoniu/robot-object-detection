"""
测试脚本：在测试集上评估模型识别率，保存结果和典型错误案例。

用法:
    python test.py                          # 自动加载 best 权重，测 datasets/desk/images/test
    python test.py --model best.engine      # 指定 TensorRT 模型
    python test.py --img-dir <图片目录> --label-dir <标注目录>

评估指标（对应验收要求）:
    正确识别率 = 正确检出物体数 / 测试集标注物体总数  (要求 >= 80%，且物体数 >= 20)
    匹配规则: 预测框与真值框类别相同且 IoU >= 0.5 记为正确。

输出:
    runs/test/results.json   逐图逐物体结果
    runs/test/summary.txt    汇总报告
    runs/test/errors/        错误案例图片（绿色=真值 GT，红色=预测）
"""

import argparse
import json
from pathlib import Path

import cv2

from detector import Detector

IOU_THRESHOLD = 0.5


def load_gt(label_path, img_w, img_h):
    """读取 YOLO 标注文件，返回 [{cls, xyxy}]（像素坐标）。"""
    gts = []
    if not label_path.exists():
        return gts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(x) for x in parts[1:5])
        gts.append({
            "cls": cls,
            "xyxy": [
                (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                (cx + w / 2) * img_w, (cy + h / 2) * img_h,
            ],
        })
    return gts


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(gts, preds):
    """贪心匹配，返回 (正确数, 漏检GT列表, 误检pred列表)。"""
    matched = [False] * len(preds)
    correct = 0
    missed = []
    for gt in gts:
        best_i, best_iou = -1, 0.0
        for i, pr in enumerate(preds):
            if matched[i] or pr["cls"] != gt["cls"]:
                continue
            v = iou(gt["xyxy"], pr["xyxy"])
            if v > best_iou:
                best_iou, best_i = v, i
        if best_i >= 0 and best_iou >= IOU_THRESHOLD:
            correct += 1
            matched[best_i] = True
        else:
            missed.append(gt)
    false_pos = [pr for i, pr in enumerate(preds) if not matched[i]]
    return correct, missed, false_pos


def draw_error(frame, gts, preds, names):
    """画错误案例：绿色框=真值，红色框=预测。"""
    for gt in gts:
        x1, y1, x2, y2 = [int(v) for v in gt["xyxy"]]
        name = names.get(gt["cls"], str(gt["cls"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "GT:" + name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for pr in preds:
        x1, y1, x2, y2 = [int(v) for v in pr["xyxy"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{pr['name']}:{pr['conf']:.2f}", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="测试集评估")
    parser.add_argument("--model", default=None, help="权重路径，默认自动查找")
    parser.add_argument("--img-dir", default="datasets/desk/images/test")
    parser.add_argument("--label-dir", default="datasets/desk/labels/test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="runs/test")
    args = parser.parse_args()

    det = Detector(args.model, conf=args.conf, device=args.device)

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out)
    err_dir = out_dir / "errors"
    err_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    imgs = sorted(p for ext in exts for p in img_dir.glob(ext))
    if not imgs:
        raise FileNotFoundError(f"{img_dir} 下没有测试图片，请先放入测试集。")

    total = correct = fp_total = 0
    per_class = {}          # {类别名: {"total": n, "correct": n}}
    missed_per_class = {}
    image_records = []
    error_cases = []

    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")
        gts = load_gt(label_path, w, h)
        preds = det.detect(frame)

        c, missed, false_pos = match(gts, preds)
        total += len(gts)
        correct += c
        fp_total += len(false_pos)

        for gt in gts:
            name = det.names.get(gt["cls"], str(gt["cls"]))
            per_class.setdefault(name, {"total": 0, "correct": 0})
            per_class[name]["total"] += 1
        for gt in missed:
            name = det.names.get(gt["cls"], str(gt["cls"]))
            missed_per_class[name] = missed_per_class.get(name, 0) + 1

        is_error = len(missed) > 0 or len(false_pos) > 0
        image_records.append({
            "file": img_path.name,
            "gt": len(gts),
            "correct": c,
            "missed": len(missed),
            "false_positive": len(false_pos),
            "ok": not is_error,
        })

        if is_error:
            error_cases.append(img_path.name)
            frame = draw_error(frame, gts, preds, det.names)
            cv2.imwrite(str(err_dir / img_path.name), frame)

    # 逐类 correct = total - missed
    for name, mc in missed_per_class.items():
        if name in per_class:
            per_class[name]["correct"] = per_class[name]["total"] - mc
    # 没漏检的类别 correct = total
    for name, pc in per_class.items():
        if name not in missed_per_class:
            pc["correct"] = pc["total"]

    accuracy = correct / total if total > 0 else 0.0
    results = {
        "iou_threshold": IOU_THRESHOLD,
        "total_objects": total,
        "correct": correct,
        "missed": total - correct,
        "false_positive": fp_total,
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "images": image_records,
        "error_cases": error_cases,
    }

    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总
    lines = [
        "=" * 50,
        "测试结果汇总",
        "=" * 50,
        f"测试物体总数 : {total}",
        f"正确检出     : {correct}",
        f"漏检         : {total - correct}",
        f"误检(多检)   : {fp_total}",
        f"正确识别率   : {accuracy * 100:.1f}%",
        "-" * 50,
        "逐类结果:",
    ]
    for name, pc in per_class.items():
        lines.append(f"  {name:<8} {pc['correct']}/{pc['total']}")
    lines += [
        "-" * 50,
        f"验收要求: 物体数 >= 20 ? {'是' if total >= 20 else '否 (当前 %d)' % total}",
        f"         识别率 >= 80% ? {'是' if accuracy >= 0.8 else '否'}",
        "=" * 50,
    ]
    if error_cases:
        lines.append(f"错误案例已保存到 {err_dir}/ 共 {len(error_cases)} 张")
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\n完整结果: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
