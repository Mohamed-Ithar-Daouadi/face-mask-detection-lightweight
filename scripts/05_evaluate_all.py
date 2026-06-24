# 05_evaluate_all.py
# Evaluates all three trained models (YOLOv5n, YOLOv8n, MobileNet-SSD) on the
# SAME held-out test set, using the SAME mAP calculation for a fair comparison.
#
# Why this script exists:
#  - During training, the YOLO models reported mAP on the VALIDATION set, and
#    MobileNet-SSD reported only a loss. None of those numbers are comparable:
#    loss formulas differ across frameworks, and val != test.
#  - Here we compute mAP@0.5 with torchmetrics (framework-agnostic) on the TEST
#    set, so all three models are judged by an identical yardstick on data none
#    of them saw during training or model selection.
#
# For each model we report:
#   - mAP@0.5            (primary metric for this use case)
#   - mAP@0.5:0.95       (stricter localization metric, for completeness)
#   - inference speed    (ms per image, GPU, warmed up)
#   - model size         (MB on disk)
#
# Requires: pip install torchmetrics pycocotools

import time
from functools import partial
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import (
    SSDLite320_MobileNet_V3_Large_Weights,
    SSDLiteClassificationHead,
)
from torchvision.models.detection import _utils as det_utils

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/yolo_format")
TEST_IMAGES = DATA_DIR / "images" / "test"
TEST_LABELS = DATA_DIR / "labels" / "test"
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4            # 3 classes + background (MobileNet-SSD)
WARMUP      = 5            # warmup inferences before timing (GPU spin-up)
CLASS_NAMES = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrectly"}
# ──────────────────────────────────────────────────────────────────────────────


def find_weights(candidates):
    """Return the first path that exists, else search via rglob."""
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    # fallback: search the tree for the filename
    name = Path(candidates[0]).name
    for root in ["runs", "results", "."]:
        hits = list(Path(root).rglob(name)) if Path(root).exists() else []
        # prefer paths that mention the model folder hint
        for h in hits:
            return h
    return None


def read_ground_truth(label_path, W, H):
    """YOLO label file -> torchmetrics target dict (xyxy abs pixels, labels 0-2)."""
    boxes, labels = [], []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                xmin = (cx - w / 2) * W
                ymin = (cy - h / 2) * H
                xmax = (cx + w / 2) * W
                ymax = (cy + h / 2) * H
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls)
    if boxes:
        return {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
    return {
        "boxes":  torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,),   dtype=torch.int64),
    }


def build_mobilenet():
    """Recreate the exact architecture used in training, head swapped to 4 classes."""
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer  = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
        norm_layer=norm_layer,
    )
    return model


def evaluate_yolo(weights_path, test_imgs):
    """Run a YOLO model over the test set; return (metric_result, ms_per_image)."""
    model = YOLO(str(weights_path))
    metric = MeanAveragePrecision(box_format="xyxy")

    times = []
    for i, img_path in enumerate(test_imgs):
        img = Image.open(img_path)
        W, H = img.size

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model(str(img_path), verbose=False, device=0 if DEVICE.type == "cuda" else "cpu")
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= WARMUP:
            times.append(dt)

        r = results[0].boxes
        pred = {
            "boxes":  r.xyxy.cpu(),
            "scores": r.conf.cpu(),
            "labels": r.cls.cpu().to(torch.int64),
        }
        gt = read_ground_truth(TEST_LABELS / (img_path.stem + ".txt"), W, H)
        metric.update([pred], [gt])

    ms = 1000.0 * sum(times) / max(1, len(times))
    return metric.compute(), ms


def evaluate_mobilenet(weights_path, test_imgs):
    """Run MobileNet-SSD over the test set; return (metric_result, ms_per_image)."""
    model = build_mobilenet()
    model.load_state_dict(torch.load(str(weights_path), map_location=DEVICE))
    model.eval().to(DEVICE)
    metric = MeanAveragePrecision(box_format="xyxy")

    times = []
    with torch.no_grad():
        for i, img_path in enumerate(test_imgs):
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            img_t = F.to_tensor(img).to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model([img_t])
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= WARMUP:
                times.append(dt)

            o = out[0]
            pred = {
                "boxes":  o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                # training shifted labels +1 (0=background); shift back to 0-2
                "labels": (o["labels"].cpu() - 1).to(torch.int64),
            }
            gt = read_ground_truth(TEST_LABELS / (img_path.stem + ".txt"), W, H)
            metric.update([pred], [gt])

    ms = 1000.0 * sum(times) / max(1, len(times))
    return metric.compute(), ms


def size_mb(path):
    return Path(path).stat().st_size / 1e6


def main():
    test_imgs = sorted(TEST_IMAGES.glob("*.png"))
    print(f"Test set: {len(test_imgs)} images\n")

    models = {
        "YOLOv5n": dict(
            kind="yolo",
            weights=find_weights([
                "runs/detect/results/yolov5n/weights/best.pt",
                "results/yolov5n/weights/best.pt",
            ]),
        ),
        "YOLOv8n": dict(
            kind="yolo",
            weights=find_weights([
                "runs/detect/results/yolov8n/weights/best.pt",
                "results/yolov8n/weights/best.pt",
            ]),
        ),
        "MobileNet-SSD": dict(
            kind="mobilenet",
            weights=find_weights([
                "results/mobilenet_ssd/best.pt",
            ]),
        ),
    }

    rows = []
    for name, cfg in models.items():
        w = cfg["weights"]
        if w is None or not Path(w).exists():
            print(f"⚠ {name}: weights not found, skipping")
            continue

        print(f"Evaluating {name}  ({w}) ...")
        if cfg["kind"] == "yolo":
            res, ms = evaluate_yolo(w, test_imgs)
        else:
            res, ms = evaluate_mobilenet(w, test_imgs)

        map50    = float(res["map_50"])
        map5095  = float(res["map"])
        rows.append((name, map50, map5095, ms, size_mb(w)))
        print(f"  mAP@0.5 = {map50:.3f} | mAP@0.5:0.95 = {map5095:.3f} | "
              f"{ms:.2f} ms/img | {size_mb(w):.1f} MB\n")

    # ── Summary table ──
    print("=" * 74)
    print(f"{'Model':<16}{'mAP@0.5':>10}{'mAP@.5:.95':>12}{'ms/img':>10}{'size(MB)':>12}")
    print("-" * 74)
    for name, m50, m5095, ms, sz in rows:
        print(f"{name:<16}{m50:>10.3f}{m5095:>12.3f}{ms:>10.2f}{sz:>12.1f}")
    print("=" * 74)

    # save to CSV for the report
    out = Path("results/comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("model,map50,map5095,ms_per_image,size_mb\n")
        for name, m50, m5095, ms, sz in rows:
            f.write(f"{name},{m50:.4f},{m5095:.4f},{ms:.3f},{sz:.2f}\n")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()