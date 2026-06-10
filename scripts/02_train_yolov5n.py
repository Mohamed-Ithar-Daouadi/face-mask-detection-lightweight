# 02_train_yolov5n.py
# Trains YOLOv5n on the face mask detection dataset
# using transfer learning from COCO pretrained weights

from ultralytics import YOLO
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_YAML  = Path("dataset.yaml")
EPOCHS     = 100          # upper limit — early stopping will kick in earlier
PATIENCE   = 20           # stop if val loss doesn't improve for 20 epochs
IMG_SIZE   = 640          # standard YOLOv5 input size (matches pretrained weights)
BATCH      = 16           # community standard for YOLOv5 fine-tuning on custom datasets
DEVICE     = 0            # force GPU — change to 'cpu' if running locally
PROJECT    = "results"    # output folder
NAME       = "yolov5n"    # subfolder inside results/
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Load pretrained YOLOv5n weights (transfer learning from COCO)
    model = YOLO("yolov5n.pt")

    print("=" * 60)
    print("Training YOLOv5n — Face Mask Detection")
    print(f"  Data:     {DATA_YAML}")
    print(f"  Epochs:   up to {EPOCHS} (early stopping patience={PATIENCE})")
    print(f"  Img size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch:    {BATCH}")
    print(f"  Device:   GPU {DEVICE}")
    print("=" * 60)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        # Augmentation
        mosaic=0.0,      # disabled — creates unrealistic image compositions
        flipud=0.0,      # disabled — faces are rarely upside down in real use
        fliplr=0.5,      # enabled  — horizontal flip is a realistic variation
        hsv_h=0.015,     # subtle hue shift — realistic lighting variation
        hsv_s=0.7,       # saturation shift — realistic camera variation
        hsv_v=0.4,       # brightness shift — realistic lighting variation
    )

    print("\n✅ Training complete!")
    print(f"   Best weights saved to: results/yolov5n/weights/best.pt")

if __name__ == "__main__":
    main()