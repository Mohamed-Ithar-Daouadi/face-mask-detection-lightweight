# 04_train_mobilenet_ssd.py
# Trains SSDLite + MobileNetV3-Large on the face mask detection dataset
# using PyTorch + torchvision with transfer learning from COCO pretrained weights.
#
# Notes on design:
#  - We reuse the existing YOLO-format labels (data/yolo_format) and convert the
#    normalized boxes back to absolute pixel coordinates on-the-fly, because the
#    torchvision detection API expects absolute [xmin, ymin, xmax, ymax] boxes.
#  - We do NOT manually resize images to 320x320. The SSDLite model has an
#    internal transform (GeneralizedRCNNTransform) that resizes the image,
#    normalizes it, AND scales the target boxes together. Doing the resize
#    ourselves would shrink the image but leave the boxes in original
#    coordinates, which silently corrupts the targets. So we only convert the
#    PIL image to a [0,1] tensor and let the model handle the rest.
#  - Weights are saved in float16 (half precision) to match Ultralytics YOLO
#    saving format, ensuring a fair file size comparison between models.

import torch
import torchvision
from functools import partial
from pathlib import Path
from PIL import Image

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import (
    SSDLite320_MobileNet_V3_Large_Weights,
    SSDLiteClassificationHead,
)
from torchvision.models.detection import _utils as det_utils
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/yolo_format")
EPOCHS      = 100          # upper limit — early stopping will usually stop sooner
PATIENCE    = 20           # stop if val loss doesn't improve for this many epochs
BATCH       = 8            # conservative — torchvision loop is less memory-optimized than Ultralytics
LR          = 0.005        # SGD learning rate (standard torchvision detection fine-tuning)
MOMENTUM    = 0.9
WEIGHT_DECAY = 0.0005
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PROJECT     = Path("results/mobilenet_ssd")
NUM_CLASSES = 4            # 3 real classes + 1 background (index 0 is background in torchvision)
SEED        = 50           # match the split seed used in 01_prepare_data.py
# ──────────────────────────────────────────────────────────────────────────────


def save_half(model, path):
    """
    Save model weights in float16 (half precision) to match the Ultralytics
    YOLO saving format and ensure a fair file size comparison between models.
    Integer tensors (e.g. BatchNorm counters) are kept as-is.
    """
    half_state = {
        k: v.half() if v.is_floating_point() else v
        for k, v in model.state_dict().items()
    }
    torch.save(half_state, path)


class FaceMaskDataset(Dataset):
    """
    Reads YOLO-format labels and converts them back to absolute pixel
    coordinates for torchvision's detection API.

    YOLO label line:  class cx cy w h   (all normalized 0-1)
    torchvision wants: boxes [xmin, ymin, xmax, ymax] in absolute pixels,
                       labels as int64 (with 0 reserved for background, so we
                       shift every class id up by 1).
    """

    def __init__(self, split):
        self.img_dir   = DATA_DIR / "images" / split
        self.label_dir = DATA_DIR / "labels" / split
        self.images    = sorted(self.img_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path   = self.images[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        boxes, labels = [], []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])

                    # normalized (cx, cy, w, h) -> absolute (xmin, ymin, xmax, ymax)
                    xmin = (cx - w / 2.0) * W
                    ymin = (cy - h / 2.0) * H
                    xmax = (cx + w / 2.0) * W
                    ymax = (cy + h / 2.0) * H

                    # clamp to image bounds and drop degenerate boxes
                    xmin = max(0.0, min(xmin, W))
                    ymin = max(0.0, min(ymin, H))
                    xmax = max(0.0, min(xmax, W))
                    ymax = max(0.0, min(ymax, H))
                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls + 1)   # +1: leave index 0 for background

        if len(boxes) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx]),
        }

        # Only convert to a [0,1] tensor — the model's internal transform
        # resizes + normalizes the image and scales the boxes together.
        img_t = F.to_tensor(img)

        return img_t, target


def collate_fn(batch):
    # Each image has a different number of boxes, so we can't stack naively.
    return tuple(zip(*batch))


def build_model():
    """
    Create an SSDLite320 + MobileNetV3-Large model with a COCO-pretrained
    backbone, then swap the classification head to predict our NUM_CLASSES.
    """
    # 1) Load the full COCO-pretrained model (backbone + head both pretrained).
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)

    # 2) Figure out how many channels feed into the head, and how many anchors
    #    the model uses per spatial location. We query these from the model so
    #    we never hardcode numbers that could be wrong for a torchvision version.
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer  = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)

    # 3) Replace ONLY the classification head (the part that outputs class
    #    scores). The box-regression head and the backbone stay pretrained.
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
        norm_layer=norm_layer,
    )
    return model


@torch.no_grad()
def evaluate_loss(model, loader):
    """
    torchvision detection models only return the loss dict when in train()
    mode and given targets. We don't update weights here (no_grad + no
    optimizer step), so this gives us a clean validation loss for early
    stopping without touching the model.
    """
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs    = [img.to(DEVICE) for img in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        total += sum(loss_dict.values()).item()
    return total / max(1, len(loader))


def train():
    torch.manual_seed(SEED)
    PROJECT.mkdir(parents=True, exist_ok=True)

    train_set = FaceMaskDataset("train")
    val_set   = FaceMaskDataset("val")

    train_loader = DataLoader(
        train_set, batch_size=BATCH, shuffle=True,
        collate_fn=collate_fn, num_workers=4,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    print("=" * 60)
    print("Training SSDLite + MobileNetV3-Large — Face Mask Detection")
    print(f"  Train:   {len(train_set)} images | Val: {len(val_set)} images")
    print(f"  Device:  {DEVICE}")
    print(f"  Epochs:  up to {EPOCHS} (early stopping patience={PATIENCE})")
    print(f"  Batch:   {BATCH}")
    print(f"  Optim:   SGD(lr={LR}, momentum={MOMENTUM}, wd={WEIGHT_DECAY})")
    print("=" * 60)

    model = build_model().to(DEVICE)

    # Fine-tune the whole network (backbone + heads). The backbone starts from
    # pretrained COCO features; the new classification head starts random.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    # Decay the LR over time so late epochs make smaller, finer updates.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train one epoch ──
        model.train()
        running = 0.0
        for imgs, targets in train_loader:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        # ── Validate ──
        val_loss = evaluate_loss(model, val_loader)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        # ── Save best + early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_half(model, PROJECT / "best.pt")  # float16 — matches YOLO format
            print(f"   ✅ new best val loss {best_val_loss:.4f} — saved best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch} "
                      f"(no val improvement for {PATIENCE} epochs)")
                break

    # Always keep the final-epoch weights too, for reference.
    save_half(model, PROJECT / "last.pt")  # float16 — matches YOLO format

    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"   Best weights: {PROJECT}/best.pt")


if __name__ == "__main__":
    train()