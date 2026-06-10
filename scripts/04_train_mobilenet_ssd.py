# 04_train_mobilenet_ssd.py
# Trains MobileNetV2 + SSD on the face mask detection dataset
# using PyTorch + torchvision with transfer learning

import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/yolo_format")
EPOCHS     = 100
PATIENCE   = 20           # early stopping patience
BATCH      = 8            # smaller than YOLO — SSD needs more memory per image
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR         = 0.001        # learning rate
PROJECT    = Path("results/mobilenet_ssd")
NUM_CLASSES = 4           # 3 classes + 1 background (torchvision always needs background)
# ──────────────────────────────────────────────────────────────────────────────

class FaceMaskDataset(Dataset):
    """
    Reads YOLO-format labels and converts back to absolute pixel coordinates
    for torchvision's detection API.
    
    YOLO format:  class cx cy w h  (normalized 0-1)
    Torchvision:  [xmin, ymin, xmax, ymax]  (absolute pixels)
    """
    def __init__(self, split, transforms=None):
        self.img_dir   = DATA_DIR / "images" / split
        self.label_dir = DATA_DIR / "labels" / split
        self.transforms = transforms
        self.images = sorted(self.img_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path   = self.images[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # Load image
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Parse YOLO labels → absolute pixel boxes
        boxes  = []
        labels = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # Convert normalized → absolute pixels
                    xmin = (cx - w / 2) * W
                    ymin = (cy - h / 2) * H
                    xmax = (cx + w / 2) * W
                    ymax = (cy + h / 2) * H

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls + 1)  # +1 because 0 is background in torchvision

        # Handle images with no annotations
        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target


def get_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((320, 320)),   # SSDLite expects 320x320
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],              # ImageNet mean
            std=[0.229, 0.224, 0.225]                # ImageNet std
        )
    ])


def collate_fn(batch):
    # torchvision detection models need a custom collate
    # because each image has a different number of boxes
    return tuple(zip(*batch))


def train():
    PROJECT.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    transforms  = get_transforms()
    train_set   = FaceMaskDataset("train", transforms)
    val_set     = FaceMaskDataset("val",   transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_set)} images | Val: {len(val_set)} images")

    # ── Model ─────────────────────────────────────────────────────────────────
    # Load pretrained MobileNetV3 + SSDLite (pretrained on COCO)
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    )

    # Replace the classification head for our 4 classes (3 + background)
    in_channels = [672, 480, 512, 256, 256, 128]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = torchvision.models.detection.ssdlite._prediction_block(
        in_channels, num_anchors, NUM_CLASSES, norm_layer=torch.nn.BatchNorm2d
    )

    model.to(DEVICE)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Only train the head — freeze the backbone (transfer learning)
    params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LR)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    print("=" * 60)
    print("Training MobileNet-SSD — Face Mask Detection")
    print(f"  Device:  {DEVICE}")
    print(f"  Epochs:  up to {EPOCHS} (patience={PATIENCE})")
    print(f"  Batch:   {BATCH}")
    print(f"  LR:      {LR}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs    = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        model.train()  # torchvision SSD only returns loss in train mode
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs    = [img.to(DEVICE) for img in imgs]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                val_loss += sum(loss_dict.values()).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d}/{EPOCHS} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # ── Save best & early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), PROJECT / "best.pt")
            print(f"  ✅ New best val loss: {best_val_loss:.4f} — model saved")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"   Best weights saved to: {PROJECT}/best.pt")


if __name__ == "__main__":
    train()