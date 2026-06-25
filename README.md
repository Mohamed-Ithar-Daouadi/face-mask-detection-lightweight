# Face Mask Detection — Lightweight Edge Deployment

A comparative study of three lightweight object-detection models for real-time face mask compliance detection on edge devices.

**Models compared:** YOLOv5n · YOLOv8n · MobileNet-SSD (SSDLite + MobileNetV3-Large)

---

## Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Speed (ms/img) | Size (MB) |
|---|---|---|---|---|
| YOLOv5n | **0.925** | 0.703 | 13.46 | 5.3 |
| YOLOv8n | 0.907 | **0.708** | 12.30 | 6.2 |
| MobileNet-SSD | 0.748 | 0.458 | **8.30** | **4.7** |

> All models evaluated on the same held-out test set using the same mAP implementation (torchmetrics), following the standard COCO protocol.

---

## Project Structure

```
face-mask-detection-lightweight/
├── data/
│   └── yolo_format/          # Preprocessed dataset (train/val/test splits)
├── Face_Mask_Detection_Dataset/  # Raw dataset (images + XML annotations)
├── results/
│   ├── yolov5n/              # YOLOv5n weights and training plots
│   ├── yolov8n/              # YOLOv8n weights and training plots
│   ├── mobilenet_ssd/        # MobileNet-SSD weights
│   ├── figures/              # Comparison charts
│   └── comparison.csv        # Final evaluation results
├── scripts/
│   ├── 01_prepare_data.py    # Convert Pascal VOC to YOLO format + split
│   ├── 02_train_yolov5n.py   # Train YOLOv5n
│   ├── 03_train_yolov8n.py   # Train YOLOv8n
│   ├── 04_train_mobilenet_ssd.py  # Train MobileNet-SSD
│   ├── 05_evaluate_all.py    # Evaluate all three models on test set
│   └── 06_compare_results.py # Generate comparison charts
├── dataset.yaml              # YOLO dataset config
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU recommended

```bash
git clone https://github.com/Mohamed-Ithar-Daouadi/face-mask-detection-lightweight.git
cd face-mask-detection-lightweight
pip install -r requirements.txt
```

---

## Dataset

Download the Face Mask Detection dataset from Kaggle:
[https://www.kaggle.com/datasets/andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

Place the extracted folder as `Face_Mask_Detection_Dataset/` in the project root. It should contain two subfolders: `images/` and `annotations/`.

---

## Usage

Run the scripts in order:

**1. Prepare the data**
```bash
python scripts/01_prepare_data.py
```
Converts Pascal VOC annotations to YOLO format and splits the dataset 70/15/15.

**2. Train the models**
```bash
python scripts/02_train_yolov5n.py
python scripts/03_train_yolov8n.py
python scripts/04_train_mobilenet_ssd.py
```
All models use early stopping (patience = 20) with a maximum of 100 epochs. Training runs on GPU automatically if available.

**3. Evaluate all models**
```bash
python scripts/05_evaluate_all.py
```
Runs all three trained models on the test set and computes mAP@0.5, mAP@0.5:0.95, inference speed and model size. Saves results to `results/comparison.csv`.

**4. Generate comparison charts**
```bash
python scripts/06_compare_results.py
```
Produces four charts saved to `results/figures/`: accuracy bar chart, speed bar chart, size bar chart and an accuracy-versus-speed scatter plot.

---

## Key Design Decisions

- **Same split for all models** — fixed seed ensures reproducibility and a fair comparison
- **Transfer learning** — all models start from COCO-pretrained weights
- **Augmentation** — mosaic and vertical flip disabled (unrealistic for deployment); horizontal flip and colour jitter kept
- **Float16 checkpoints** — all model weights saved in half precision for a fair file-size comparison
- **Framework-independent evaluation** — torchmetrics mAP used for all three models so results are directly comparable

---

## Project Context

This project was developed as part of the Deep Vision course at OTH Amberg-Weiden. The full technical report is available in the repository.

---

## License

MIT