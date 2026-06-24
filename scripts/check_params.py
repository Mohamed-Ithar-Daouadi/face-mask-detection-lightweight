import torch

for name, path in [
    ("YOLOv5n", "runs/detect/results/yolov5n/weights/best.pt"),
    ("YOLOv8n", "runs/detect/results/yolov8n/weights/best.pt"),
    ("MobileNet-SSD", "results/mobilenet_ssd/best.pt"),
]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # YOLO saves a dict with 'model'; MobileNet saves a raw state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
    else:
        sd = ckpt
    total = sum(p.numel() for p in sd.values() if hasattr(p, "numel"))
    print(f"{name}: {total:,} parameters")