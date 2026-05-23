# 🍛 Nepali Food Detection — YOLOv8

A real-time object detection system trained to recognise traditional Nepali foods using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Two model variants — **YOLOv8n** (nano) and **YOLOv8s** (small) — were trained and evaluated, offering a trade-off between inference speed and detection accuracy.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction / Inference](#prediction--inference)
  - [Evaluation](#evaluation)
- [Training Results](#training-results)
  - [Metrics Summary](#metrics-summary)
  - [Per-Epoch Metrics — YOLOv8n](#per-epoch-metrics--yolov8n)
  - [Per-Epoch Metrics — YOLOv8s](#per-epoch-metrics--yolov8s)
  - [Prediction Results](#prediction-results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Nepali cuisine encompasses a rich variety of dishes that are visually distinct yet challenging for general-purpose models to classify. This project fine-tunes YOLOv8 on a custom Nepali food dataset to enable:

- **Real-time food detection** from images or video streams
- **Bounding-box localisation** of individual food items
- **Multi-class classification** of Nepali dishes

Two backbone sizes were benchmarked:

| Model    | Parameters | Speed   | Run directory          | Best for              |
|----------|-----------|---------|------------------------|-----------------------|
| YOLOv8n  | ~3.2 M    | Fastest | `runs/detect/train`    | Edge / mobile devices |
| YOLOv8s  | ~11.2 M   | Fast    | `runs/detect/train4`   | Balanced accuracy     |

---

## Repository Structure

```
nepali-food-detection-yolo/
├── Dataset/                    # Training, validation, and test splits
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── runs/
│   └── detect/
│       ├── train-3/              # YOLOv8n — primary training run
│       │   ├── results.csv     # Per-epoch metrics (27 epochs)
│       │   ├── weights/
│       │   │   ├── best.pt     # Best checkpoint (epoch 15, mAP50=29.4%)
│       │   │   └── last.pt
│       │   └── ...             # Plots, confusion matrix, PR curve
│       └── train-4/             # YOLOv8s — training run
│           ├── results.csv     # Per-epoch metrics (27 epochs)
│           ├── weights/
│           │   ├── best.pt     # Best checkpoint (epoch 17, mAP50=32.7%)
│           │   └── last.pt
│           └── ...
├── docs/
│   └── images/                 # Prediction sample images
├── src/                        # Python source scripts
├── requirements.txt
└── .gitignore
```

---

## Dataset

The [dataset](https://www.kaggle.com/datasets/rajeevpaudel1/nepali-food-image-dataset) contains images of traditional Nepali foods annotated in **YOLO format** (`.txt` label files with normalised bounding-box coordinates).

> **Note:** The dataset is stored under `Dataset/` with the standard `images/` and `labels/` split. Update the `data.yaml` path to point to your local clone before training.

Example `data.yaml`:

```yaml
path: ./Dataset
train: images/train
val:   images/val
test:  images/test

nc: <number_of_classes>
names:
  - Momo
  - Chatpate
  - Dal Bhat
  - Sel Roti
  - Samosa
  # ... add all classes
```

---

## Models

| Variant  | Base weights  | Training run         | Best epoch | Best mAP@50 |
|----------|--------------|----------------------|-----------|------------|
| YOLOv8n  | `yolov8n.pt` | `runs/detect/train-3`  | 15        | 29.4%      |
| YOLOv8s  | `yolov8s.pt` | `runs/detect/train-4` | 17        | 32.7%      |

Pre-trained ImageNet weights were used as the starting point and fine-tuned on the Nepali food dataset via transfer learning.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AvisarBhandari/nepali-food-detection-yolo.git
cd nepali-food-detection-yolo
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

- `ultralytics==8.4.51` — YOLOv8 framework
- `opencv-python==4.13.0.92` — image / video processing
- `Pillow==12.2.0` — image I/O

---

## Usage

### Training

**Train YOLOv8n:**

```bash
yolo detect train \
  model=yolov8n.pt \
  data=Dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  project=runs/detect \
  name=train
```

**Train YOLOv8s:**

```bash
yolo detect train \
  model=yolov8s.pt \
  data=Dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  project=runs/detect \
  name=train4
```

Or using the Python API:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")   # or "yolov8n.pt"
model.train(
    data="Dataset/data.yaml",
    epochs=100,
    imgsz=640,
    project="runs/detect",
    name="train4"
)
```

### Prediction / Inference

Run inference on a single image:

```bash
yolo detect predict \
  model=runs/detect/train4/weights/best.pt \
  source=path/to/image.jpg \
  conf=0.25 \
  save=True
```

Run on a folder of images:

```bash
yolo detect predict \
  model=runs/detect/train4/weights/best.pt \
  source=Dataset/images/test/ \
  conf=0.25 \
  save=True
```

Using Python:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")
results = model.predict(source="path/to/image.jpg", conf=0.25, save=True)

for r in results:
    print(r.boxes)   # bounding boxes
    r.show()         # display
```

Prediction outputs are saved under `runs/detect/predict/`.

### Evaluation

```bash
yolo detect val \
  model=runs/detect/train4/weights/best.pt \
  data=Dataset/data.yaml
```

---

## Training Results

### Metrics Summary

Both models were trained for **27 epochs**. Key results at the best checkpoint:

| Metric           | YOLOv8n (`train-3`) | YOLOv8s (`train-4`) |
|-----------------|------------------|-------------------|
| Best epoch       | 15               | **17**            |
| mAP@50           | 29.4%            | **32.7%**         |
| mAP@50-95        | 20.1%            | **23.8%**         |
| Precision        | 21.6%            | 21.5%             |
| Recall           | 60.1%            | **65.3%**         |
| Val box loss     | 1.291            | 1.322             |
| Val cls loss     | 5.013            | 4.074             |
| Val dfl loss     | 1.447            | 1.481             |
| Train time       | ~4.5 min         | ~8.4 min          |

> **YOLOv8s outperforms YOLOv8n** on all accuracy metrics — gaining +3.3 pp mAP@50 and +3.7 pp mAP@50-95 — at the cost of ~2× training time.

---

### Per-Epoch Metrics — YOLOv8n

> Run: `runs/detect/train-3` | Base model: `yolov8n.pt` | 27 epochs

| Epoch | Box Loss | Cls Loss | DFL Loss | Precision | Recall | mAP@50 | mAP@50-95 |
|------:|---------:|---------:|---------:|----------:|-------:|-------:|----------:|
| 1 | 1.3785 | 3.4138 | 1.5652 | 0.323 | 0.137 | 0.094 | 0.054 |
| 2 | 1.4214 | 2.4061 | 1.5562 | 0.154 | 0.372 | 0.171 | 0.103 |
| 3 | 1.4137 | 2.1136 | 1.5447 | 0.098 | 0.452 | 0.161 | 0.105 |
| 4 | 1.4140 | 2.0890 | 1.5230 | 0.107 | 0.387 | 0.143 | 0.081 |
| 5 | 1.3849 | 1.9541 | 1.5195 | 0.155 | 0.498 | 0.239 | 0.164 |
| 6 | 1.3226 | 1.8404 | 1.4652 | 0.170 | 0.645 | 0.245 | 0.155 |
| 7 | 1.3137 | 1.7841 | 1.4728 | 0.186 | 0.567 | 0.244 | 0.151 |
| 8 | 1.3186 | 1.6942 | 1.4471 | 0.261 | 0.506 | 0.233 | 0.160 |
| 9 | 1.2912 | 1.6276 | 1.4255 | 0.187 | 0.588 | 0.220 | 0.152 |
| 10 | 1.2876 | 1.5790 | 1.4141 | 0.194 | 0.626 | 0.274 | 0.182 |
| 11 | 1.3009 | 1.5400 | 1.4208 | 0.216 | 0.651 | 0.276 | 0.189 |
| 12 | 1.2490 | 1.4631 | 1.3904 | 0.176 | 0.744 | 0.247 | 0.162 |
| 13 | 1.2473 | 1.4329 | 1.3932 | 0.203 | 0.709 | 0.290 | 0.199 |
| 14 | 1.2326 | 1.3735 | 1.3867 | 0.181 | 0.716 | 0.266 | 0.186 |
| **15** ⭐ | **1.2250** | **1.3707** | **1.3687** | **0.216** | **0.601** | **0.294** | **0.201** |
| 16 | 1.2048 | 1.3320 | 1.3618 | 0.174 | 0.683 | 0.266 | 0.194 |
| 17 | 1.2116 | 1.2963 | 1.3650 | 0.187 | 0.555 | 0.291 | 0.218 |
| 18 | 1.1712 | 1.2397 | 1.3224 | 0.166 | 0.647 | 0.254 | 0.178 |
| 19 | 1.1629 | 1.1892 | 1.3239 | 0.146 | 0.680 | 0.196 | 0.131 |
| 20 | 1.1594 | 1.1800 | 1.3066 | 0.173 | 0.728 | 0.247 | 0.171 |
| 21 | 1.1421 | 1.1612 | 1.3029 | 0.171 | 0.806 | 0.277 | 0.186 |
| 22 | 1.1618 | 1.1556 | 1.3257 | 0.176 | 0.735 | 0.270 | 0.192 |
| 23 | 1.1385 | 1.1403 | 1.3103 | 0.148 | 0.643 | 0.239 | 0.180 |
| 24 | 1.1208 | 1.1107 | 1.2881 | 0.203 | 0.587 | 0.245 | 0.173 |
| 25 | 1.1200 | 1.1141 | 1.2941 | 0.178 | 0.742 | 0.279 | 0.196 |
| 26 | 1.0883 | 1.0714 | 1.2807 | 0.179 | 0.577 | 0.261 | 0.187 |
| 27 | 1.0887 | 1.0467 | 1.2707 | 0.185 | 0.788 | 0.261 | 0.187 |

⭐ Best checkpoint — saved as `runs/detect/train-3/weights/best.pt`

---

### Per-Epoch Metrics — YOLOv8s

> Run: `runs/detect/train-4` | Base model: `yolov8s.pt` | 27 epochs

| Epoch | Box Loss | Cls Loss | DFL Loss | Precision | Recall | mAP@50 | mAP@50-95 |
|------:|---------:|---------:|---------:|----------:|-------:|-------:|----------:|
| 1 | 1.4221 | 3.2192 | 1.5798 | 0.169 | 0.544 | 0.205 | 0.127 |
| 2 | 1.4214 | 1.9302 | 1.5229 | 0.285 | 0.414 | 0.239 | 0.148 |
| 3 | 1.4081 | 1.6644 | 1.5319 | 0.298 | 0.327 | 0.212 | 0.138 |
| 4 | 1.4455 | 1.7096 | 1.5383 | 0.120 | 0.600 | 0.168 | 0.101 |
| 5 | 1.4210 | 1.6576 | 1.5457 | 0.167 | 0.511 | 0.258 | 0.183 |
| 6 | 1.3695 | 1.6475 | 1.5026 | 0.213 | 0.453 | 0.251 | 0.158 |
| 7 | 1.3556 | 1.5008 | 1.5031 | 0.326 | 0.450 | 0.276 | 0.190 |
| 8 | 1.3274 | 1.4509 | 1.4602 | 0.362 | 0.381 | 0.267 | 0.179 |
| 9 | 1.3298 | 1.3870 | 1.4463 | 0.166 | 0.771 | 0.270 | 0.195 |
| 10 | 1.3007 | 1.3088 | 1.4268 | 0.249 | 0.486 | 0.290 | 0.189 |
| 11 | 1.2975 | 1.3284 | 1.4418 | 0.184 | 0.641 | 0.282 | 0.201 |
| 12 | 1.2417 | 1.2415 | 1.4022 | 0.214 | 0.706 | 0.299 | 0.190 |
| 13 | 1.2526 | 1.2377 | 1.4091 | 0.201 | 0.630 | 0.291 | 0.202 |
| 14 | 1.2341 | 1.1976 | 1.4069 | 0.235 | 0.550 | 0.295 | 0.200 |
| 15 | 1.2078 | 1.1786 | 1.3752 | 0.201 | 0.779 | 0.293 | 0.183 |
| 16 | 1.1832 | 1.1308 | 1.3651 | 0.362 | 0.435 | 0.307 | 0.214 |
| **17** ⭐ | **1.2014** | **1.1148** | **1.3757** | **0.215** | **0.653** | **0.327** | **0.238** |
| 18 | 1.1449 | 1.0628 | 1.3164 | 0.172 | 0.685 | 0.228 | 0.157 |
| 19 | 1.1443 | 1.0210 | 1.3294 | 0.206 | 0.684 | 0.256 | 0.180 |
| 20 | 1.1454 | 1.0095 | 1.3073 | 0.164 | 0.680 | 0.234 | 0.169 |
| 21 | 1.1183 | 0.9925 | 1.2987 | 0.167 | 0.763 | 0.244 | 0.170 |
| 22 | 1.1384 | 0.9806 | 1.3249 | 0.283 | 0.621 | 0.298 | 0.206 |
| 23 | 1.0983 | 0.9658 | 1.2925 | 0.168 | 0.713 | 0.281 | 0.216 |
| 24 | 1.0792 | 0.9522 | 1.2720 | 0.178 | 0.686 | 0.231 | 0.158 |
| 25 | 1.0654 | 0.9368 | 1.2768 | 0.184 | 0.779 | 0.236 | 0.167 |
| 26 | 1.0316 | 0.9016 | 1.2587 | 0.185 | 0.724 | 0.274 | 0.198 |
| 27 | 1.0264 | 0.8729 | 1.2514 | 0.186 | 0.731 | 0.256 | 0.189 |

⭐ Best checkpoint — saved as `runs/detect/train-4/weights/best.pt`

---

### Prediction Results

Annotated predictions are saved under `runs/detect/predict/`.
To replace the placeholders below, commit your result images and update the paths.

#### 🥟 Momo

<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/aa519da3-96c8-4530-ae15-12d89e65f8e4" />

---

#### 🌶️ Chatpate

<img width="811" height="639" alt="image" src="https://github.com/user-attachments/assets/9b809995-ae28-4b6d-bbdd-647e95444360" />

---

Each prediction output includes:

- **Bounding boxes** drawn around each detected food item
- **Class label** (e.g., `Momo`, `Chatpate`)
- **Confidence score** (e.g., `0.80`)

---

## Requirements

Full list from `requirements.txt`:

```
ultralytics==8.4.51
opencv-python==4.13.0.92
opencv-python-headless==4.13.0.92
Pillow==12.2.0
typing_extensions==4.15.0
filelock==3.29.0
cryptography==48.0.0
pyOpenSSL==26.2.0
ipython==9.13.0
ipywidgets==8.1.8
Sphinx==9.1.0
redis==5.2.1
```

> Python **3.8+** recommended. A CUDA-enabled GPU is strongly recommended for training; CPU inference is supported.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-food-class`)
3. Commit your changes (`git commit -m 'Add new food class: Dhido'`)
4. Push to the branch (`git push origin feature/new-food-class`)
5. Open a Pull Request

---
