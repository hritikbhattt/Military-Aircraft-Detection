# Military Aircraft Detection (YOLOv8)

![CI](https://github.com/hritikbhattt/Military-Aircraft-Detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Real-time military aircraft detection and classification across **36 fighter jets, bombers, transports, and reconnaissance aircraft**, built on YOLOv8. Supports image, video, and webcam inference with a modular, testable pipeline: dataset preparation → training → evaluation → inference.

![Inference Demo](assets/Screenshot-inference.png)

## Contents
- [Overview](#overview)
- [Supported Aircraft](#supported-aircraft)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Testing](#testing)
- [Results](#results)
- [Project Structure](#project-structure)
- [Design Notes](#design-notes)

## Overview
This project trains a YOLOv8 object detector to localize and classify military aircraft in still images and video. It covers the full applied-ML lifecycle rather than just a model file:

- **Data preparation**: converts a public VOC-style annotated dataset into YOLO format, with a class-normalization layer that maps raw dataset label variants (e.g. `f16`, `F_16`) onto a canonical 36-class taxonomy.
- **Training**: configurable YOLOv8 fine-tuning (model size, epochs, image size, early stopping) via `scripts/train.py`.
- **Evaluation**: both the official Ultralytics `val()` metrics and an independent, from-scratch per-class AP/precision/recall report (`data/metrics.py`) computed with plain numpy IoU matching — useful as a sanity check that the reported mAP isn't a black box.
- **Inference**: unified `detect.py` CLI for images, video files, and a live webcam feed.
- **Testing & CI**: unit tests for the box-conversion math and the AP/IoU implementation, plus a GitHub Actions workflow that trains one epoch on a generated synthetic dataset to catch pipeline breakage before you spend GPU time on the real one.

## Supported Aircraft
`F-22` `F-35` `F-16` `F-15` `F-18` `F-14` `F-4` `B-2` `B-1` `B-52` `F-117` `SR-71` `A-10` `C-130` `C-17` `C-5` `U-2` `YF-23` `XB-70` `Su-57` `MiG-31` `Tu-95` `Tu-160` `J-20` `Rafale` `EF2000` `JAS-39` `Mirage-2000` `V-22` `MQ-9` `RQ-4` `E-2` `AG600` `Be200` `US-2` `A400M`

## Pipeline
```
Kaggle dataset (VOC CSVs)
        │  data/prepare_dataset.py  (label normalization + VOC→YOLO box conversion)
        ▼
data/images/{train,val,test}
data/labels/{train,val,test}
        │  scripts/train.py  (YOLOv8 fine-tuning)
        ▼
   models/best.pt
        │
        ├── scripts/evaluate.py  → per-class AP, mAP, confusion matrix
        └── scripts/detect.py    → image / video / webcam inference
```

## Installation
```bash
git clone https://github.com/hritikbhattt/Military-Aircraft-Detection.git
cd Military-Aircraft-Detection
pip install -r requirements.txt
```
Python 3.8+, PyTorch 2.0+, and a CUDA-capable GPU are recommended for training (CPU works for inference and for the synthetic smoke test below).

## Dataset
Trained on the [Military Aircraft Detection Dataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset) (Kaggle), which ships one CSV per image in VOC format (`filename, width, height, class, xmin, ymin, xmax, ymax`). `data/prepare_dataset.py`:
1. Downloads the dataset via `kagglehub` (needs a free Kaggle API token at `~/.kaggle/kaggle.json`).
2. Normalizes raw label strings to the canonical 36-class list (`data/classes.py`), dropping aircraft outside that list.
3. Converts VOC pixel boxes to normalized YOLO `(cx, cy, w, h)`.
4. Splits 80/10/10 into train/val/test and writes `data/images/*` + `data/labels/*`.

```bash
python data/prepare_dataset.py                       # full dataset
python data/prepare_dataset.py --limit-per-class 150  # smaller run for faster iteration
```

**No dataset yet, or want to verify the pipeline first?** Generate a small synthetic stand-in dataset (colored shapes with exact, programmatically-known bounding boxes) and confirm the whole thing runs end to end in under a minute:
```bash
python data/generate_synthetic_samples.py --per-class 8
python scripts/train.py --model yolov8n.pt --epochs 1 --batch 8 --imgsz 320
```

## Training
```bash
python scripts/train.py
python scripts/train.py --model yolov8m.pt --epochs 100 --batch 16 --imgsz 640 --patience 20
```
Saves runs under `runs/detect/`, logs standard Ultralytics metrics/plots, and copies the best checkpoint to `models/best.pt`.

## Evaluation
```bash
python scripts/evaluate.py --weights models/best.pt --split val
```
Prints Ultralytics' official mAP50 / mAP50-95 / precision / recall, then re-derives a per-class AP table independently via `data/metrics.py` (own IoU matching + 101-point interpolated AP, no torch dependency) as a cross-check, and saves a per-class AP bar chart to `outputs/eval/per_class_ap.png`.

## Inference
```bash
python scripts/detect.py --source test_files/image.jpg
python scripts/detect.py --source test_files/video.mp4
python scripts/detect.py --source 0                          # webcam
python scripts/detect.py --source test_files/video.mp4 --play
python scripts/detect.py --source test_files/image.jpg --conf 0.5
```

## Testing
```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```
Covers: IoU correctness on known geometries, AP computation against hand-derived PR curves, VOC→YOLO box conversion, and the raw-label→canonical-class normalization (including a completeness check that every one of the 36 classes has at least one raw-label mapping). CI (`.github/workflows/ci.yml`) runs these on every push, then separately trains one epoch on the synthetic dataset as an end-to-end smoke test.

## Results
_Fill in after training on the real dataset — `scripts/evaluate.py` prints/saves everything below._

| Metric | Value |
|---|---|
| mAP50 | TBD |
| mAP50-95 | TBD |
| Precision | TBD |
| Recall | TBD |
| Inference speed | TBD ms/frame (device: TBD) |

## Project Structure
```
military-aircraft-detection/
├── .github/workflows/ci.yml
├── data/
│   ├── classes.py                  # canonical 36-class list + raw-label normalization
│   ├── data.yaml                   # ultralytics dataset config
│   ├── metrics.py                  # pure-numpy IoU / AP / mAP (independent of torch)
│   ├── prepare_dataset.py          # Kaggle download + VOC→YOLO conversion + split
│   └── generate_synthetic_samples.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── detect.py
├── tests/
│   ├── test_classes.py
│   └── test_metrics.py
├── models/                         # best.pt lands here after training
├── test_files/                     # sample images/video for detect.py
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

## Design Notes
- **Why a from-scratch metrics module alongside Ultralytics' own `val()`?** It's a correctness cross-check, and it means the eval report doesn't silently depend on Ultralytics' internal box-matching conventions — the numpy implementation is unit tested (`tests/test_metrics.py`) against hand-computed IoU/AP values.
- **Why a synthetic dataset generator?** Real dataset download + training is a 30-60+ minute GPU commitment; the synthetic generator produces exact, known bounding boxes so path/shape/config bugs in the pipeline surface in seconds, both locally and in CI.
- **Class normalization**: the source dataset's raw labels don't always match a clean display name (`SR71` vs `SR71A12` vs `SR-71`); `data/classes.py` centralizes that mapping so it's the single place to extend if the dataset adds more aircraft.

## Author
Dhruv Gaur
