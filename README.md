# Vehicle Perception: Neurosymbolic Multi-Object Tracking

[![Papers with Code](https://img.shields.io/badge/Paper-PMLR2025-blue)](https://github.com/azzy13/vehicle_perception)

## Overview

This repository presents a **neurosymbolic approach** to **multi-object tracking (MOT)** in **dynamic camera environments**. Building on **ByteTrack**, this project integrates **neural detection models (YOLOX)** with **symbolic motion models (Kalman Filter & Constant Turn Rate and Acceleration - CTRA)** to enhance vehicle tracking in real-world settings such as autonomous driving and drone surveillance.

## Key Features

- **Enhanced Motion Modeling**: Incorporates **CTRA** to account for **non-linear** object motion.
- **Dual Association Strategy**: First matches **high-confidence detections**, then **low-confidence detections** to **recover lost tracks**.
- **Multi-Dataset Integration**: Training data sourced from **KITTI, BDD100k, COCO, and UA-DETRAC**.
- **Optimized Tracking Parameters**: Bayesian **hyperparameter optimization** with **Optuna**.
- **Support for Dynamic Cameras**: Robust to **perspective shifts, motion blur, and frame rate variations**.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/azzy13/vehicle_perception.git
   cd vehicle_perception
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Setup dataset directories (see below for dataset structure).

## Dataset Preparation

Ensure the following dataset structure before training:

```
datasets
   ├── mot
   |    ├── train
   |    ├── test
   ├── crowdhuman
   |    ├── train
   |    ├── val
   ├── KITTI
   |    ├── images
   |    ├── labels
   ├── UA-DETRAC
   |    ├── images
   |    ├── annotations
```

Convert datasets to COCO format using:

```bash
python tools/convert_kitti_to_coco.py
python tools/convert_crowdhuman_to_coco.py
python tools/convert_detrac_to_coco.py
```

## Training

Train the model using the following command:

```bash
python tools/train.py -f exps/example/mot/carla_drone.py -c pretrained/ground.pth.tar -d 8 -b 48 --fp16 -o
```

- `-f` specifies the experiment file.
- `-d` sets the number of GPUs.
- `-b` sets the batch size.
- `--fp16` enables mixed precision training.

## Tracking

Run tracking on test videos:
python3 tools/demo_track.py video 

```bash
python tools/track.py -f exps/example/mot/carla_drone.py -c pretrained/ground.pth.tar -b 1 -d 1 --fp16 --fuse
```

For visualization, use:

```bash
python tools/demo_track.py video -f exps/example/mot/carla_drone.py -c pretrained/ground.pth.tar --fp16 --fuse --save_result
```

## Results

Tracking evaluated on KITTI (10 FPS, dynamic camera conditions):

| Model                | MOTA  | IDF1  | IDs | FPS  |
| -------------------- | ----- | ----- | --- | ---- |
| ByteTrack (Baseline) | 59.1% | 74.1% | 355 | 29.6 |
| ByteTrack + CTRA     | 57.8% | 74.0% | 405 | 29.1 |

**Findings:** Despite **CTRA's improved motion modeling**, results indicate a **slight degradation** in tracking performance, warranting further investigation.

## Citation

If you find this work useful, please cite:

```
@article{hasan2025mot,
  title={Integrating Neural and Symbolic Methods for Multi-Object Tracking in Dynamic Camera Environments},
  author={Hasan, Azhar and Richardson, Alex and Karsai, Gabor},
  booktitle={Proceedings of Machine Learning Research},
  year={2025}
}
```

## Contact

For questions or contributions, reach out to **Azhar Hasan** at **azhar.hasan@vanderbilt.edu** or open an **issue** on this repository.

---

_Acknowledgment: Built on top of [ByteTrack](https://github.com/ifzhang/ByteTrack)._
