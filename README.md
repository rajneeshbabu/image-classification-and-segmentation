# Image Classification and Segmentation
## PCA · Multi-Label Classification · Semantic Segmentation · Statistical Testing

[![GitHub Pages](https://img.shields.io/badge/🌐_Project_Page-GitHub_Pages-6366f1?style=for-the-badge)](https://rajneeshbabu.github.io/image-classification-and-segmentation/)
[![IISc](https://img.shields.io/badge/IISc-MLDS_2026-003580?style=for-the-badge)](https://cds.iisc.ac.in)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)

**Course:** Machine Learning for Data Science (MLDS) · IISc Bengaluru · 2026  
**Student:** Rajneesh Babu · SR No. 26058  
**Project Page:** [rajneeshbabu.github.io/image-classification-and-segmentation](https://rajneeshbabu.github.io/image-classification-and-segmentation/)

---

## Overview

A three-part assignment covering the full pipeline of a computer vision project — from classical dimensionality reduction to deep learning-based multi-label classification and semantic segmentation, validated with rigorous non-parametric statistical testing.

| Part | Topic | Key Result |
|------|-------|-----------|
| **1** | PCA from scratch (NumPy) | ~100 components capture >90% variance |
| **2A** | Multi-label classification — ResNet-50 | 20-class VOC, BCELoss + sigmoid |
| **2B** | Semantic segmentation — ResNet-50 + ASPP | Model A mIoU = **0.7872** |
| **3** | Wilcoxon signed-rank + Bootstrap CI | p = 6.19e-34, CI = [0.7642, 0.8068] |

---

## Part 1 — PCA from Scratch

NumPy-only implementation of Principal Component Analysis — no `sklearn.decomposition.PCA` used.

- **Covariance matrix** computed, then **eigendecomposition** via `np.linalg.eigh`
- **Variance explained** curve shows clear elbow near k = 100 (~90% variance)
- **Reconstruction** at k = 25, 50, 100, 200 — visual quality vs. MSE trade-off
- k = 100–200 is visually near-identical to the original; high-frequency textures are the first lost

---

## Part 2A — Multi-Label Classification

**Dataset:** PASCAL VOC — 20 object classes  
**Architecture:** ResNet-50 (ImageNet pretrained) + custom classification head

```
ResNet-50 backbone → GlobalAvgPool → [2048]
  → Dropout(0.5) → Linear(2048→512) → BN → ReLU
  → Dropout(0.3) → Linear(512→20) → Sigmoid
```

**Training details:**
- Loss: `BCELoss` (binary cross-entropy per class)
- Discriminative LRs: backbone `5e-5`, head `1e-3`
- Scheduler: cosine annealing
- Augmentation: RandomHorizontalFlip, ColorJitter (torchvision.transforms)
- Output: per-class probabilities in [0, 1] for all 20 VOC classes

**Classes:** aeroplane · bicycle · bird · boat · bottle · bus · car · cat · chair · cow · diningtable · dog · horse · motorbike · person · pottedplant · sheep · sofa · train · tvmonitor

---

## Part 2B — Semantic Segmentation

Two models trained on the same dataset for Part 3 statistical comparison.

### Model A — ResNet-50 + ASPP + U-Net Decoder *(Primary)*

```
ResNet-50 encoder (ImageNet pretrained)
  → ASPP (Atrous Spatial Pyramid Pooling, rates=[1,6,12,18])
  → U-Net decoder: UpBlock×4 (skip connections from encoder stages)
  → Final conv → 21-class logits
  + Auxiliary head from layer3 (CE + Dice loss)
```

- Input size: 384×384 (letterbox padding)
- Inference: multi-scale (384, 480) + horizontal flip TTA (4 forward passes)
- **Val mIoU: 0.7872**

### Model B — ResNet-18 + Simple U-Net *(Comparison baseline)*

- Lighter encoder (ResNet-18), no ASPP, no auxiliary head
- **Val mIoU: 0.6848**

**Loss:** Cross-Entropy + Dice loss  
**Augmentation:** albumentations (RandomHorizontalFlip, ColorJitter, RandomCrop)  
**Classes:** 21 (20 VOC objects + background), boundary label 255 ignored

---

## Part 3 — Statistical Testing

### Why Wilcoxon, not t-test?

IoU scores are bounded to [0,1] and right-skewed. The Shapiro–Wilk test confirms the paired differences are **non-normal (p = 2.895e-13)**, violating the t-test assumption. The Wilcoxon signed-rank test is non-parametric and robust to outliers.

### Results

| Test | Result |
|------|--------|
| Paired samples (n) | 330 |
| Model A mean mIoU | **0.7872** |
| Model B mean mIoU | 0.6848 |
| Shapiro–Wilk (differences) | W = 0.9084, p = 2.895e-13 |
| **Wilcoxon signed-rank** | **W = 6106.0, p = 6.1936e-34** |
| Paired t-test (contrast) | t = 12.47, p = 1.65e-29 |
| Best model Bootstrap CI (95%) | **[0.7642, 0.8068]** |

**Conclusion:** p << 0.05 — we reject H₀ and conclude Models A and B differ **significantly** in per-image mIoU. Model A is conclusively better.

---

## Repository Structure

```
26058_Rajneesh_Babu_Assignment_2/
├── training_notebook.ipynb          # Part 1 (PCA) + Part 2A (clf) + Part 2B (seg)
├── statistical_tests.ipynb          # Part 3 — Wilcoxon + Bootstrap
├── submission.csv                   # RLE-encoded segmentation predictions
├── report.pdf                       # Full assignment report
├── requirements.txt
├── README.md
├── index.html                       # GitHub Pages project page
├── classification/
│   ├── model.py                     # ClassificationModel wrapper (evaluator-ready)
│   └── weights/
│       └── classifier.pth           # Trained ResNet-50 weights
└── segmentation/
    ├── model.py                     # SegmentationModel wrapper (evaluator-ready)
    └── weights/
        ├── segmenter.pth            # Model A — ResNet-50 + ASPP
        └── segmenter_B.pth          # Model B — ResNet-18 U-Net
```

---

## Installation

```bash
git clone https://github.com/rajneeshbabu/image-classification-and-segmentation.git
cd image-classification-and-segmentation
pip install -r requirements.txt
```

---

## Running the Notebooks

### Training Notebook (Parts 1, 2A, 2B)

```bash
jupyter notebook training_notebook.ipynb
```

Run all cells **top to bottom**. The notebook will:
1. Perform PCA from scratch and plot variance explained + reconstructions
2. Train the ResNet-50 multi-label classifier (Part 2A)
3. Train Model A (ResNet-50 + ASPP) and Model B (ResNet-18 U-Net) segmenters (Part 2B)
4. Save weights to `classification/weights/` and `segmentation/weights/`
5. Generate `submission.csv` with RLE-encoded segmentation predictions

> **GPU recommended.** Training runs on CPU if no GPU is available but will be significantly slower.

### Statistical Tests Notebook (Part 3)

```bash
jupyter notebook statistical_tests.ipynb
```

> **Prerequisite:** Run `training_notebook.ipynb` first so both `.pth` weight files exist.

Loads both pre-trained segmentation models, evaluates on the 15% labeled hold-out split (seed 42), then runs:
- **Wilcoxon signed-rank test** on paired per-image mIoU
- **Bootstrap 95% CI** (B = 1000, percentile method) on the best model

---

## Using the Evaluator Wrappers

Both `model.py` files expose a simple `predict(image: np.ndarray)` interface compatible with the course evaluator.

```python
import numpy as np
from classification.model import ClassificationModel
from segmentation.model import SegmentationModel

# Classification — returns dict of {class_name: probability}
clf = ClassificationModel(weights_dir="classification/weights")
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
probs = clf.predict(image)
print(probs)  # {'aeroplane': 0.03, 'person': 0.91, ...}

# Segmentation — returns H×W uint8 label map (0–20)
seg = SegmentationModel(weights_dir="segmentation/weights")
label_map = seg.predict(image)
print(label_map.shape)  # (480, 640)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Model training and inference |
| `torchvision >= 0.15` | ResNet backbones, transforms |
| `numpy >= 1.23` | PCA, array ops |
| `pandas >= 1.5` | Submission CSV |
| `Pillow >= 9.0` | Image loading |
| `scikit-image >= 0.20` | Image processing |
| `scikit-learn >= 1.2` | Metrics, train/val split |
| `scipy >= 1.10` | Wilcoxon, Shapiro-Wilk tests |
| `albumentations >= 1.4` | Segmentation augmentation |
| `matplotlib >= 3.6` | PCA plots, visualisations |
| `tqdm >= 4.65` | Training progress bars |

---

*© 2026 Rajneesh Babu · SR 26058 · M.Tech CDS · IISc Bengaluru*
