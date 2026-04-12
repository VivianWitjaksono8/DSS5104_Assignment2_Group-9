<div align="center">

# Deep Learning for Image Segmentation
### Scene Understanding on the Oxford-IIIT Pet Dataset

<br/>

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20Local-yellow?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4%2B-76B900?style=flat-square&logo=nvidia)

<br/>

**DSS5104 — Machine Learning and Predictive Modelling | Assignment 2**

| Member | Student ID |
|--------|------------|
| Vivian Witjaksono | A0326440M |
| Chaisathid Patanan | A0327119E |
| Raissa Shafira Indra | A0329591U |

**Group 9 · April 2026**

</div>

---

## 📌 Overview

This repository contains the full experimental pipeline for **semantic segmentation** on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (~7,300 images, 37 pet breeds). The study systematically benchmarks four segmentation architectures — **UNet, DeepLabV3, FPN, and SegFormer** — alongside ablation studies on backbone selection, loss functions, and data augmentation strategies, with a zero-shot baseline from **SAM (Segment Anything Model)**.

### Key Results at a Glance

| Model | mIoU (Val) | mIoU (Test) | FPS |
|---|---|---|---|
| **SegFormer** ⭐ | **0.8178** | **0.8293** | 57.23 |
| DeepLabV3 + Dice | 0.8082 | — | 61.21 |
| UNet | 0.8031 | — | 131.68 |
| FPN | 0.7927 | — | **135.44** |
| SAM Zero-Shot (baseline) | 0.2194 | — | — |

> **Reproducibility Note:** All experiments were run on NVIDIA T4 GPUs via Google Colab with mixed precision training. Due to GPU non-determinism, metric values may vary slightly between runs, but relative model rankings remain consistent.

---

## Project Structure

```
DSS5104-Assignment2-Group9/
│
├──  Assignment2_DSS5104_Group9.ipynb          # Main experiment notebook
├──  Assignment2_DSS5104_Group9_Documentation.pdf  # Full PDF report
├──  requirements_install.txt                  # All pip dependencies
├── 📖 README.md                                 # This file
│
└── 📂 (auto-generated at runtime)
    ├── checkpoints/                             # Saved model weights (.pth)
    ├── results/                                 # Evaluation metrics & plots
    └── sam_checkpoints/                         # SAM ViT-B weights
```

> All datasets are downloaded automatically via `torchvision.datasets.OxfordIIITPet` at runtime. No manual data download is required.

---

##  Environment Setup

### Prerequisites

- Python **3.9+**
- CUDA-capable GPU (NVIDIA T4 or better recommended)
- pip **22+**

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/VivianWitjaksono8/DSS5104_Assignment2_Group9.git
cd DSS5104_Assignment2_Group9
```

### 2. (Optional) Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# Or using conda
conda create -n segmentation python=3.9 -y
conda activate segmentation
```

### 3. Install Dependencies

```bash
pip install -r requirements_install.txt
```

> 💡 **Google Colab users:** The first cell of the notebook (`# Install Packages`) already contains all the installation commands. Simply run it and you're good to go — no separate install step needed.

### 4. Enable GPU (Google Colab)

Before running the notebook on Colab:

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4 or higher) → Save
```

### 5. Run the Notebook

**Option A — Google Colab (Recommended)**

Upload `Assignment2_DSS5104_Group9.ipynb` to Google Colab, enable GPU (step 4), then run all cells:

```
Runtime → Run all  (Ctrl + F9)
```

**Option B — Local Jupyter**

```bash
jupyter notebook Assignment2_DSS5104_Group9.ipynb
# or
jupyter lab Assignment2_DSS5104_Group9.ipynb
```

Then run all cells from top to bottom (`Cell → Run All`).

---

## 📦 Dependencies

All dependencies are listed in `requirements_install.txt`. Key packages:

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.0 | Core deep learning framework |
| `torchvision` | ≥0.15 | Dataset loading, pretrained models |
| `segmentation-models-pytorch` | latest | UNet, DeepLabV3, FPN architectures |
| `transformers` | latest | SegFormer (HuggingFace) |
| `albumentationsx` | latest | Data augmentation pipeline |
| `segment-anything` | latest | SAM zero-shot baseline |
| `fvcore` | latest | FLOPs / parameter counting |
| `seaborn` | latest | Visualization |

Install all at once:

```bash
pip install -r requirements_install.txt
```

---

## 🧪 Experiments & Hyperparameters

### Dataset Configuration

| Parameter | Value |
|---|---|
| Dataset | Oxford-IIIT Pet (torchvision) |
| Total images | 7,349 |
| Train / Val / Test split | 2,944 / 736 / 3,669 |
| Input resolution | 256 × 256 px |
| Normalization mean | (0.485, 0.456, 0.406) |
| Normalization std | (0.229, 0.224, 0.225) |
| Random seed | 42 |
| Segmentation classes | 3 (Foreground, Background, Boundary) |

### Training Configuration (All Models)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Max epochs | 30 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| LR scheduler | Cosine Annealing |
| Gradient clipping | 1.0 |
| Mixed precision | ✅ (torch.cuda.amp) |
| Early stopping patience | 7 epochs |
| Checkpoint criterion | Best Val mIoU |

### Model Architectures

| Model | Backbone | Pretrained | Params (M) | FLOPs (G) |
|---|---|---|---|---|
| UNet | ResNet34 | ImageNet | 24.44 | 27.33 |
| DeepLabV3 | ResNet34 | ImageNet | 26.01 | 16.34 |
| FPN | ResNet34 | ImageNet | 22.99 | 6.88 |
| SegFormer | MiT-B2 | ImageNet | 27.46 | 8.39 |

### Ablation Studies

**Backbone Ablation (UNet):**

| Backbone | mIoU | FPS |
|---|---|---|
| ResNet34 | 0.8039 | 61.96 |
| EfficientNet-B3 | 0.8000 | 51.37 |

**Loss Function Ablation (DeepLabV3 + ResNet34):**

| Loss Function | mIoU | Boundary IoU |
|---|---|---|
| **Dice** ⭐ | **0.8082** | **0.8205** |
| DiceFocal | 0.8002 | — |
| CrossEntropy | 0.7976 | 0.7854 |
| Focal | 0.7867 | — |

**Augmentation Ablation (DeepLabV3 + ResNet34 + Dice):**

| Strategy | mIoU |
|---|---|
| **Flip Only** ⭐ | **0.8038** |
| Rotate Only | 0.8022 |
| Full Pipeline | 0.8009 |
| No Augmentation | 0.7979 |
| Dropout Only | 0.7975 |
| Colour Only | 0.7969 |

### Augmentation Pipeline (Full)

| Category | Transform | Parameters | Probability |
|---|---|---|---|
| Geometric | Horizontal Flip | — | p=0.50 |
| Geometric | Vertical Flip | — | p=0.20 |
| Geometric | Rotation | limit=±30° | p=0.50 |
| Geometric | Random Resized Crop | 256×256, scale 0.8–1.0 | p=0.50 |
| Geometric | Elastic Transform | alpha=120, sigma=6 | p=0.30 |
| Photometric | Colour Jitter | default | p=0.30 |
| Photometric | Random Brightness/Contrast | default | p=0.30 |
| Blur | Gaussian Blur | default | p=0.20 |
| Occlusion | Coarse Dropout | max_holes=8, max_size=32 | p=0.30 |

---

## 📊 Notebook Sections

The notebook is organized into the following sections, which can be run sequentially:

| # | Section | Description |
|---|---|---|
| 1 | Install Packages | pip installs for all libraries |
| 2 | Library Dependencies | All imports |
| 3 | Data Import | Download Oxford-IIIT Pet via torchvision |
| 4 | Data Cleansing & Preparation | Validation, label remapping, preprocessing |
| 5 | Augmentation Pipeline | AlbumentationsX pipeline definition |
| 6 | Model Architecture | UNet, DeepLabV3, FPN, SegFormer definitions |
| 7 | Training Pipeline | `train_model()` with early stopping, cosine LR, GradScaler |
| 8 | Baseline Training & Results | Train all 4 architectures, report val mIoU |
| 9 | Backbone Ablation | ResNet34 vs EfficientNet-B3 on UNet |
| 10 | Loss Function Ablation | CE vs Dice vs Focal vs DiceFocal on DeepLabV3 |
| 11 | Augmentation Ablation | 6 augmentation strategies on DeepLabV3+Dice |
| 12 | Full Model Comparison | mIoU, Dice, Pixel Acc, per-class IoU summary |
| 13 | Efficiency Benchmark | Params, FLOPs, FPS for all models |
| 14 | Accuracy vs Speed Tradeoff | Scatter plot analysis |
| 15 | Radar Chart | Multi-metric comparison visualization |
| 16 | Test Set Evaluation | Final evaluation on held-out test set |
| 17 | SAM Zero-Shot Comparison | SAM ViT-B vs SegFormer |
| 18 | Qualitative Analysis | Good predictions & failure case visualization |
| 19 | Key Question Analysis | Answers to 7 analytical questions |
| 20 | Conclusion | Summary and deployment recommendations |

---

## 🔑 Deployment Recommendations

Based on experimental results:

| Scenario | Recommended Model | Rationale |
|---|---|---|
| Real-time / High Throughput | **FPN** (135 FPS) | Fastest inference |
| Speed–Accuracy Balance | **UNet** (131 FPS, mIoU 0.8031) | Best CNN trade-off |
| Best CNN Performance | **DeepLabV3 + Dice** (61 FPS, mIoU 0.8082) | Top CNN accuracy |
| Highest Accuracy | **SegFormer** (57 FPS, mIoU 0.8178/0.8293) | Best overall |

---

## ⚠️ Common Issues & Troubleshooting

**CUDA out of memory:**
> Reduce batch size in the training config cell (`batch_size = 8` instead of 16).

**SAM checkpoint not found:**
> The SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`) is downloaded automatically in the SAM section cell. Ensure you have a stable internet connection when running that cell.

**`albumentationsx` not found:**
> Run `pip install albumentationsx` — this is the updated fork of Albumentations used in this project.

**Metrics vary between runs:**
> This is expected behavior due to GPU non-determinism. Relative rankings between models are stable across runs; absolute values may differ by ±0.005 mIoU.

---

## 📎 References

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — Parkhi et al., 2012
- [UNet](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [DeepLabV3](https://arxiv.org/abs/1706.05587) — Chen et al., 2017
- [Feature Pyramid Network (FPN)](https://arxiv.org/abs/1612.03144) — Lin et al., 2017
- [SegFormer](https://arxiv.org/abs/2105.15203) — Xie et al., 2021
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) — Kirillov et al., 2023
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [AlbumentationsX](https://github.com/albumentations-team/albumentations)

---

<div align="center">
  <sub>DSS5104 · Group 9 · National University of Singapore · April 2026</sub>
</div>
