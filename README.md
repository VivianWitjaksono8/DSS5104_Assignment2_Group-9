<div>

# Deep Learning for Image Segmentation
### Scene Understanding on the Oxford-IIIT Pet Dataset

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20Local-yellow?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4%2B-76B900?style=flat-square&logo=nvidia)

</div>

---

**DSS5104 — Machine Learning and Predictive Modelling - Academic Year 2025 - 2026 | Assignment 2**

| Member | Student ID |
|--------|------------|
| Vivian Witjaksono | A0326440M |
| Chaisathid Patanan | A0327119E |
| Raissa Shafira Indra | A0329591U |

**Group 9 · April 2026**

---

### Code Access

To view and run the code for this project, please download the notebook file from the GitHub repository:

**`Assignment2_DSS5104_Group9.ipynb`**

Once downloaded, you can open and execute the notebook using either:

- **Visual Studio Code (VS Code)**  
- **Google Colab**

Make sure all required dependencies are installed before running the notebook.

---

### Notes

1. All code development and experiments were conducted in Google Colab with GPU support. Due to GPU non-determinism, results may vary slightly across runs, but overall model rankings remain consistent.  
2. HTML-rendered results **(`HTMLResult_DSS5104_Assignment2_Group9.html`)** are included in this repository to avoid re-execution of the full computational pipeline and to facilitate efficient review of experimental outputs.

---

## Overview

This repository contains the full experimental pipeline for semantic segmentation on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (~7,300 images, 37 pet breeds). The study benchmarks four segmentation architectures — **UNet, DeepLabV3, FPN, and SegFormer** — with ablation studies on backbone selection, loss function design, and data augmentation strategies. A zero-shot **SAM (Segment Anything Model)** baseline is included for reference.

---

## Results

### Full Model Comparison — Validation Set

All 10 configurations evaluated under identical conditions.

| Model | mIoU | Dice | Pixel Acc | IoU (FG) | IoU (BG) | IoU (Boundary) | Boundary IoU |
|---|---|---|---|---|---|---|---|
| **SegFormer** | **0.8178** | **0.8922** | **0.9362** | **0.9010** | **0.9401** | **0.6122** | 0.8124 |
| DeepLabV3 + Dice | 0.8082 | 0.8853 | 0.9357 | 0.8909 | 0.9426 | 0.5911 | **0.8205** |
| DeepLabV3 | 0.8055 | 0.8839 | 0.9306 | 0.8902 | 0.9346 | 0.5917 | 0.7992 |
| DeepLabV3 + ResNet34 | 0.8039 | 0.8827 | 0.9296 | 0.8887 | 0.9340 | 0.5889 | 0.7930 |
| UNet | 0.8031 | 0.8825 | 0.9285 | 0.8862 | 0.9318 | 0.5912 | 0.7857 |
| DeepLabV3 + DiceFocal | 0.8002 | 0.8802 | 0.9275 | 0.8867 | 0.9305 | 0.5832 | 0.7876 |
| DeepLabV3 + EfficientB3 | 0.8000 | 0.8798 | 0.9280 | 0.8877 | 0.9329 | 0.5794 | 0.7879 |
| DeepLabV3 + CE | 0.7976 | 0.8785 | 0.9263 | 0.8836 | 0.9299 | 0.5792 | 0.7854 |
| FPN | 0.7927 | 0.8754 | 0.9243 | 0.8732 | 0.9304 | 0.5746 | 0.7779 |
| DeepLabV3 + Focal | 0.7867 | 0.8709 | 0.9208 | 0.8761 | 0.9224 | 0.5615 | 0.7708 |

### Final Evaluation — Test Set (4 Baseline Models)

| Model | mIoU | Dice | Pixel Acc | IoU (FG) | IoU (BG) | IoU (Boundary) |
|---|---|---|---|---|---|---|
| **SegFormer** | **0.8293** | **0.9009** | **0.9389** | **0.8946** | **0.9461** | **0.6473** |
| DeepLabV3 | 0.8126 | 0.8897 | 0.9310 | 0.8820 | 0.9373 | 0.6185 |
| UNet | 0.8125 | 0.8900 | 0.9300 | 0.8793 | 0.9356 | 0.6225 |
| FPN | 0.8038 | 0.8842 | 0.9264 | 0.8676 | 0.9345 | 0.6094 |

Test results are consistent with validation — no significant overfitting observed.

### Efficiency Benchmark

| Model | Params (M) | FPS | mIoU (Val) |
|---|---|---|---|
| FPN | 22.99 | 135.44 | 0.7927 |
| UNet | 24.44 | 131.68 | 0.8031 |
| DeepLabV3 | 26.01 | 61.96 | 0.8055 |
| DeepLabV3 + Dice | 26.01 | 61.21 | 0.8082 |
| SegFormer | 27.46 | 57.23 | 0.8178 |
| DeepLabV3 + EfficientB3 | 14.47 | 51.37 | 0.8000 |

### SAM Zero-Shot vs SegFormer

| Metric | SegFormer | SAM Zero-Shot | Difference |
|---|---|---|---|
| mIoU | 0.8213 | 0.2194 | +0.6019 |
| IoU (Foreground) | 0.9041 | 0.0848 | +0.8193 |
| IoU (Background) | 0.9411 | 0.5734 | +0.3677 |
| IoU (Boundary) | 0.6187 | 0.0000 | +0.6187 |
| Pixel Accuracy | 0.9377 | 0.5812 | +0.3565 |

> For full interactive result visualizations — plots, qualitative predictions, radar charts — open **`HTMLResult_DSS5104_Assignment2_Group9.html`** directly in a browser.

---

## Key Findings & Discussions

**Architecture.** SegFormer is the best model at mIoU 0.8178 (val) / 0.8293 (test), outperforming all CNN models across every metric. Its advantage is most visible in boundary segmentation, where global self-attention captures long-range spatial context that local CNN receptive fields miss. Among CNNs, **DeepLabV3** performs best (mIoU 0.8055), followed closely by UNet (0.8031) and FPN (0.7927). The gaps between them are small, confirming that CNN architecture choice has limited practical impact on a well-curated dataset of this size. DeepLabV3's slight edge likely comes from its Atrous Spatial Pyramid Pooling (ASPP), which captures multi-scale context more effectively than UNet's skip connections or FPN's feature pyramid alone.

**Backbone.** A heavier backbone does not guarantee better results. ResNet34 (mIoU 0.8039, 61.96 FPS) outperforms EfficientNet-B3 (mIoU 0.8000, 51.37 FPS) despite having more parameters. EfficientNet-B3 offers no practical advantage in either accuracy or speed at this dataset scale, likely requiring more training data or longer training to fully converge.

**Loss Function.** Dice Loss is the most effective choice, achieving mIoU 0.8082 and the highest Boundary IoU of all models (0.8205). It directly optimizes class overlap, making it better suited for the minority boundary class (~13% of pixels) than Cross-Entropy. Focal Loss alone performs worst (mIoU 0.7867), and the Dice+Focal combination (0.8002) does not surpass pure Dice.

**Augmentation.** Simple geometric augmentations work best. Flip Only achieves the highest mIoU (0.8038), followed by Rotate Only (0.8022) — both outperform the no-augmentation baseline (0.7979) and the full pipeline (0.8009). Color and dropout-based augmentations showed the least benefit, suggesting pose variation matters more than appearance variation for this dataset.

**Efficiency.** FPN (135 FPS) and UNet (131 FPS) are the best choices for real-time deployment. SegFormer provides the highest accuracy at a higher compute cost (57 FPS). DeepLabV3 + Dice offers the best CNN accuracy-to-speed ratio at 61 FPS.

**Failure Cases.** Qualitative analysis was conducted on DeepLabV3 + DiceFocal as the representative CNN model. All models consistently struggle with three patterns: (1) low-contrast scenes where the pet blends into the background, (2) unusual or non-upright poses underrepresented in training data, and (3) fine structures such as tails or legs that span only a few pixels at 256×256 resolution. These are largely data-driven limitations rather than architecture-specific issues.

**SAM Zero-Shot.** SAM (mIoU 0.2194) falls far below any trained model and scores zero Boundary IoU — it has no concept of the semantic boundary class defined in this dataset. Fine-tuning remains necessary for structured segmentation tasks.

---

## Project Structure

```
DSS5104-Assignment2-Group9/
│
├── Assignment2_DSS5104_Group9.ipynb                         # Main experiment notebook
├── Assignment2_DSS5104_Group9_Documentation.pdf             # Full written report
├── HTMLResult_DSS5104_Assignment2_Group9.html               # Full results viewer (open in browser)
├── requirements_install.txt                                 # All pip dependencies
├── requirements_full with notes and hyperparameters.txt     # All pip dependencies with notes and hyperparameters used
└── README.md                                                        
```

> The following directories are auto-generated at runtime and are not committed to the repository:
> ```
> checkpoints/          Saved model weights (.pth)
> results/              Evaluation metrics and plots
> sam_checkpoints/      SAM ViT-H weights (~2.5 GB)
> ```

All datasets are downloaded automatically via `torchvision.datasets.OxfordIIITPet`. No manual data download required.

---

## Getting Started

### Prerequisites

- Python **3.10 or 3.11** — Python 3.14 is not supported
- CUDA-capable GPU (NVIDIA T4 or better recommended)
- pip 22+

### 1. Clone the Repository

```bash
git clone https://github.com/VivianWitjaksono8/DSS5104_Assignment2_Group-9.git
cd DSS5104_Assignment2_Group-9
```

### 2. (Optional) Create a Virtual Environment

```bash
# venv
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# or conda
conda create -n segmentation python=3.10 -y
conda activate segmentation
```

### 3. Install PyTorch

Choose the command that matches your hardware:

```bash
# GPU — CUDA 11.8
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

# GPU — CUDA 12.1
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements_install.txt
```

### 5. Install SAM

SAM is not available on PyPI and must be installed directly from GitHub:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 6. Download SAM Checkpoint (~2.5 GB)

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Place the `.pth` file in the root directory, or update the checkpoint path in the SAM section of the notebook.

### 7. Enable GPU on Google Colab

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4 or higher) → Save
```

> Google Colab users: the first cell of the notebook already contains all install commands. Run that cell and skip steps 3–6.

### 8. Run the Notebook

**Google Colab (recommended):** Upload `Assignment2_DSS5104_Group9.ipynb`, enable GPU, then run all cells:

```
Runtime → Run all   (Ctrl + F9)
```

**Local Jupyter:**

```bash
jupyter notebook Assignment2_DSS5104_Group9.ipynb
# or
jupyter lab Assignment2_DSS5104_Group9.ipynb
```

Run all cells top to bottom.

---

## Dependencies

All dependencies are listed in `requirements_install.txt`. Key packages:

| Package | Version | Purpose |
|---|---|---|
| `torch` | >=2.0.0 | Core deep learning framework |
| `torchvision` | >=0.15.0 | Dataset loading, pretrained backbones |
| `segmentation-models-pytorch` | >=0.3.3 | UNet, DeepLabV3, FPN |
| `transformers` | >=4.35.0 | SegFormer (nvidia/mit-b2) |
| `albumentations` | >=1.3.0 | Data augmentation pipeline |
| `segment-anything` | GitHub | SAM zero-shot baseline |
| `fvcore` | >=0.1.5 | FLOPs and parameter counting |
| `scipy` | >=1.10.0 | Boundary IoU (morphological operations) |
| `scikit-learn` | >=1.3.0 | Train/val split |
| `seaborn` | >=0.12.0 | Visualization |

---

## Hyperparameters

### Dataset

| Parameter | Value |
|---|---|
| Dataset | Oxford-IIIT Pet (torchvision) |
| Train / Val / Test | 2,944 / 736 / 3,669 |
| Input resolution | 256 × 256 |
| Num classes | 3 (Foreground, Background, Boundary) |
| Normalization mean | (0.485, 0.456, 0.406) |
| Normalization std | (0.229, 0.224, 0.225) |
| Batch size | 16 |
| Num workers | 2 |
| Random seed | 42 |

### Training

| Parameter | Main | Backbone Ablation | Loss/Aug Ablation |
|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| LR scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| Max epochs | 30 | 20 | 10 |
| Early stopping patience | 7 | 5 | 5 |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Mixed precision | torch.cuda.amp | torch.cuda.amp | torch.cuda.amp |

### Loss Functions

| Loss | Key Parameters |
|---|---|
| CrossEntropy | class-frequency weighted |
| Dice | smooth = 1e-6 |
| Focal | gamma = 2.0 |
| DiceFocal | alpha = 0.5 (equal Dice + Focal weighting) |

### Models

| Model | Configuration |
|---|---|
| UNet | encoder=resnet34, weights=imagenet |
| DeepLabV3 | encoder=resnet34, weights=imagenet |
| FPN | encoder=resnet34, weights=imagenet |
| SegFormer | nvidia/mit-b2, image_size=256, num_classes=3 |
| SAM (zero-shot) | sam_vit_h_4b8939.pth, SamAutomaticMaskGenerator |

### Augmentation Pipeline

| Transform | Key Parameters | Probability |
|---|---|---|
| Horizontal Flip | — | 0.50 |
| Vertical Flip | — | 0.20 |
| Rotation | limit = ±30° | 0.50 |
| Random Resized Crop | 256×256, scale 0.8–1.0 | 0.50 |
| Elastic Transform | alpha=120, sigma=6 | 0.30 |
| Colour Jitter | default | 0.30 |
| Random Brightness/Contrast | default | 0.30 |
| Gaussian Blur | default | 0.20 |
| Coarse Dropout | max_holes=8, max_size=32 | 0.30 |

---

## Deployment Recommendations

| Scenario | Recommended Model | Rationale |
|---|---|---|
| Real-time / high throughput | FPN | 135 FPS, lightest architecture |
| Speed–accuracy balance | UNet | 131 FPS, mIoU 0.8031 |
| Best CNN performance | DeepLabV3 + Dice | 61 FPS, mIoU 0.8082 |
| Highest accuracy | SegFormer | 57 FPS, mIoU 0.8178 val / 0.8293 test |

---

## Notebook Sections

| # | Section |
|---|---|
| 1 | Install Packages |
| 2 | Library Dependencies |
| 3 | Data Import |
| 4 | Data Cleansing and Preparation |
| 5 | Augmentation Pipeline |
| 6 | Model Architecture Definitions |
| 7 | Training Pipeline (early stopping, cosine LR, mixed precision) |
| 8 | Baseline Training — UNet, DeepLabV3, FPN, SegFormer |
| 9 | Backbone Ablation — ResNet34 vs EfficientNet-B3 |
| 10 | Loss Function Ablation — CE, Dice, Focal, DiceFocal |
| 11 | Augmentation Ablation — 6 strategies |
| 12 | Full Model Comparison — all metrics |
| 13 | Efficiency Benchmark — Params, FLOPs, FPS |
| 14 | Accuracy vs Speed Tradeoff |
| 15 | Radar Chart — multi-metric comparison |
| 16 | Final Test Set Evaluation |
| 17 | SAM Zero-Shot Comparison |
| 18 | Qualitative Analysis — good predictions and failure cases |
| 19 | Key Question Analysis |
| 20 | Conclusion |

---

## Troubleshooting

**CUDA out of memory:** Reduce batch size from 16 to 8 in the training config cell.

**SAM checkpoint not found:** Ensure `sam_vit_h_4b8939.pth` is placed in the root directory. The SAM section cell also includes a download command.

**`albumentations` import errors:** This project uses standard `albumentations>=1.3.0`. If you previously installed `albumentationsx`, uninstall it first: `pip uninstall albumentationsx && pip install albumentations`.

**`from google.colab import drive` error when running locally:** Remove or comment out any Google Drive mount cells — they are only needed on Colab.

**Metrics vary between runs:** Expected behavior due to GPU non-determinism. Relative model rankings are stable; absolute values may vary by ±0.005 mIoU.

---

## References

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — Parkhi et al., 2012
- [UNet](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [DeepLabV3](https://arxiv.org/abs/1706.05587) — Chen et al., 2017
- [FPN](https://arxiv.org/abs/1612.03144) — Lin et al., 2017
- [SegFormer](https://arxiv.org/abs/2105.15203) — Xie et al., 2021
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) — Kirillov et al., 2023
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://github.com/albumentations-team/albumentations)

---

<div align="center">
  <sub>DSS5104 · Group 9 · National University of Singapore · April 2026</sub>
</div>
