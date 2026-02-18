# Desert-vision
# Offroad Autonomy Segmentation

A deep learning solution for semantic segmentation of desert terrain, built for the **Duality AI Offroad Autonomy Segmentation Challenge** using synthetic data from the Falcon simulation platform.

---

## 👥 Team Members

- Tanvi Agarwal
- Prisha Choithani
- Ved Kumar Sahu
- Tanay Gupta

---

## 📋 Project Overview

This project trains a U-Net neural network to identify and classify 7 terrain categories in desert images — including rocks, vegetation, sky, and ground — at the pixel level. This type of AI is critical for enabling unmanned ground vehicles (UGVs) to navigate complex off-road environments safely.

| Property | Details |
|----------|---------|
| Task | Semantic Segmentation |
| Architecture | U-Net |
| Dataset | Duality AI Falcon Platform |
| Total Images | 3,174 (2,857 train + 317 val) |
| Classes | 7 terrain categories |
| Metric | Mean IoU (mIoU) |
| Final mIoU | **0.4475** (CPU Model) |

---

## 📁 Project Structure

```
offroad_project/
├── config.py          # All settings and hyperparameters
├── dataset.py         # Data loading and augmentation
├── model.py           # U-Net architecture
├── iou_metrics.py     # IoU accuracy calculation
├── train.py           # Main training script
├── predict.py         # Run predictions on new images
├── README.md          # This file
└── data/
    ├── train/
    │   ├── images/    # 2,857 training photos
    │   └── masks/     # 2,857 segmentation masks
    └── val/
        ├── images/    # 317 validation photos
        └── masks/     # 317 validation masks
```

---

## 📄 File Descriptions

### `config.py`
Central configuration file containing all hyperparameters and settings. Edit this file to change training behaviour without touching any other code.

Key settings:
```python
IMAGE_HEIGHT  = 256      # Input image height
IMAGE_WIDTH   = 256      # Input image width
BATCH_SIZE    = 4        # Images per training batch
EPOCHS        = 50       # Number of training rounds
LEARNING_RATE = 0.001    # How fast the model learns
NUM_CLASSES   = 7        # Number of terrain categories
```

### `dataset.py`
Handles loading images and masks from disk, applying augmentations, and converting non-standard mask values (200, 300, 500, 550, 800, 7100, 10000) to standard class indices (0–6).

Augmentations applied during training:
- Horizontal flip (50% probability)
- Random brightness/contrast (30% probability)
- Gaussian noise (20% probability)

### `model.py`
Implements the U-Net architecture with **31,043,911 trainable parameters**. The encoder progressively extracts features (64→128→256→512 filters), the bottleneck captures deep context (1024 filters), and the decoder reconstructs spatial predictions using skip connections.

### `iou_metrics.py`
Calculates Intersection over Union (IoU) per class and mean IoU (mIoU) — the official evaluation metric for this challenge.

```
IoU  = Intersection / Union
mIoU = average IoU across all 7 classes
```

### `train.py`
Main script that orchestrates the full training pipeline: loads data, builds the model, runs training and validation loops, saves the best model checkpoint, and generates training history graphs.

### `predict.py`
Loads a trained model checkpoint and runs inference on new images, producing colour-coded segmentation maps showing the predicted terrain category for every pixel.

---

## ⚙️ Setup and Installation

### Requirements
- Python 3.9 or 3.10
- VS Code (recommended)

### Step 1: Install Python
Download from [python.org](https://python.org) and check **"Add Python to PATH"** during installation.

### Step 2: Install Libraries
Open VS Code terminal and run:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow matplotlib opencv-python scikit-learn tqdm albumentations
```

### Step 3: Verify Installation
```bash
python -c "import torch; import cv2; import albumentations; print('All libraries ready!')"
```

### Step 4: Prepare Dataset
Place your dataset files in the correct folders:
```
data/train/images/   ← training photos (.png)
data/train/masks/    ← training masks (.png)
data/val/images/     ← validation photos (.png)
data/val/masks/      ← validation masks (.png)
```

---

## 🏋️ Training

Run the training script:
```bash
python train.py
```

You will see output like:
```
Using device: cpu
Found 2857 images
Training images: 2857
Validation images: 317
Building U-Net model...
Model has 31,043,911 parameters!
Epoch  1/50 | Loss: 0.9663 | mIoU: 0.2817
   Best model saved! (mIoU: 0.2817)
Epoch  2/50 | Loss: 0.7230 | mIoU: 0.3162
   Best model saved! (mIoU: 0.3162)
...
Epoch 50/50 | Loss: 0.6200 | mIoU: 0.4271
Training complete!
```

After training:
- `best_model.pth` — your trained model (saved automatically)
- `training_history.png` — loss and mIoU graphs

---

## 🔍 Running Predictions

After training completes:
```bash
python predict.py
```

This will:
1. Load your trained model
2. Run it on a test image
3. Show a side-by-side comparison of original vs predicted
4. Save the result as `prediction_result.png`

---

## 📊 Results

### Model 1 — CPU Baseline

| Epoch | Train Loss | Val mIoU |
|-------|------------|----------|
| 1 | 0.9663 | 0.2817 |
| 2 | 0.7230 | 0.3162 |
| 3 | 0.7225 | 0.3298 |
| 4 | 0.6947 | 0.3441 |
| 5 | 0.6498 | 0.3535 |
| 6 | 0.6700 | 0.3713 |
| 50 | — | **0.4475** |

### Class Segmentation Colours

| Class Index | Mask Value | Colour |
|-------------|------------|--------|
| 0 | 200 | ⬛ Black |
| 1 | 300 | 🟩 Green |
| 2 | 500 | 🩶 Grey |
| 3 | 550 | 🟦 Blue |
| 4 | 800 | 🟧 Orange |
| 5 | 7100 | 🟨 Yellow |
| 6 | 10000 | 🟥 Red |

---
## Screenshots

<img width="1920" height="1200" alt="Screenshot 2026-02-18 155145" src="https://github.com/user-attachments/assets/46c33720-4566-49b1-a405-88f202b66928" />

---
<img width="1919" height="1195" alt="Screenshot 2026-02-18 161832" src="https://github.com/user-attachments/assets/87b2f0e2-fb84-4bd6-bb72-ea24b87e427a" />

---
<img width="1919" height="1199" alt="Screenshot 2026-02-18 204810" src="https://github.com/user-attachments/assets/c2b70cb7-b6ab-4b78-929a-43493eb7b152" />

---
### Training history
<img width="1919" height="704" alt="Screenshot 2026-02-18 225204" src="https://github.com/user-attachments/assets/4ea51c7f-c8fb-452c-b82d-124d16a4f7a2" />

---

## 🔧 Key Technical Decisions

### Why U-Net?
U-Net's skip connections preserve fine spatial details that are lost during downsampling, which is critical for accurate terrain boundary detection. It also trains effectively on relatively small datasets.

### Why Custom Mask Conversion?
The Falcon platform generates masks with non-sequential pixel values (200, 300, 500, 550, 800, 7100, 10000) rather than standard 0-based class indices. Our `convert_mask()` function in `dataset.py` maps these to sequential indices before training.

### Why Adam Optimizer?
Adam adapts the learning rate per parameter automatically, making it more stable and effective than standard SGD for segmentation tasks without requiring extensive manual tuning.

### Why ReduceLROnPlateau Scheduler?
Automatically halves the learning rate when validation loss stops improving, allowing fine-grained weight updates in later training stages without manual intervention.

---



