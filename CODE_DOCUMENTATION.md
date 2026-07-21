# Codebase Documentation

This document provides a comprehensive explanation of every module and script within the project repository. It serves as a technical guide for developers and hackathon judges to understand the exact purpose of each file in the AI pipeline.

---

## 1. Data Preprocessing
### `preprocessing/preprocess.py`
**Purpose:** Prepares the raw Landsat `.TIF` files for neural network training.
**What it does:**
- Iterates through the raw satellite scene folders.
- Extracts the Thermal bands (10, 11), Optical RGB bands (2, 3, 4), and the NIR band (5).
- Normalizes pixel values strictly between `0` and `1` using robust percentile clipping.
- Slices the massive satellite images into small `256x256` pixel grids (patches).
- Filters out empty patches (nodata) and saves the valid ones as compressed `.npz` arrays to accelerate training.

---

## 2. Core Utilities
### `utils/dataset.py`
**Purpose:** The PyTorch DataLoader class.
**What it does:** 
- Automatically loads the `.npz` patches from the `processed/` folder during training.
- Converts the numpy arrays into PyTorch Tensors.
- Artificially downsamples the High-Res thermal patch into a Low-Res version so that the Real-ESRGAN model can learn how to super-resolve it back to High-Res.
- Feeds the batched data directly into the GPUs.

### `configs/config.yaml`
**Purpose:** Centralized configuration management.
**What it does:** Stores all hyperparameters in one place (e.g., learning rates, batch sizes, patch sizes, epochs, and directory paths) so you never have to hardcode values inside the Python scripts.

---

## 3. AI Models
### `models/realesrgan/model.py`
**Purpose:** Defines the Super-Resolution architecture (Stage 1).
**What it does:** Contains the PyTorch class for the `RRDBNet` (Residual-in-Residual Dense Block Network). This model takes a blurry, low-resolution thermal patch and outputs a sharp, 4x upscaled thermal patch.

### `models/pix2pix/model.py`
**Purpose:** Defines the Colorization GAN architecture (Stage 2).
**What it does:** Contains two PyTorch classes:
1. `UNetGenerator`: The "Artist" that takes the sharpened thermal patch and generates a fake RGB image.
2. `Discriminator`: The "Detective" (a PatchGAN) that looks at the generated RGB and the Ground Truth RGB, and tries to guess which one is fake. This adversarial process forces the Generator to produce hyper-realistic colors.

### `models/semantic/semantic_loss.py`
**Purpose:** The custom physical validation loss function.
**What it does:** Contains the PyTorch logic to dynamically calculate the Normalized Difference Vegetation Index (NDVI) and Normalized Difference Water Index (NDWI) during training. It penalizes the Pix2Pix Generator if it hallucinates colors that don't match the actual physical properties of the earth based on the infrared signatures.

---

## 4. Training Pipelines
### `training/train_realesrgan.py`
**Purpose:** Trains the Stage 1 Super-Resolution model.
**What it does:** Initializes the `RRDBNet`, loads the dataset, and runs the backpropagation loop using L1/MSE loss. It saves the best-performing weights as `realesrgan_best.pth`.

### `training/train_pix2pix.py`
**Purpose:** Trains the Stage 2 Colorization model.
**What it does:** Initializes the Pix2Pix Generator and Discriminator. It runs the complex adversarial training loop, combining standard GAN loss, L1 pixel loss, and our custom Semantic Loss. It saves the best-performing weights as `pix2pix_gen_best.pth`.

---

## 5. Evaluation & Inference
### `evaluation/metrics.py`
**Purpose:** Quantitative accuracy measurement.
**What it does:** Contains functions to calculate mathematical image quality metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

### `inference/predict.py`
**Purpose:** End-to-end full scene prediction.
**What it does:** 
- Loads the saved `.pth` model weights.
- Takes an entire massive raw satellite scene and loops over it using an **overlapping sliding window**.
- Runs both the Real-ESRGAN and Pix2Pix models sequentially.
- Uses a **2D Hann window blending strategy** to stitch the generated patches back together perfectly without visible grid lines.
- Saves the final High-Resolution colorized `.TIF` image.

---

## 6. User Interface
### `app.py`
**Purpose:** The interactive web dashboard.
**What it does:** Built using Streamlit, this file creates the beautiful front-end interface. It allows users to select raw data folders, runs the inference models dynamically in the background, and plots side-by-side visual comparisons and NDVI physical validation maps in the browser.
