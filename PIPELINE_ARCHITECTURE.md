# ISRO Hackathon: Thermal-to-RGB AI Pipeline Architecture

This document outlines the end-to-end architecture of our dual-stage Artificial Intelligence pipeline designed to synthesize High-Resolution True-Color RGB imagery directly from low-resolution Thermal Infrared (TIRS) satellite data. 

This solution is critical for generating visual earth observation maps during events where optical cameras are blinded (e.g., thick cloud cover, nighttime, or dense wildfire smoke) but thermal sensors remain effective.

---

## 1. Data Engineering & Preprocessing (`preprocessing/preprocess.py`)
Because raw satellite scenes (like Landsat 8/9 `.TIF` files) are massive, they cannot be fed directly into a neural network. 

1. **Band Extraction:** We isolate Band 10 and Band 11 (Thermal), Band 5 (Near-Infrared), and Bands 2, 3, 4 (Optical RGB Ground Truth).
2. **Normalization:** The raw pixel values are normalized to a strict `[0, 1]` or `[-1, 1]` range using robust percentile clipping (ignoring outlier anomalies).
3. **Patch Generation:** The massive scene is sliced into `256x256` pixel grids (patches). Patches consisting mostly of empty black background (nodata) are algorithmically filtered out.
4. **Storage:** The valid patches are saved as compressed `.npz` files for rapid loading during GPU training.

---

## 2. Stage One: Thermal Super-Resolution (`models/realesrgan/`)
Thermal sensors natively capture data at a much lower spatial resolution (e.g., 100m/pixel) than optical sensors (e.g., 30m/pixel). If we colorize low-resolution thermal data, the resulting map will be incredibly blurry.

To fix this, the first stage of our pipeline uses a **Real-ESRGAN** (Enhanced Super-Resolution Generative Adversarial Network).
- **Input:** Low-resolution Thermal Patches.
- **Architecture:** Built using a Residual-in-Residual Dense Block (RRDBNet) backbone, which extracts deep spatial features without losing gradient information.
- **Output:** A mathematically upsampled, High-Resolution Thermal map (4x resolution boost) with artificially sharpened edges (like coastlines and rivers).

---

## 3. Stage Two: Structural Colorization (`models/pix2pix/`)
Once the thermal data is high-resolution, it must be translated into RGB color. We use a custom **Pix2Pix** conditional GAN framework for this image-to-image translation.

### The Generator (U-Net)
- Takes the High-Resolution Thermal data from Stage 1 as input.
- Uses an encoder-decoder architecture with "skip connections" to ensure the exact physical shapes of the landscape are preserved while it "paints" the RGB colors over them.

### The Discriminator (PatchGAN)
- Evaluates the Generator's artificially colored patches against the actual Ground Truth optical patches taken by the satellite on a clear day.
- Forces the Generator to learn the exact statistical mapping between temperature signatures and visual colors (e.g., teaching it that 15°C usually correlates to the blue of the ocean, and 22°C correlates to the green of a forest).

---

## 4. Physical Validation: Semantic Loss (`models/semantic/`)
Standard GANs often "hallucinate" colors just to trick the discriminator, which can result in physically inaccurate maps. To prevent this, we engineered a custom **Semantic Consistency Loss**.

During training, the pipeline dynamically calculates the **NDVI** (Normalized Difference Vegetation Index) and **NDWI** (Normalized Difference Water Index) using the AI-generated colors and the raw Near-Infrared (Band 5) data. 
- If the AI paints a forest (green) in a location where the physical NDVI signature says there is no vegetation, the custom loss function heavily penalizes the network. 
- This mathematically forces the AI to obey the actual physical laws of the earth rather than just painting a pretty picture.

---

## 5. End-to-End Inference & Stitching (`inference/predict.py`)
During deployment on new data, the pipeline must reconstruct a massive full-scene satellite image from the AI's 256x256 outputs without leaving visible grid lines.

- **Overlapping Sliding Window:** The inference script extracts patches with a 50% overlap.
- **Hann Window Blending:** A 2D mathematical gradient (feathering) is applied to every generated patch. 
- **Seamless Stitching:** The overlapping patches are averaged together using their blended weights, completely eliminating harsh borders and stitch lines, resulting in a flawless, massive `.TIF` output map.

---

## 6. The User Interface (`app.py`)
We built a premium, dark-themed **Streamlit Web Dashboard** to easily test and deploy the pipeline. 
- Users can select raw satellite folders from the UI.
- The dashboard automatically runs the end-to-end pipeline in the background.
- It displays a side-by-side comparison of the raw thermal input, the Ground Truth RGB, the AI-Generated RGB, and the physical NDVI consistency maps for immediate visual validation.
