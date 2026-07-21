# ISRO Thermal to RGB Project Instructions

Follow these instructions to run the AI pipeline on a new machine (such as a college PC with an NVIDIA GPU).

## 1. Install Dependencies
Open a **PowerShell** terminal in this `DATASET` folder and install the required Python libraries:
```powershell
pip install -r requirements.txt
```

## 2. Set the Environment Path
Windows needs to know where the custom AI modules are located. In the same PowerShell window, run:
```powershell
$env:PYTHONPATH="."
```
*(Note: If you are using Command Prompt instead of PowerShell, use `set PYTHONPATH=.`)*

## 3. Train the AI Models
Because the data preprocessing is already completed and saved in the `FILES/processed/` folder, you can jump straight into training. Run these commands one after the other.

First, train the Super-Resolution model (Real-ESRGAN):
```powershell
python training/train_realesrgan.py
```
*(Wait for this to finish and save its checkpoints)*

Next, train the Colorization model (Pix2Pix):
```powershell
python training/train_pix2pix.py
```

## 4. Run the Web Dashboard
Once the training is completely finished and the checkpoints are saved, you can launch the interactive Streamlit dashboard to test the AI on new satellite images:
```powershell
streamlit run app.py
```
This will open a web browser where you can upload `.TIF` files and instantly view the generated High-Resolution RGB images and NDVI validation maps!
