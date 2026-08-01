import os
import sys
import yaml
import glob
from pathlib import Path
import torch
import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from evaluation.metrics import calculate_psnr, calculate_ssim

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.realesrgan.model import RRDBNet
from models.pix2pix.model import UNetGenerator

# Page Configuration for Premium Look
st.set_page_config(
    page_title="ISRO Satellite IR Colorization & Enhancement",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics (dark mode styling, cards, smooth transitions, custom fonts)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0f111a;
        color: #ffffff;
    }
    
    /* Title and Header styling */
    .title-gradient {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #8f9cae;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(4px);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 254, 0.3);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f2fe;
        margin-top: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8f9cae;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button custom style */
    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
        transform: scale(1.02);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #08090f;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_percentile_min_max(band_arr, p_min=2, p_max=98):
    valid_pixels = band_arr[band_arr > 0]
    if len(valid_pixels) == 0:
        return 0.0, 1.0
    return np.percentile(valid_pixels, p_min), np.percentile(valid_pixels, p_max)

def normalize_band(band_arr, b_min, b_max):
    if b_max - b_min < 1e-5:
        return np.zeros_like(band_arr, dtype=np.float32)
    normalized = (band_arr.astype(np.float32) - b_min) / (b_max - b_min)
    return np.clip(normalized, 0.0, 1.0)

def load_scene_data(scene_dir):
    b2_files = glob.glob(os.path.join(scene_dir, "*_B2.TIF"))
    if not b2_files:
        return None
    base_path = b2_files[0].replace("_B2.TIF", "")
    
    paths = {
        "B2": f"{base_path}_B2.TIF",
        "B3": f"{base_path}_B3.TIF",
        "B4": f"{base_path}_B4.TIF",
        "B5": f"{base_path}_B5.TIF",
        "B10": f"{base_path}_B10.TIF",
        "B11": f"{base_path}_B11.TIF",
    }
    
    bands_data = {}
    for b, p in paths.items():
        if not os.path.exists(p):
            return None
        with rasterio.open(p) as src:
            # Subsample image for faster local demo loading (e.g. step by 4)
            # Full 9000x9000 is too large to process in real-time in a streamlit UI!
            # Reading a 1024x1024 crop from the center is perfect for demonstration.
            h, w = src.shape
            cy, cx = h // 2, w // 2
            crop_size = 1024
            window = rasterio.windows.Window(cx - crop_size//2, cy - crop_size//2, crop_size, crop_size)
            bands_data[b] = src.read(1, window=window)
            
    return bands_data

def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    checkpoints_dir = config["training"]["checkpoints_dir"]
    
    # Header
    st.markdown('<div class="title-gradient">Bharatiya Antariksh Hackathon 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Infrared Image Colorization and Enhancement for Satellite Imagery</div>', unsafe_allow_html=True)
    
    # Find scene folders
    scene_folders = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)) and d != "processed"])
    
    # Sidebar config
    st.sidebar.markdown("### 🛰️ Dataset & Scene selection")
    if not scene_folders:
        st.sidebar.error("No Landsat scenes found in FILES directory.")
        return
        
    selected_scene_name = st.sidebar.selectbox("Select Landsat Scene", scene_folders)
    selected_scene_dir = os.path.join(raw_dir, selected_scene_name)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Model Configuration")
    
    # Get available checkpoints
    all_chks = os.listdir(checkpoints_dir) if os.path.exists(checkpoints_dir) else []
    sr_chks = sorted([f for f in all_chks if f.startswith("realesrgan_") and f.endswith(".pth")])
    pix_chks = sorted([f for f in all_chks if f.startswith("pix2pix_gen_") and f.endswith(".pth")])
    
    if not sr_chks:
        st.sidebar.warning("⚠️ No Real-ESRGAN checkpoints found.")
        sr_chk = None
        sr_found = False
    else:
        def_sr_idx = sr_chks.index("realesrgan_best.pth") if "realesrgan_best.pth" in sr_chks else len(sr_chks)-1
        sel_sr = st.sidebar.selectbox("Real-ESRGAN Checkpoint", sr_chks, index=def_sr_idx)
        sr_chk = os.path.join(checkpoints_dir, sel_sr)
        sr_found = True
        
    if not pix_chks:
        st.sidebar.warning("⚠️ No Pix2Pix checkpoints found.")
        pix_chk = None
        pix_found = False
    else:
        def_pix_idx = pix_chks.index("pix2pix_gen_best.pth") if "pix2pix_gen_best.pth" in pix_chks else len(pix_chks)-1
        sel_pix = st.sidebar.selectbox("Pix2Pix Checkpoint", pix_chks, index=def_pix_idx)
        pix_chk = os.path.join(checkpoints_dir, sel_pix)
        pix_found = True
        
    if sr_found and pix_found:
        st.sidebar.success("✔️ Trained model checkpoints selected.")
        
    # Load scene data
    with st.spinner("Loading satellite bands..."):
        data = load_scene_data(selected_scene_dir)
        
    if data is None:
        st.error("Failed to load bands. Please ensure B2, B3, B4, B5, B10, and B11 TIFF files exist.")
        return
        
    # Preprocess crop
    # Normalize inputs
    rgb_min_max = {b: get_percentile_min_max(data[b]) for b in ["B4", "B3", "B2"]}
    gt_rgb = np.stack([
        normalize_band(data["B4"], *rgb_min_max["B4"]),
        normalize_band(data["B3"], *rgb_min_max["B3"]),
        normalize_band(data["B2"], *rgb_min_max["B2"])
    ], axis=-1)
    
    t10_min, t10_max = get_percentile_min_max(data["B10"])
    t11_min, t11_max = get_percentile_min_max(data["B11"])
    norm_t10 = normalize_band(data["B10"], t10_min, t10_max)
    norm_t11 = normalize_band(data["B11"], t11_min, t11_max)
    ir_input = np.stack([norm_t10, norm_t11], axis=-1)
    
    # Display raw details in tabs
    tab_inspect, tab_enhance = st.tabs(["🔍 Inspect Bands", "🚀 Enhancement & Colorization"])
    
    with tab_inspect:
        st.markdown("### Raw Satellite Spectral Bands (1024x1024 Center Crop)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(norm_t10, caption="Band 10: Thermal Infrared (TIRS) 1", use_container_width=True, clamp=True)
        with col2:
            st.image(norm_t11, caption="Band 11: Thermal Infrared (TIRS) 2", use_container_width=True, clamp=True)
        with col3:
            # NIR band
            nir_min, nir_max = get_percentile_min_max(data["B5"])
            norm_nir = normalize_band(data["B5"], nir_min, nir_max)
            st.image(norm_nir, caption="Band 5: Near Infrared (NIR)", use_container_width=True, clamp=True)
            
        st.markdown("---")
        st.markdown("### Ground Truth RGB Composition")
        st.image(gt_rgb, caption="True Color RGB Composite (Bands 4, 3, 2)", use_container_width=True)

    with tab_enhance:
        st.markdown("### Run Inference Pipeline")
        st.write("Apply Real-ESRGAN super-resolution to TIRS bands, then convert to colorized RGB using Pix2Pix.")
        
        if st.button("🚀 Process Image"):
            # Load PyTorch models
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scale = config["models"]["realesrgan"]["scale"]
            
            # Instantiating Real-ESRGAN
            nf_sr = config["models"]["realesrgan"]["num_filters"]
            nb_sr = config["models"]["realesrgan"]["num_blocks"]
            net_sr = RRDBNet(in_nc=2, out_nc=2, nf=nf_sr, nb=nb_sr, gc=32, upscale=scale).to(device)
            if sr_found:
                net_sr.load_state_dict(torch.load(sr_chk, map_location=device))
            net_sr.eval()
            
            # Instantiating Pix2Pix
            net_g = UNetGenerator(input_nc=2, output_nc=3, num_downs=8, ngf=64).to(device)
            if pix_found:
                net_g.load_state_dict(torch.load(pix_chk, map_location=device))
            net_g.eval()
            
            # Perform inference on crops
            # To handle 1024x1024, we slice it into 4x4 non-overlapping patches of 256x256
            patch_size = config["data"]["patch_size"]
            h_crop, w_crop, _ = ir_input.shape
            
            stitched_rgb = np.zeros((h_crop, w_crop, 3), dtype=np.float32)
            stitched_sr_ir = np.zeros((h_crop, w_crop, 2), dtype=np.float32)
            
            with torch.no_grad():
                for y in range(0, h_crop, patch_size):
                    for x in range(0, w_crop, patch_size):
                        # Extract patch
                        patch_ir = ir_input[y : y + patch_size, x : x + patch_size]
                        patch_ir_tensor = torch.from_numpy(patch_ir).permute(2, 0, 1).unsqueeze(0).float().to(device)
                        
                        # Simulate low-res first for the demonstration of super-res enhancement
                        lr_ir_tensor = torch.nn.functional.interpolate(
                            patch_ir_tensor, 
                            size=(patch_size // scale, patch_size // scale), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        # Super resolve
                        sr_ir_tensor = net_sr(lr_ir_tensor)
                        sr_ir_tensor = torch.clamp(sr_ir_tensor, 0.0, 1.0)
                        
                        # Colorize
                        fake_rgb_tensor = net_g(sr_ir_tensor)
                        # Scale back to [0, 1]
                        fake_rgb_tensor = (fake_rgb_tensor + 1.0) / 2.0
                        fake_rgb_tensor = torch.clamp(fake_rgb_tensor, 0.0, 1.0)
                        
                        # Save
                        stitched_sr_ir[y : y + patch_size, x : x + patch_size] = sr_ir_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        stitched_rgb[y : y + patch_size, x : x + patch_size] = fake_rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        
            # Metrics
            psnr_val = calculate_psnr(stitched_rgb, gt_rgb)
            ssim_val = calculate_ssim(stitched_rgb, gt_rgb)
            
            # Display metrics cards
            st.markdown("### 📊 Quantitative Evaluation")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Signal-to-Noise Ratio (PSNR)</div>
                    <div class="metric-value">{psnr_val:.2f} dB</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Structural Similarity Index (SSIM)</div>
                    <div class="metric-value">{ssim_val:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            st.markdown("### 🎨 Comparison Results")
            
            # Displays
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(norm_t10, caption="Input Raw Thermal (Band 10)", use_container_width=True)
                st.image(stitched_rgb, caption="Enhanced & Colorized RGB (Model Output)", use_container_width=True)
            with col_res2:
                st.image(stitched_sr_ir[:, :, 0], caption="Super-Resolved Thermal (Real-ESRGAN)", use_container_width=True)
                st.image(gt_rgb, caption="Ground Truth True Color RGB", use_container_width=True)
                
            st.markdown("---")
            st.markdown("### 🌱 Semantic Physical Consistency (NDVI Maps)")
            st.write("Compare the Normalized Difference Vegetation Index (NDVI) between the Ground Truth RGB and the Model Output to verify physical accuracy.")
            
            # Calculate NDVI: using NIR (B5) and Red (B4)
            # In gt_rgb, Red is index 0. In stitched_rgb, Red is index 0.
            # B5 is norm_nir
            nir_min, nir_max = get_percentile_min_max(data["B5"])
            norm_nir = normalize_band(data["B5"], nir_min, nir_max)
            
            gt_red = gt_rgb[:, :, 0]
            gen_red = stitched_rgb[:, :, 0]
            
            eps = 1e-6
            gt_ndvi = (norm_nir - gt_red) / (norm_nir + gt_red + eps)
            gen_ndvi = (norm_nir - gen_red) / (norm_nir + gen_red + eps)
            
            # Scale NDVI [-1, 1] to [0, 1] for gray scaling in streamlit
            gt_ndvi_disp = (gt_ndvi + 1.0) / 2.0
            gen_ndvi_disp = (gen_ndvi + 1.0) / 2.0
            
            col_ndvi1, col_ndvi2 = st.columns(2)
            with col_ndvi1:
                st.image(gt_ndvi_disp, caption="Ground Truth NDVI Map", use_container_width=True, clamp=True)
            with col_ndvi2:
                st.image(gen_ndvi_disp, caption="Generated NDVI Map", use_container_width=True, clamp=True)

if __name__ == "__main__":
    main()
