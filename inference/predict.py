import os
import yaml
import torch
import numpy as np
import rasterio
from models.realesrgan.model import RRDBNet
from models.pix2pix.model import UNetGenerator
from evaluation.metrics import calculate_psnr, calculate_ssim

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def pad_image(arr, patch_size):
    """Pad image array so its dimensions are multiples of patch_size."""
    h, w = arr.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if len(arr.shape) == 3:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    else:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='edge')
    return padded, pad_h, pad_w

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

@torch.no_grad()
def run_inference(scene_dir, model_sr_path, model_pix2pix_path, output_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = config["data"]["patch_size"]
    scale = config["models"]["realesrgan"]["scale"]
    
    # Identify bands in the scene
    import glob
    b2_files = glob.glob(os.path.join(scene_dir, "*_B2.TIF"))
    if not b2_files:
        print(f"Error: B2 band not found in scene {scene_dir}")
        return None
    base_path = b2_files[0].replace("_B2.TIF", "")
    
    paths = {
        "B2": f"{base_path}_B2.TIF",
        "B3": f"{base_path}_B3.TIF",
        "B4": f"{base_path}_B4.TIF",
        "B10": f"{base_path}_B10.TIF",
        "B11": f"{base_path}_B11.TIF",
    }
    
    # Load and normalize full scene bands
    bands_data = {}
    original_profile = None
    for b, p in paths.items():
        with rasterio.open(p) as src:
            bands_data[b] = src.read(1)
            if b == "B2":
                original_profile = src.profile.copy()
                
    h_orig, w_orig = bands_data["B2"].shape
    
    # Normalize RGB (for ground truth comparison)
    gt_rgb = []
    for b in ["B4", "B3", "B2"]:
        b_min, b_max = get_percentile_min_max(bands_data[b])
        gt_rgb.append(normalize_band(bands_data[b], b_min, b_max))
    gt_rgb = np.stack(gt_rgb, axis=-1) # (H, W, 3) range [0, 1]
    
    # Normalize Thermals
    norm_thermals = []
    for b in ["B10", "B11"]:
        b_min, b_max = get_percentile_min_max(bands_data[b])
        norm_thermals.append(normalize_band(bands_data[b], b_min, b_max))
    norm_thermals = np.stack(norm_thermals, axis=-1) # (H, W, 2) range [0, 1]
    
    # Pad images for grid processing
    padded_thermals, pad_h, pad_w = pad_image(norm_thermals, patch_size)
    h_pad, w_pad, _ = padded_thermals.shape
    
    # Instantiate models
    # 1. Real-ESRGAN
    nf_sr = config["models"]["realesrgan"]["num_filters"]
    nb_sr = config["models"]["realesrgan"]["num_blocks"]
    net_sr = RRDBNet(in_nc=2, out_nc=2, nf=nf_sr, nb=nb_sr, gc=32, upscale=scale).to(device)
    if os.path.exists(model_sr_path):
        net_sr.load_state_dict(torch.load(model_sr_path, map_location=device))
        print(f"Loaded SR model weights from {model_sr_path}")
    else:
        print("Warning: SR model checkpoint not found. Using randomly initialized weights.")
    net_sr.eval()
    
    # 2. Pix2Pix
    net_g = UNetGenerator(input_nc=2, output_nc=3, num_downs=8, ngf=64).to(device)
    if os.path.exists(model_pix2pix_path):
        net_g.load_state_dict(torch.load(model_pix2pix_path, map_location=device))
        print(f"Loaded Pix2Pix model weights from {model_pix2pix_path}")
    else:
        print("Warning: Pix2Pix model checkpoint not found. Using randomly initialized weights.")
    net_g.eval()
    
    # Overlap parameters
    stride = patch_size // 2
    
    # Create 2D Hann window for blending
    window_1d = np.hanning(patch_size)
    window_2d = np.outer(window_1d, window_1d).astype(np.float32)
    # Expand window for broadcasting
    window_rgb = np.expand_dims(window_2d, axis=-1)  # (256, 256, 1)
    window_ir = np.expand_dims(window_2d, axis=-1)   # (256, 256, 1)

    # Output containers
    stitched_rgb = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
    stitched_enhanced_ir = np.zeros((h_pad, w_pad, 2), dtype=np.float32)
    
    # Weight accumulators
    weight_sum = np.zeros((h_pad, w_pad, 1), dtype=np.float32)
    
    print("Running overlapping patch-wise end-to-end inference...")
    # Loop over patches
    for y in range(0, h_pad - patch_size + 1, stride):
        for x in range(0, w_pad - patch_size + 1, stride):
            # Extract thermal patch
            patch_ir = padded_thermals[y : y + patch_size, x : x + patch_size] # (256, 256, 2)
            
            # 1. Simulate low-res thermal (for testing the end-to-end pipeline)
            # Downsample patch
            patch_ir_tensor = torch.from_numpy(patch_ir).permute(2, 0, 1).unsqueeze(0).float().to(device) # (1, 2, 256, 256)
            lr_ir_tensor = torch.nn.functional.interpolate(
                patch_ir_tensor, 
                size=(patch_size // scale, patch_size // scale), 
                mode='bilinear', 
                align_corners=False
            ) # (1, 2, 64, 64)
            
            # 2. Run Real-ESRGAN (Super-Resolution)
            sr_ir_tensor = net_sr(lr_ir_tensor) # (1, 2, 256, 256)
            sr_ir_tensor = torch.clamp(sr_ir_tensor, 0.0, 1.0)
            
            # 3. Run Pix2Pix (Colorization)
            fake_rgb_tensor = net_g(sr_ir_tensor) # (1, 3, 256, 256) range [-1, 1]
            
            # Convert back to [0, 1] range
            fake_rgb_tensor = (fake_rgb_tensor + 1.0) / 2.0
            fake_rgb_tensor = torch.clamp(fake_rgb_tensor, 0.0, 1.0)
            
            # Blend into stitched outputs
            sr_out = sr_ir_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            rgb_out = fake_rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            stitched_enhanced_ir[y : y + patch_size, x : x + patch_size] += sr_out * window_ir
            stitched_rgb[y : y + patch_size, x : x + patch_size] += rgb_out * window_rgb
            weight_sum[y : y + patch_size, x : x + patch_size] += window_rgb
            
    # Normalize by accumulated weights
    weight_sum[weight_sum == 0] = 1e-8
    stitched_rgb /= weight_sum
    stitched_enhanced_ir /= weight_sum
            
    # Crop padding back to original dimensions
    final_rgb = stitched_rgb[:h_orig, :w_orig]
    final_enhanced_ir = stitched_enhanced_ir[:h_orig, :w_orig]
    
    # Save the output stitched RGB as a GeoTIFF preserving original metadata
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    original_profile.update(
        count=3,
        dtype=rasterio.uint8,
        nodata=0
    )
    
    # Scale final_rgb to [0, 255] uint8
    final_rgb_uint8 = (final_rgb * 255.0).astype(np.uint8)
    
    # Set background zero values to 0
    bg_mask = (bands_data["B2"] == 0)
    final_rgb_uint8[bg_mask] = 0
    
    with rasterio.open(output_path, "w", **original_profile) as dst:
        # Write band-wise (rasterio expects count-first (C, H, W))
        dst.write(final_rgb_uint8[:, :, 0], 1) # Red
        dst.write(final_rgb_uint8[:, :, 1], 2) # Green
        dst.write(final_rgb_uint8[:, :, 2], 3) # Blue
        
    print(f"Stitched colorized image saved to {output_path} successfully!")
    
    # Compute full scene evaluation metrics
    # Mask out background pixels for metric calculations
    valid_mask = ~bg_mask
    if np.sum(valid_mask) > 0:
        psnr_val = calculate_psnr(final_rgb[valid_mask], gt_rgb[valid_mask])
        ssim_val = calculate_ssim(final_rgb, gt_rgb) # SSIM requires full image
        print(f"Full Scene Metrics - PSNR: {psnr_val:.4f} dB, SSIM: {ssim_val:.4f}")
    else:
        psnr_val, ssim_val = 0.0, 0.0
        
    return psnr_val, ssim_val

if __name__ == "__main__":
    # Test script with dummy parameters or run on first validation scene if checkpoints exist
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    checkpoints_dir = config["training"]["checkpoints_dir"]
    outputs_dir = config["training"]["outputs_dir"]
    
    import os
    scene_folders = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)) and d != "processed"])
    if scene_folders:
        first_scene = os.path.join(raw_dir, scene_folders[0])
        model_sr = os.path.join(checkpoints_dir, "realesrgan_best.pth")
        model_pix2pix = os.path.join(checkpoints_dir, "pix2pix_gen_epoch_4.pth")
        out_tif = os.path.join(outputs_dir, "stitched_output_test.TIF")
        
        print(f"Testing full scene prediction on {first_scene}")
        run_inference(first_scene, model_sr, model_pix2pix, out_tif, config)
