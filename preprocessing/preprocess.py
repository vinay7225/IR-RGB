import os
import sys
import yaml
import glob
from pathlib import Path
import numpy as np
import rasterio
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_config(config_path="configs/config.yaml"):
    config_file = PROJECT_ROOT / config_path
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def get_percentile_min_max(band_arr, p_min=2, p_max=98):
    """Compute robust min and max values excluding background (0) values."""
    valid_pixels = band_arr[band_arr > 0]
    if len(valid_pixels) == 0:
        return 0.0, 1.0
    return np.percentile(valid_pixels, p_min), np.percentile(valid_pixels, p_max)

def normalize_band(band_arr, b_min, b_max):
    """Normalize band to [0, 1] using min and max, clipping outliers."""
    if b_max - b_min < 1e-5:
        return np.zeros_like(band_arr, dtype=np.float32)
    normalized = (band_arr.astype(np.float32) - b_min) / (b_max - b_min)
    return np.clip(normalized, 0.0, 1.0)

def preprocess_scene(scene_dir, config):
    """Load scene bands, normalize them, and return combined arrays."""
    # Find files matching the bands
    b2_files = glob.glob(os.path.join(scene_dir, "*_B2.TIF"))
    if not b2_files:
        return None
    
    # Extract base file path to find other bands
    base_path = b2_files[0].replace("_B2.TIF", "")
    
    paths = {
        "B2": f"{base_path}_B2.TIF", # Blue
        "B3": f"{base_path}_B3.TIF", # Green
        "B4": f"{base_path}_B4.TIF", # Red
        "B5": f"{base_path}_B5.TIF", # NIR (for NDVI)
        "B10": f"{base_path}_B10.TIF", # Thermal 1
        "B11": f"{base_path}_B11.TIF", # Thermal 2
    }
    
    # Read all bands
    data = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Warning: Band {name} not found at {path}")
            return None
        with rasterio.open(path) as src:
            data[name] = src.read(1)
            
    # Make sure all shapes match
    shape = data["B2"].shape
    for name, arr in data.items():
        if arr.shape != shape:
            print(f"Error: Shape mismatch for band {name} in scene {scene_dir}")
            return None
            
    # Compute robust normalization parameters
    normalized_data = {}
    
    # Normalizing RGB (B4, B3, B2)
    rgb_bands = ["B4", "B3", "B2"]
    for b in rgb_bands:
        b_min, b_max = get_percentile_min_max(data[b])
        normalized_data[b] = normalize_band(data[b], b_min, b_max)
        
    # Normalizing NIR (B5)
    b5_min, b5_max = get_percentile_min_max(data["B5"])
    normalized_data["B5"] = normalize_band(data["B5"], b5_min, b5_max)
    
    # Normalizing Thermals (B10, B11)
    thermal_bands = ["B10", "B11"]
    for b in thermal_bands:
        b_min, b_max = get_percentile_min_max(data[b])
        normalized_data[b] = normalize_band(data[b], b_min, b_max)
        
    # Combine bands into stacks
    # RGB stack: H x W x 3
    rgb_stack = np.stack([normalized_data["B4"], normalized_data["B3"], normalized_data["B2"]], axis=-1)
    # IR stack: H x W x 2
    ir_stack = np.stack([normalized_data["B10"], normalized_data["B11"]], axis=-1)
    # NIR stack: H x W x 1
    nir_stack = np.expand_dims(normalized_data["B5"], axis=-1)
    
    # Create mask of valid data (pixels where all bands are non-zero)
    # Since background DN is 0, we identify background pixels
    valid_mask = (data["B2"] > 0) & (data["B10"] > 0)
    
    return rgb_stack, ir_stack, nir_stack, valid_mask

def save_patches(scene_name, rgb_stack, ir_stack, nir_stack, valid_mask, output_dir, config):
    """Slice stacked bands into patches and save to output directory."""
    patch_size = config["data"]["patch_size"]
    stride = config["data"]["stride"]
    max_bg_pct = config["data"]["max_background_pct"]
    
    h, w, _ = rgb_stack.shape
    os.makedirs(output_dir, exist_ok=True)
    
    patch_count = 0
    # Slide patch window across the scene
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patches
            patch_valid = valid_mask[y : y + patch_size, x : x + patch_size]
            
            # Check background proportion
            bg_proportion = 1.0 - (np.sum(patch_valid) / (patch_size * patch_size))
            if bg_proportion > max_bg_pct:
                continue
                
            patch_rgb = rgb_stack[y : y + patch_size, x : x + patch_size]
            patch_ir = ir_stack[y : y + patch_size, x : x + patch_size]
            patch_nir = nir_stack[y : y + patch_size, x : x + patch_size]
            
            # Save as NPZ
            patch_filename = f"{scene_name}_p_{y}_{x}.npz"
            np.savez_compressed(
                os.path.join(output_dir, patch_filename),
                rgb=patch_rgb,
                ir=patch_ir,
                nir=patch_nir
            )
            patch_count += 1
            
    return patch_count

def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    
    # Get all scene directories
    scene_dirs = sorted([
        os.path.join(raw_dir, d) for d in os.listdir(raw_dir) 
        if os.path.isdir(os.path.join(raw_dir, d)) and d != "processed"
    ])
    
    print(f"Found {len(scene_dirs)} scenes.")
    
    # Split scenes between train and validation (avoiding spatial leakage)
    split_idx = int(len(scene_dirs) * config["data"]["train_split"])
    train_scenes = scene_dirs[:split_idx]
    val_scenes = scene_dirs[split_idx:]
    
    print(f"Splitting into {len(train_scenes)} training scenes and {len(val_scenes)} validation scenes.")
    
    # Process training scenes
    print("\nProcessing training scenes...")
    total_train_patches = 0
    for scene in train_scenes:
        scene_name = os.path.basename(scene).replace(" ", "_")
        print(f"Processing scene: {scene_name}")
        scene_data = preprocess_scene(scene, config)
        if scene_data is None:
            continue
        rgb, ir, nir, mask = scene_data
        patches = save_patches(
            scene_name, rgb, ir, nir, mask, 
            os.path.join(processed_dir, "train"), config
        )
        print(f"Generated {patches} valid patches.")
        total_train_patches += patches
        
    # Process validation scenes
    print("\nProcessing validation scenes...")
    total_val_patches = 0
    for scene in val_scenes:
        scene_name = os.path.basename(scene).replace(" ", "_")
        print(f"Processing scene: {scene_name}")
        scene_data = preprocess_scene(scene, config)
        if scene_data is None:
            continue
        rgb, ir, nir, mask = scene_data
        patches = save_patches(
            scene_name, rgb, ir, nir, mask, 
            os.path.join(processed_dir, "val"), config
        )
        print(f"Generated {patches} valid patches.")
        total_val_patches += patches
        
    print(f"\nPreprocessing finished! Total train patches: {total_train_patches}, Total val patches: {total_val_patches}")

if __name__ == "__main__":
    main()
