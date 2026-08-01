import os
import sys
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.realesrgan.model import RRDBNet
from utils.dataset import LandsatDataset

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    try:
        import torch_xla.core.xla_model as xm
        is_tpu = True
    except ImportError:
        is_tpu = False

    if is_tpu:
        device = xm.xla_device()
        print(f"Using TPU device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # Paths
    processed_dir = config["data"]["processed_dir"]
    checkpoints_dir = config["training"]["checkpoints_dir"]
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Dataset and Dataloader
    scale = config["models"]["realesrgan"]["scale"]
    train_dataset = LandsatDataset(os.path.join(processed_dir, "train"), is_train=True, scale_factor=scale)
    val_dataset = LandsatDataset(os.path.join(processed_dir, "val"), is_train=False, scale_factor=scale)
    
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Run preprocessing first.")
        return
        
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} train patches and {len(val_dataset)} val patches.")
    
    # Model
    nf = config["models"]["realesrgan"]["num_filters"]
    nb = config["models"]["realesrgan"]["num_blocks"]
    model = RRDBNet(in_nc=2, out_nc=2, nf=nf, nb=nb, gc=32, upscale=scale).to(device)
    
    # Optimizer and Loss
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"], betas=(0.9, 0.999))
    
    epochs = config["training"]["epochs_sr"]
    save_freq = config["training"]["save_epoch_freq"]
    
    print("Starting Real-ESRGAN training...")
    best_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in loop:
            # ir_lr is low-res thermal, shape (B, 2, 64, 64)
            # ir_hr is high-res thermal, shape (B, 2, 256, 256)
            lr_ir = batch["ir_lr"].to(device)
            hr_ir = batch["ir_hr"].to(device)
            
            optimizer.zero_grad()
            sr_ir = model(lr_ir)
            loss = criterion(sr_ir, hr_ir)
            loss.backward()
            if is_tpu:
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        avg_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                lr_ir = batch["ir_lr"].to(device)
                hr_ir = batch["ir_hr"].to(device)
                sr_ir = model(lr_ir)
                loss = criterion(sr_ir, hr_ir)
                avg_val_loss += loss.item()
                
        if len(val_loader) > 0:
            avg_val_loss = avg_val_loss / len(val_loader)
        else:
            avg_val_loss = avg_train_loss
            
        print(f"Epoch {epoch} finished - Train Loss: {avg_train_loss:.5f} - Val Loss: {avg_val_loss:.5f}")
        
        # Save checkpoints
        if epoch % save_freq == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoints_dir, f"realesrgan_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_checkpoint_path = os.path.join(checkpoints_dir, "realesrgan_best.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Saved new best checkpoint to {best_checkpoint_path}")
            
    print("Real-ESRGAN training finished!")

if __name__ == "__main__":
    main()
