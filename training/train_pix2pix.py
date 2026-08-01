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

from models.pix2pix.model import UNetGenerator, PatchGANDiscriminator
from models.semantic.semantic_loss import SemanticIndexLoss
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
    
    # Models
    # Input is 2-channel IR image, Output is 3-channel RGB image
    net_g = UNetGenerator(input_nc=2, output_nc=3, num_downs=8, ngf=64).to(device)
    # Input is 2-channel IR + 3-channel RGB = 5 channels
    net_d = PatchGANDiscriminator(input_nc=5, ndf=64).to(device)
    
    # Loss functions
    criterion_gan = nn.MSELoss() # Least Squares GAN
    criterion_l1 = nn.L1Loss()
    criterion_semantic = SemanticIndexLoss().to(device)
    
    # Optimizers
    lr = config["training"]["lr"]
    beta1 = config["training"]["beta1"]
    beta2 = config["training"]["beta2"]
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Hyperparameters
    lambda_l1 = config["models"]["pix2pix"]["lambda_l1"]
    lambda_semantic = config["models"]["pix2pix"]["lambda_semantic"]
    epochs = config["training"]["epochs_pix2pix"]
    save_freq = config["training"]["save_epoch_freq"]
    
    print("Starting Pix2Pix Colorization training...")
    best_g_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        net_g.train()
        net_d.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in loop:
            # Inputs
            real_ir = batch["ir_hr"].to(device) # shape: (B, 2, 256, 256)
            real_rgb = batch["rgb"].to(device)  # shape: (B, 3, 256, 256), range [-1, 1]
            nir = batch["nir"].to(device)        # shape: (B, 1, 256, 256), range [0, 1]
            
            # --- 1. Train Discriminator ---
            optimizer_d.zero_grad()
            
            # Fake generation
            fake_rgb = net_g(real_ir)
            
            # D(cat(real_ir, real_rgb))
            real_pair = torch.cat((real_ir, real_rgb), dim=1)
            pred_real = net_d(real_pair)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            
            # D(cat(real_ir, fake_rgb))
            fake_pair = torch.cat((real_ir, fake_rgb.detach()), dim=1)
            pred_fake = net_d(fake_pair)
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            # Combined D loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            if is_tpu:
                xm.optimizer_step(optimizer_d, barrier=True)
            else:
                optimizer_d.step()
            
            # --- 2. Train Generator ---
            optimizer_g.zero_grad()
            
            # GAN loss: G wants D to think the fake is real
            fake_pair_for_g = torch.cat((real_ir, fake_rgb), dim=1)
            pred_fake_for_g = net_d(fake_pair_for_g)
            loss_g_gan = criterion_gan(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            
            # Reconstruction Loss
            loss_g_l1 = criterion_l1(fake_rgb, real_rgb)
            
            # Semantic Constraint Loss
            loss_g_semantic = criterion_semantic(fake_rgb, real_rgb, nir)
            
            # Combined G loss
            loss_g = loss_g_gan + (lambda_l1 * loss_g_l1) + (lambda_semantic * loss_g_semantic)
            loss_g.backward()
            if is_tpu:
                xm.optimizer_step(optimizer_g, barrier=True)
            else:
                optimizer_g.step()
            
            # Logging
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            loop.set_postfix(G_loss=loss_g.item(), D_loss=loss_d.item())
            
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        print(f"Epoch {epoch} finished - G Loss: {avg_g_loss:.5f} - D Loss: {avg_d_loss:.5f}")
        
        # Save checkpoints
        if epoch % save_freq == 0 or epoch == epochs:
            checkpoint_g_path = os.path.join(checkpoints_dir, f"pix2pix_gen_epoch_{epoch}.pth")
            checkpoint_d_path = os.path.join(checkpoints_dir, f"pix2pix_disc_epoch_{epoch}.pth")
            torch.save(net_g.state_dict(), checkpoint_g_path)
            torch.save(net_d.state_dict(), checkpoint_d_path)
            print(f"Saved checkpoints at epoch {epoch}")
            
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_checkpoint_path = os.path.join(checkpoints_dir, "pix2pix_gen_best.pth")
            torch.save(net_g.state_dict(), best_checkpoint_path)
            print(f"Saved new best Generator checkpoint to {best_checkpoint_path}")
            
    print("Pix2Pix training finished!")

if __name__ == "__main__":
    main()
