import torch
import torch.nn as nn

class SemanticIndexLoss(nn.Module):
    """
    Semantic Integrity Loss using physical spectral indices:
    - NDVI (Normalized Difference Vegetation Index)
    - NDWI (Normalized Difference Water Index)
    Forces the generated RGB channels to preserve vegetation and water reflectance properties.
    """
    def __init__(self, eps=1e-6):
        super(SemanticIndexLoss, self).__init__()
        self.eps = eps
        self.l1_loss = nn.L1Loss()

    def denormalize(self, x):
        """Convert from [-1, 1] (GAN output/target) to [0, 1] range."""
        return (x + 1.0) / 2.0

    def compute_ndvi(self, red, nir):
        """NDVI = (NIR - Red) / (NIR + Red)"""
        return (nir - red) / (nir + red + self.eps)

    def compute_ndwi(self, green, nir):
        """NDWI = (Green - NIR) / (Green + NIR)"""
        return (green - nir) / (green + nir + self.eps)

    def forward(self, gen_rgb, gt_rgb, nir):
        """
        gen_rgb: Generated RGB image tensor, shape (B, 3, H, W) in range [-1, 1]
        gt_rgb: Ground truth RGB image tensor, shape (B, 3, H, W) in range [-1, 1]
        nir: Input NIR band (B5) tensor, shape (B, 1, H, W) in range [0, 1]
        """
        # Convert generated and GT to [0, 1] range
        gen_rgb_01 = self.denormalize(gen_rgb)
        gt_rgb_01 = self.denormalize(gt_rgb)
        
        # Extract channels: Red is channel 0, Green is channel 1
        gen_red = gen_rgb_01[:, 0:1, :, :]
        gen_green = gen_rgb_01[:, 1:2, :, :]
        
        gt_red = gt_rgb_01[:, 0:1, :, :]
        gt_green = gt_rgb_01[:, 1:2, :, :]
        
        # Compute indices for ground truth and generated
        gt_ndvi = self.compute_ndvi(gt_red, nir)
        gen_ndvi = self.compute_ndvi(gen_red, nir)
        
        gt_ndwi = self.compute_ndwi(gt_green, nir)
        gen_ndwi = self.compute_ndwi(gen_green, nir)
        
        # Calculate loss
        ndvi_loss = self.l1_loss(gen_ndvi, gt_ndvi)
        ndwi_loss = self.l1_loss(gen_ndwi, gt_ndwi)
        
        return ndvi_loss + ndwi_loss
