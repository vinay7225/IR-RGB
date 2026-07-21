import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def calculate_psnr(gen_rgb, gt_rgb):
    """
    gen_rgb: Generated RGB image tensor/numpy array, shape (H, W, 3) or (3, H, W) in range [0, 1]
    gt_rgb: Ground truth RGB image tensor/numpy array, shape (H, W, 3) or (3, H, W) in range [0, 1]
    """
    if isinstance(gen_rgb, torch.Tensor):
        gen_rgb = gen_rgb.cpu().numpy()
    if isinstance(gt_rgb, torch.Tensor):
        gt_rgb = gt_rgb.cpu().numpy()
        
    # Standardize to (H, W, 3)
    if gen_rgb.shape[0] == 3:
        gen_rgb = gen_rgb.transpose(1, 2, 0)
    if gt_rgb.shape[0] == 3:
        gt_rgb = gt_rgb.transpose(1, 2, 0)
        
    return psnr_func(gt_rgb, gen_rgb, data_range=1.0)

def calculate_ssim(gen_rgb, gt_rgb):
    """
    gen_rgb: Generated RGB image, range [0, 1]
    gt_rgb: Ground truth RGB image, range [0, 1]
    """
    if isinstance(gen_rgb, torch.Tensor):
        gen_rgb = gen_rgb.cpu().numpy()
    if isinstance(gt_rgb, torch.Tensor):
        gt_rgb = gt_rgb.cpu().numpy()
        
    # Standardize to (H, W, 3)
    if gen_rgb.shape[0] == 3:
        gen_rgb = gen_rgb.transpose(1, 2, 0)
    if gt_rgb.shape[0] == 3:
        gt_rgb = gt_rgb.transpose(1, 2, 0)
        
    # Calculate SSIM channel-wise or with channel_axis
    # Note: scikit-image ssim uses channel_axis parameter in newer versions (like 0.19+)
    try:
        return ssim_func(gt_rgb, gen_rgb, channel_axis=2, data_range=1.0)
    except TypeError:
        # Fallback for older versions of scikit-image where multicast might differ
        ssims = []
        for i in range(3):
            ssims.append(ssim_func(gt_rgb[:, :, i], gen_rgb[:, :, i], data_range=1.0))
        return np.mean(ssims)

def calculate_fid_from_features(mu1, sigma1, mu2, sigma2):
    """
    Calculate Fréchet Inception Distance between two Gaussian distributions defined by (mu1, sigma1) and (mu2, sigma2).
    mu: 1D numpy array representing feature means
    sigma: 2D numpy array representing covariance matrix
    """
    import scipy.linalg
    # mu difference
    diff = mu1 - mu2
    
    # product of covariances
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # numerical instability fallback
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    # check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def main():
    # Simple test of metrics
    print("Testing metrics calculation...")
    dummy_gen = torch.rand(3, 256, 256)
    dummy_gt = dummy_gen + torch.randn(3, 256, 256) * 0.05
    dummy_gt = torch.clamp(dummy_gt, 0.0, 1.0)
    
    psnr_val = calculate_psnr(dummy_gen, dummy_gt)
    ssim_val = calculate_ssim(dummy_gen, dummy_gt)
    
    print(f"Test PSNR: {psnr_val:.4f} dB")
    print(f"Test SSIM: {ssim_val:.4f}")

if __name__ == "__main__":
    main()
