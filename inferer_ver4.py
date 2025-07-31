# To run this script, activate: conda activate py311_3
# %% Call packages
import argparse
import os
import sys
import time
import warnings

# Suppress most warnings for cleaner logs (comment out if debugging is needed)
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random

# Set default dtype to float32 to avoid dtype mismatches
torch.set_default_dtype(torch.float32)

# TensorBoard logging
from torch.utils.tensorboard.writer import SummaryWriter
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # os.path.dirname(__file__): parent dir ../MOTFM/  + join(MOTFM, ..) = any directory in MOTFM

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.ensemble import HistGradientBoostingRegressor

from utils.general_utils import (
    load_config,
    load_and_prepare_data,
    create_dataloader,
    save_checkpoint,
    load_checkpoint,
)
from utils.utils_fm import build_model, validate_and_save_samples
from utils.scaling import GBoost, Sigmoid2, QT
from monai.inferers.utils import sliding_window_inference # sliding window

#---------added part--------
import SimpleITK as sitk # read .mha
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
is_interactive = sys.stdout.isatty()
from matplotlib import pyplot as plt



print("Configuration and Paths")

# Experiment configuration
exp_name = "test21_2"
SAMPLING_QUALITY = "fast"  # Options: "fast" (10 steps), "medium" (50 steps), "high" (100 steps), "ultra" (200 steps)
SAMPLING_STEPS = {
    "fast": 10,
    "medium": 50, 
    "high": 100,
    "ultra": 200
}

# Paths - all defined in one location
config_path = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/exp_configs/test20-testseed1-nomaskloss.yaml"
inference_path = "/media/prajbori/sda/private/dataset/proj_synthrad/training/synthRAD2025_Task1_Train_D/Task1/HN_x/1HND001"
latest_ckpt_dir = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/some_checkpoints/test20-testseed1-nomaskloss/epoch_500"

CT_UPPER = 3071.0
CT_LOWER = -1024.0
MRI_UPPER = 1357.0
MRI_LOWER = 0.0
OVERLAP_SIZE = (4, 32, 32)

print("Initiate model and load weights")

config = load_config(config_path)

# Read core settings from config
num_epochs = config["train_args"]["num_epochs"]
batch_size = config["train_args"]["batch_size"]    
lr = config["train_args"]["lr"] * batch_size

# Decide which device to use
device = (
    torch.device('cuda:0') if torch.cuda.is_available()
    else torch.device("cpu")
)
print("Using device:", device)

# Model configuration flags
mask_conditioning = config["general_args"]["mask_conditioning"]
class_conditioning = config["general_args"]["class_conditioning"]

# random seed for reproducibility
seed = config["general_args"].get("seed", None)
if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create one Generator for DataLoader shuffling:
    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)
else:
    dl_gen = None

# Build model with condition
model_cfg = config["model_args"].copy()
model_cfg["mask_conditioning"] = config["general_args"]["mask_conditioning"]
model = build_model(model_cfg, device=device)

# Load sampling params from config
da = config["data_args"]
# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load the latest checkpoint if available
start_epoch, loaded_config = load_checkpoint(
    model, optimizer, checkpoint_dir=latest_ckpt_dir, device=device, valid_only=False
)

# Ensure model is in float32
model = model.float()

# Define path object (scheduler included)
path = AffineProbPath(scheduler=CondOTScheduler())
solver_config = config["solver_args"]


print("High-quality sampling function")
def high_quality_sampling(model, x_init, mri_patch, cond_tensor, device, num_steps=50):
    """
    Multi-step sampling for CT synthesis using flow matching.
    
    Args:
        model: Trained flow matching model
        x_init: Initial noise [B, C, D, H, W]
        mri_patch: MRI conditioning [B, C, D, H, W]
        cond_tensor: Class conditioning [B, 3] (one-hot AB/HN/TH)
        device: Computation device
        num_steps: Sampling steps (default: 50)
    
    Returns:
        Predicted CT image tensor
    """
    model.eval()
    
    # Create time grid for sampling
    t_grid = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float32)
    
    # Initialize with noise
    x_t = x_init.clone()
    
    with torch.no_grad():
        for i in range(num_steps - 1):
            # Current time step
            t = t_grid[i].unsqueeze(0).expand(x_t.shape[0])
            
            # Ensure cond_tensor has the correct shape for the model: [B, 1, 3]
            # The model expects [B, 1, cross_attention_dim] where cross_attention_dim=3
            if cond_tensor.dim() == 2:
                cond_tensor = cond_tensor.unsqueeze(1).float()  # [B, 3] -> [B, 1, 3]
            
            # Get velocity prediction
            v_pred = model(x=x_t, t=t, cond=cond_tensor, masks=mri_patch)
            
            # Time step size
            dt = t_grid[i+1] - t_grid[i]
            
            # Update x_t using Euler method
            x_t = x_t + dt * v_pred
    
    return x_t

print("Utility functions")
def calculate_batch_metrics(pred_patch, ct_patch, mask_patch, scale, ct_min, ct_max, npy_root, seed):
    """
    Calculate MAE, PSNR, SSIM for a single patch.
    
    Args:
        pred_patch: Predicted CT patch [D, H, W] (normalized)
        ct_patch: Ground truth CT patch [D, H, W] (normalized)
        mask_patch: Validity mask [D, H, W] (1=valid, 0=padding)
        scale: Scaling method for denormalization
        ct_min/max: Original CT value range
        npy_root: Path for uniform scaling files
        seed: Random seed
    
    Returns:
        (mae, psnr, ssim) - metrics on valid regions only
    """
    # Convert to numpy if needed
    if isinstance(pred_patch, torch.Tensor):
        pred_patch = pred_patch.cpu().numpy()
    if isinstance(ct_patch, torch.Tensor):
        ct_patch = ct_patch.cpu().numpy()
    if isinstance(mask_patch, torch.Tensor):
        mask_patch = mask_patch.cpu().numpy()
    
    # Remove channel dimension if present
    if pred_patch.ndim == 4 and pred_patch.shape[0] == 1:
        pred_patch = pred_patch[0]  # [D, H, W]
    if ct_patch.ndim == 4 and ct_patch.shape[0] == 1:
        ct_patch = ct_patch[0]  # [D, H, W]
    if mask_patch.ndim == 4 and mask_patch.shape[0] == 1:
        mask_patch = mask_patch[0]  # [D, H, W]
    
    # Create binary mask for valid regions
    valid_mask = mask_patch > 0.5
    
    # Skip if no valid regions
    if not valid_mask.any():
        return np.nan, np.nan, np.nan
    
    # Denormalize both patches to original HU range
    pred_denorm = denormalize_ct(pred_patch.copy(), ct_min, ct_max, scale, npy_root, seed)
    ct_denorm = denormalize_ct(ct_patch.copy(), ct_min, ct_max, scale, npy_root, seed)
    
    # Apply padding value (-1024) to non-valid regions
    pred_denorm[~valid_mask] = CT_LOWER
    ct_denorm[~valid_mask] = CT_LOWER
    
    # Calculate metrics only on valid regions
    pred_valid = pred_denorm[valid_mask]
    ct_valid = ct_denorm[valid_mask]
    
    # MAE
    mae = np.mean(np.abs(pred_valid - ct_valid))
    
    # PSNR
    mse = np.mean((pred_valid - ct_valid) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_val = np.max(ct_valid)
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    
    # SSIM (calculate on middle slice)
    mid_slice = pred_denorm.shape[0] // 2
    
    # Get middle slices
    pred_slice = pred_denorm[mid_slice]
    ct_slice = ct_denorm[mid_slice]
    mask_slice = valid_mask[mid_slice]
    
    # Find valid region in the slice
    ys, xs = np.where(mask_slice)
    if ys.size > 0:
        ymin, ymax = ys.min(), ys.max() + 1
        xmin, xmax = xs.min(), xs.max() + 1
        pred_crop = pred_slice[ymin:ymax, xmin:xmax]
        ct_crop = ct_slice[ymin:ymax, xmin:xmax]
        
        if min(pred_crop.shape) >= 7:  # SSIM needs minimum size
            data_range = np.max(ct_crop) - np.min(ct_crop)
            if data_range > 0:
                ssim_val = ssim(pred_crop, ct_crop, data_range=data_range)
            else:
                ssim_val = np.nan
        else:
            ssim_val = np.nan
    else:
        ssim_val = np.nan
    
    return mae, psnr, ssim_val


def denormalize_ct(ct_arr, ct_min, ct_max, scale, npy_root, seed):
    """
    Denormalize CT values back to original HU range.
    
    Args:
        ct_arr: Normalized CT array
        ct_min/max: Original CT value range
        scale: Scaling method ('linear', 'sigmoid', 'sigmoid2', 'uniform', 'uniform2')
        npy_root: Path for uniform scaling files
        seed: Random seed
    
    Returns:
        Denormalized CT array in HU range
    """
    if scale == 'linear':
        lower, upper = CT_LOWER, CT_UPPER 
        midpoint = (upper + lower) / 2.0
        range_half = (upper - lower) / 2.0
        ct_arr = ct_arr * range_half + midpoint
    elif scale == 'sigmoid':
        # Inverse sigmoid transformation
        lower, upper = CT_LOWER, CT_UPPER 
        p_low = 0.005
        x0 = (upper + lower) / 2.0
        logit_high = np.log((1 - p_low) / p_low)
        k = 2 * logit_high / (upper - lower)
        s = (ct_arr + 1) / 2
        ct_arr = x0 + (1/k) * np.log(s / (1 - s))
    elif scale == 'sigmoid2':
        ct_sigmoid2 = Sigmoid2()
        ct_arr = ct_sigmoid2.inverse(ct_arr)
    elif scale == 'uniform':
        ct_qt = QT(file_path=os.path.join(npy_root, "ct_mask1.npy"), seed=seed)
        ct_arr = ct_qt.inverse(ct_arr)
    elif scale == 'uniform2':
        ct_gbdt = GBoost(file_path=os.path.join(npy_root, "ct_mask1.npy"), seed=seed)
        ct_arr = ct_gbdt.inverse(ct_arr)
    return ct_arr


# ---- Inserted helpers for sliding window ----
def reorient_to_RAS(img: sitk.Image) -> sitk.Image:
    """
    Reorient image to RAS coordinate system.
    
    Args:
        img: Input SimpleITK image
    
    Returns:
        Reoriented image in RAS
    """
    dicom_orient = sitk.DICOMOrientImageFilter()
    dicom_orient.SetDesiredCoordinateOrientation("RAS")
    return dicom_orient.Execute(img)

def scale_ct_array(arr: np.ndarray, scale: str, npy_root: str, seed):
    """
    Normalize CT array to model input range.
    
    Args:
        arr: Raw CT array in HU values
        scale: Scaling method ('linear', 'sigmoid', 'sigmoid2', 'uniform', 'uniform2')
        npy_root: Path for uniform scaling files
        seed: Random seed
    
    Returns:
        Normalized CT array
    """
    arr = arr.astype(np.float32)
    if scale == 'linear':
        lower, upper = CT_LOWER, CT_UPPER
        midpoint = (upper + lower) / 2.0
        range_half = (upper - lower) / 2.0
        arr = np.clip(arr, lower, upper)
        arr = (arr - midpoint) / range_half
    elif scale == 'sigmoid':
        lower, upper = CT_LOWER, CT_UPPER
        arr = np.clip(arr, lower, upper)
        p_low = 0.005
        x0 = (upper + lower) / 2.0
        logit_high = np.log((1 - p_low) / p_low)
        k = 2 * logit_high / (upper - lower)
        s = 1 / (1 + np.exp(-k * (arr - x0)))
        arr = 2 * s - 1
    elif scale == 'sigmoid2':
        ct_sigmoid2 = Sigmoid2()
        arr = ct_sigmoid2.forward(arr)
    elif scale == 'uniform':
        ct_qt = QT(file_path=os.path.join(npy_root, "ct_mask1.npy"), seed=seed)
        arr = ct_qt.forward(arr)
    elif scale == 'uniform2':
        ct_gbdt = GBoost(file_path=os.path.join(npy_root, "ct_mask1.npy"), seed=seed)
        arr = ct_gbdt.forward(arr)
    return arr

def scale_mri_array(arr: np.ndarray, scale_mask: str, npy_root: str, seed):
    """
    Normalize MRI array to model input range.
    
    Args:
        arr: Raw MRI array
        scale_mask: Scaling method ('linear', 'sigmoid', 'uniform', 'uniform2')
        npy_root: Path for uniform scaling files
        seed: Random seed
    
    Returns:
        Normalized MRI array
    """
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0, MRI_UPPER)
    if scale_mask == 'linear':
        arr = (arr - (MRI_UPPER / 2)) / (MRI_UPPER / 2)
    elif scale_mask == 'sigmoid':
        lower, upper = 0.0, MRI_UPPER
        p_low = 0.005
        x0 = (upper + lower) / 2.0
        k = -np.log((1/p_low) - 1) / ((upper - lower) / 2)
        arr = 2.0 / (1.0 + np.exp(-k * (arr - x0))) - 1.0
        arr = arr * 0.99
    elif scale_mask == 'uniform':
        mri_qt = QT(file_path=os.path.join(npy_root, "mri_mask1.npy"), seed=seed)
        arr = mri_qt.forward(arr)
    elif scale_mask == 'uniform2':
        mri_gbdt = GBoost(file_path=os.path.join(npy_root, "mri_mask1.npy"), seed=seed)
        arr = mri_gbdt.forward(arr)
    return arr


def save_inference_result(output_volume, output_dir, exp_name, original_shape, original_image=None):
    """
    Save predicted CT volume as MHA file with proper metadata.
    
    Args:
        output_volume: Predicted CT volume as numpy array
        output_dir: Output directory path
        exp_name: Experiment name for filename
        original_shape: Original volume dimensions
        original_image: Original image for metadata preservation (optional)
    
    Returns:
        Path to saved output file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ct_{exp_name}.mha")
    mask_path = os.path.join(output_dir, "mask.mha")
    original_mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(original_mask)
    output_volume[mask_array == CT_LOWER] = CT_LOWER

    # If we have the original image, resample output_volume to match its orientation and grid
    if original_image is not None:
        # Prepare a RAS-oriented reference image from the original
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation("RAS")
        original_ras = orienter.Execute(original_image)

        # Create a SimpleITK image from output_volume and copy RAS metadata
        output_ras_img = sitk.GetImageFromArray(output_volume)
        output_ras_img.CopyInformation(original_ras)

        # Resample to match the original image's orientation and grid
        output_image = sitk.Resample(
            output_ras_img,
            original_image,
            sitk.Transform(),
            sitk.sitkLinear,
            CT_LOWER,
            sitk.sitkFloat32
        )
    else:
        output_image = sitk.GetImageFromArray(output_volume)
        output_image.SetSpacing([1.0, 1.0, 1.0])
        output_image.SetOrigin([0.0, 0.0, 0.0])
    sitk.WriteImage(output_image, output_path)
    print(f"Inference result saved to: {output_path}")
    print(f"Output volume shape: {output_volume.shape}")
    print(f"Output volume range: [{np.min(output_volume):.2f}, {np.max(output_volume):.2f}]")
    return output_path

def calculate_metrics(original_ct_path, predicted_ct_path, mask_path=None):
    """
    Calculate MAE, PSNR, SSIM between CT images.
    
    Args:
        original_ct_path: Path to ground truth CT file
        predicted_ct_path: Path to predicted CT file
        mask_path: Optional path to mask file
    
    Returns:
        (mae, psnr, ssim) - image quality metrics
    """
    # Load images
    original_ct = sitk.ReadImage(original_ct_path)
    predicted_ct = sitk.ReadImage(predicted_ct_path)
    
    # Convert to numpy arrays
    original_array = sitk.GetArrayFromImage(original_ct)
    predicted_array = sitk.GetArrayFromImage(predicted_ct)
    
    # Load mask if provided
    if mask_path:
        mask_image = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_image)
        # Only calculate metrics on masked regions
        mask_bool = mask_array > 0
        original_array = original_array[mask_bool]
        predicted_array = predicted_array[mask_bool]
    
    # Calculate MAE
    mae = np.mean(np.abs(original_array - predicted_array))
    
    # Calculate PSNR
    mse = np.mean((original_array - predicted_array) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_val = np.max(original_array)
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    
    # Calculate SSIM
    # Reshape for SSIM calculation (SSIM expects 2D images)
    if len(original_array.shape) == 3:
        # Calculate SSIM for each slice and average
        ssim_values = []
        for i in range(original_array.shape[0]):
            ssim_val = ssim(original_array[i], predicted_array[i], 
                           data_range=np.max(original_array[i]) - np.min(original_array[i]))
            ssim_values.append(ssim_val)
        ssim_score = np.mean(ssim_values)
    else:
        ssim_score = ssim(original_array, predicted_array, 
                         data_range=np.max(original_array) - np.min(original_array))
    
    return mae, psnr, ssim_score


# %% Run inference (MONAI Sliding Window)
print("Run inference with MONAI sliding window")

# Load volumes
ct_path = os.path.join(inference_path, "ct.mha")
mri_path = os.path.join(inference_path, "mr.mha")
mask_path = os.path.join(inference_path, "mask.mha")

original_ct_image = sitk.ReadImage(ct_path)
original_mri_image = sitk.ReadImage(mri_path)
original_mask_image = sitk.ReadImage(mask_path)

# Reorient to RAS for processing
ct_ras = reorient_to_RAS(original_ct_image)
mri_ras = reorient_to_RAS(original_mri_image)
mask_ras = reorient_to_RAS(original_mask_image)

ct_arr = sitk.GetArrayFromImage(ct_ras).astype(np.float32)
mri_arr = sitk.GetArrayFromImage(mri_ras).astype(np.float32)
mask_arr = sitk.GetArrayFromImage(mask_ras).astype(np.float32)

original_shape = ct_arr.shape  # [Z, Y, X]
print(f"Original volume shape (ZYX): {original_shape}")

# Statistics for denorm/metrics
ct_max = float(np.max(ct_arr))
ct_min = float(np.min(ct_arr))
mri_max = float(np.max(mri_arr))

# Apply scaling (match training)
ct_scaled = scale_ct_array(ct_arr.copy(), da["scale"], da["npy_root"], seed)
mri_scaled = scale_mri_array(mri_arr.copy(), da["scale_mask"], da["npy_root"], seed)

# Tensors: shape to [B, C, Z, Y, X]
ct_t = torch.from_numpy(ct_scaled)[None, None]
mri_t = torch.from_numpy(mri_scaled)[None, None]
mask_t = torch.from_numpy(mask_arr.astype(np.float32))[None, None]

# Full-volume random noise as starting point
x_noise = torch.randn_like(mri_t, dtype=torch.float32)

# Stack channels so each window contains everything needed by the predictor
# channels: [0]=noise, [1]=mri, [2]=mask
inputs_stack = torch.cat([x_noise, mri_t, mask_t], dim=1)  # [1,3,Z,Y,X]

# Class conditioning from folder name
if 'AB' in inference_path:
    lbl = np.array([1, 0, 0], dtype=np.float32)
elif 'HN' in inference_path:
    lbl = np.array([0, 1, 0], dtype=np.float32)
elif 'TH' in inference_path:
    lbl = np.array([0, 0, 1], dtype=np.float32)
else:
    lbl = np.array([0, 1, 0], dtype=np.float32)
cond_tensor_full = torch.from_numpy(lbl)[None, :]  # [1,3]

num_sampling_steps = SAMPLING_STEPS[SAMPLING_QUALITY]

def predictor(patch: torch.Tensor) -> torch.Tensor:
    """
    Process patch during sliding window inference.
    
    Args:
        patch: Input patch [N, 3, z, y, x] with channels [noise, mri, mask]
    
    Returns:
        Predicted CT patch [N, 1, z, y, x]
    """
    # patch: [N, 3, z, y, x]
    x_init = patch[:, 0:1].to(device, dtype=torch.float32)
    mri_p  = patch[:, 1:2].to(device, dtype=torch.float32)
    mask_p = patch[:, 2:3].to(device, dtype=torch.float32)
    cond   = cond_tensor_full.to(device, dtype=torch.float32).expand(x_init.shape[0], -1)
    out = high_quality_sampling(
        model=model,
        x_init=x_init,
        mri_patch=mri_p,
        cond_tensor=cond,
        device=device,
        num_steps=num_sampling_steps,
    )
    # Zero outside body to stabilize blending
    out = out * (mask_p > 0.5)
    return out.detach().to(x_init.dtype)

roi_size = (16, 128, 128)
overlap = 0.5  # half-patch, nnU-Net style
print(f"Sliding window roi_size={roi_size}, overlap={overlap}, mode=gaussian, steps={num_sampling_steps}")

pred_norm = sliding_window_inference(
    inputs=inputs_stack,                # [1,3,Z,Y,X] on CPU
    roi_size=roi_size,
    sw_batch_size=2,
    predictor=predictor,
    overlap=overlap,
    mode="gaussian",                   # center-heavy blending
    sigma_scale=0.125,
    padding_mode="reflect",
    cval=0.0,
    sw_device=device,                  # run patches on GPU
    device=torch.device("cpu"),        # stitch on CPU to reduce VRAM
    progress=True,
)  # -> [1,1,Z,Y,X]

# Move to CPU numpy and denormalize
pred_norm = pred_norm.cpu().numpy()[0, 0]
pred_denorm = denormalize_ct(pred_norm.copy(), ct_min, ct_max, da["scale"], da["npy_root"], seed)

# Outside-body to CT_LOWER
pred_denorm[mask_arr <= 0.5] = CT_LOWER

# Save with original orientation metadata
output_volume = pred_denorm.astype(np.float32)
output_path = save_inference_result(output_volume, inference_path, exp_name, original_shape, original_ct_image)

# %% Calculate and display metrics
print("Calculate and display metrics")
original_ct_path = os.path.join(inference_path, "ct.mha")
predicted_ct_path = output_path
mask_path = os.path.join(inference_path, "mask.mha")
mae_full, psnr_full, ssim_score_full = calculate_metrics(original_ct_path, predicted_ct_path, mask_path)
print(f"\nFinal Summary:")
print(f"Full Image - MAE: {mae_full:.4f}, PSNR: {psnr_full:.4f} dB, SSIM: {ssim_score_full:.4f}")

# %% Plot comparison (similar to inferer_ver3)
print("Plot comparison")
# Load images for visualization
ct_test21 = sitk.ReadImage(predicted_ct_path)
ct = sitk.ReadImage(original_ct_path)
mask_img = sitk.ReadImage(mask_path)

# Use saved volumes directly (they already match the original orientation)
ct_test21_arr = sitk.GetArrayFromImage(ct_test21)
ct_arr       = sitk.GetArrayFromImage(ct)
mask_arr     = sitk.GetArrayFromImage(mask_img)

# Use correct anatomical axes: [z, y, x]
z_pic = 50  # axial slice index
y_pic = 100 # coronal slice index
x_pic = 150 # sagittal slice index

# Axial (z): [z, :, :]
axial_vmin = min(ct_arr[z_pic, :, :].min(), ct_test21_arr[z_pic, :, :].min())
axial_vmax = max(ct_arr[z_pic, :, :].max(), ct_test21_arr[z_pic, :, :].max())
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
im1 = plt.imshow(ct_test21_arr[z_pic, :, :], cmap="gray", vmin=axial_vmin, vmax=axial_vmax)
plt.title(f"Predicted CT (axial z={z_pic})")
plt.axis('off')
cbar1 = plt.colorbar(im1, fraction=0.046, pad=0.04)
cbar1.set_label('HU')
plt.subplot(1, 3, 2)
im2 = plt.imshow(ct_arr[z_pic, :, :], cmap="gray", vmin=axial_vmin, vmax=axial_vmax)
plt.title(f"Original CT (axial z={z_pic})")
plt.axis('off')
cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
cbar2.set_label('HU')
plt.subplot(1, 3, 3)
im3 = plt.imshow(mask_arr[z_pic, :, :], cmap="gray")
plt.title(f"Mask (axial z={z_pic})")
plt.axis('off')
plt.colorbar(im3, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Coronal (y): [:, y, :]
coronal_vmin = min(ct_arr[:, y_pic, :].min(), ct_test21_arr[:, y_pic, :].min())
coronal_vmax = max(ct_arr[:, y_pic, :].max(), ct_test21_arr[:, y_pic, :].max())
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
im1 = plt.imshow(ct_test21_arr[:, y_pic, :], cmap="gray", vmin=coronal_vmin, vmax=coronal_vmax)
plt.title(f"Predicted CT (coronal y={y_pic})")
plt.axis('off')
cbar1 = plt.colorbar(im1, fraction=0.046, pad=0.04)
cbar1.set_label('HU')
plt.subplot(1, 3, 2)
im2 = plt.imshow(ct_arr[:, y_pic, :], cmap="gray", vmin=coronal_vmin, vmax=coronal_vmax)
plt.title(f"Original CT (coronal y={y_pic})")
plt.axis('off')
cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
cbar2.set_label('HU')
plt.subplot(1, 3, 3)
im3 = plt.imshow(mask_arr[:, y_pic, :], cmap="gray")
plt.title(f"Mask (coronal y={y_pic})")
plt.axis('off')
plt.colorbar(im3, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Sagittal (x): [:, :, x]
sagittal_vmin = min(ct_arr[:, :, x_pic].min(), ct_test21_arr[:, :, x_pic].min())
sagittal_vmax = max(ct_arr[:, :, x_pic].max(), ct_test21_arr[:, :, x_pic].max())
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
im1 = plt.imshow(ct_test21_arr[:, :, x_pic], cmap="gray", vmin=sagittal_vmin, vmax=sagittal_vmax)
plt.title(f"Predicted CT (sagittal x={x_pic})")
plt.axis('off')
cbar1 = plt.colorbar(im1, fraction=0.046, pad=0.04)
cbar1.set_label('HU')
plt.subplot(1, 3, 2)
im2 = plt.imshow(ct_arr[:, :, x_pic], cmap="gray", vmin=sagittal_vmin, vmax=sagittal_vmax)
plt.title(f"Original CT (sagittal x={x_pic})")
plt.axis('off')
cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
cbar2.set_label('HU')
plt.subplot(1, 3, 3)
im3 = plt.imshow(mask_arr[:, :, x_pic], cmap="gray")
plt.title(f"Mask (sagittal x={x_pic})")
plt.axis('off')
plt.colorbar(im3, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Show GPU usage
print("\n" + "="*50)
print("GPU USAGE AFTER INFERENCE:")
print("="*50)
os.system("nvidia-smi")





# %%
