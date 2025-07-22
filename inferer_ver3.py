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
from torch.utils.tensorboard import SummaryWriter
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
config_path = "/media/prajbori/sda/private/github/proj_synthrad/MOTFM3D/exp_configs/test20-testseed1.yaml"
inference_path = "/media/prajbori/sda/private/dataset/proj_synthrad/training/synthRAD2025_Task1_Train_D/Task1/HN_x/1HND001"
latest_ckpt_dir = "/media/prajbori/sda/private/github/proj_synthrad/MOTFM3D/checkpoints/test20-testseed1/latest"

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

print("Inference Dataset Class")
class MhaDataset_infer(Dataset):
    """Dataset for inference with overlapping patches to reconstruct full volume."""
    def __init__(self, 
                 root_dir,
                 patch_size=(16, 128, 128),
                 overlap=OVERLAP_SIZE,  # overlap in each dimension
                 scale='linear',
                 scale_mask='linear',
                 npy_root=None,
                 seed=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.overlap = OVERLAP_SIZE
        self.scale = scale
        self.scale_mask = scale_mask
        self.npy_root = npy_root
        self.seed = seed
        
        # Load the single volume for inference
        ct_path = os.path.join(root_dir, "ct.mha")
        mri_path = os.path.join(root_dir, "mr.mha")
        mask_path = os.path.join(root_dir, "mask.mha")
        
        print(f"Loading volume from {root_dir}...")
        
        # Load original images to preserve orientation
        self.original_ct_image = sitk.ReadImage(ct_path)
        self.original_mri_image = sitk.ReadImage(mri_path)
        self.original_mask_image = sitk.ReadImage(mask_path)
        
        # Load and preprocess the volume (reorient to RAS for processing)
        self.ct_arr = sitk.GetArrayFromImage(
            self.reorient_to_RAS(self.original_ct_image)
        ).astype(np.float32)
        self.mri_arr = sitk.GetArrayFromImage(
            self.reorient_to_RAS(self.original_mri_image)
        ).astype(np.float32)
        self.mask_arr = sitk.GetArrayFromImage(
            self.reorient_to_RAS(self.original_mask_image)
        ).astype(np.float32)
        
        # Store original info
        self.ct_max = np.max(self.ct_arr)
        self.ct_min = np.min(self.ct_arr)
        self.mri_max = np.max(self.mri_arr)
        
        # Apply scaling
        self._apply_scaling()
        
        # Generate overlapping patches
        self.patches = self._generate_overlapping_patches()
        
        print(f"Generated {len(self.patches)} overlapping patches")
        print(f"Volume shape: {self.ct_arr.shape}")
        print(f"Patch size: {patch_size}")
        print(f"Overlap: {OVERLAP_SIZE}")
    
    def _apply_scaling(self):
        """Apply the same scaling as in training."""
        # CT scaling
        if self.scale == 'linear':
            lower, upper = CT_LOWER, CT_UPPER 
            midpoint = (upper + lower) / 2.0
            range_half = (upper - lower) / 2.0
            self.ct_arr = np.clip(self.ct_arr, lower, upper)
            self.ct_arr = (self.ct_arr - midpoint) / range_half
        elif self.scale == 'sigmoid':
            lower, upper = CT_LOWER, CT_UPPER 
            self.ct_arr = np.clip(self.ct_arr, lower, upper)
            p_low = 0.005
            x0 = (upper + lower) / 2.0
            logit_high = np.log((1 - p_low) / p_low)
            k = 2 * logit_high / (upper - lower)
            s = 1 / (1 + np.exp(-k * (self.ct_arr - x0)))
            self.ct_arr = 2 * s - 1
        elif self.scale == 'sigmoid2':
            ct_sigmoid2 = Sigmoid2()
            self.ct_arr = ct_sigmoid2.forward(self.ct_arr)
        elif self.scale == 'uniform':
            ct_qt = QT(file_path=os.path.join(self.npy_root, "ct_mask1.npy"), seed=self.seed)
            self.ct_arr = ct_qt.forward(self.ct_arr)
        elif self.scale == 'uniform2':
            ct_gbdt = GBoost(file_path=os.path.join(self.npy_root, "ct_mask1.npy"), seed=self.seed)
            self.ct_arr = ct_gbdt.forward(self.ct_arr)
        
        # MRI scaling
        self.mri_arr = np.clip(self.mri_arr, 0, MRI_UPPER)
        if self.scale_mask == 'linear':
            self.mri_arr = (self.mri_arr - (MRI_UPPER / 2)) / (MRI_UPPER / 2)
        elif self.scale_mask == 'sigmoid':
            lower, upper = 0.0, MRI_UPPER
            p_low = 0.005
            x0 = (upper + lower) / 2.0
            k = -np.log((1/p_low) - 1) / ((upper - lower) / 2)
            self.mri_arr = 2.0 / (1.0 + np.exp(-k * (self.mri_arr - x0))) - 1.0
            self.mri_arr = self.mri_arr * 0.99
        elif self.scale_mask == 'uniform':
            mri_qt = QT(file_path=os.path.join(self.npy_root, "mri_mask1.npy"), seed=self.seed)
            self.mri_arr = mri_qt.forward(self.mri_arr)
        elif self.scale_mask == 'uniform2':
            mri_gbdt = GBoost(file_path=os.path.join(self.npy_root, "mri_mask1.npy"), seed=self.seed)
            self.mri_arr = mri_gbdt.forward(self.mri_arr)
    
    def _generate_overlapping_patches(self):
        """Generate overlapping patches covering the entire volume."""
        H, W, D = self.ct_arr.shape
        ph, pw, pd = self.patch_size
        oh, ow, od = self.overlap
        
        # Calculate step sizes
        step_h = ph - oh
        step_w = pw - ow
        step_d = pd - od
        
        patches = []
        
        # Generate patches with simple step-based approach
        for h in range(0, H, step_h):
            for w in range(0, W, step_w):
                for d in range(0, D, step_d):
                    patches.append({
                        'coords': (h, w, d),
                        'size': (ph, pw, pd)
                    })
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        start_h, start_w, start_d = patch_info['coords']
        size_h, size_w, size_d = patch_info['size']
        
        # Extract patches (may extend beyond volume boundaries)
        end_h = min(start_h + size_h, self.ct_arr.shape[0])
        end_w = min(start_w + size_w, self.ct_arr.shape[1])
        end_d = min(start_d + size_d, self.ct_arr.shape[2])
        
        ct_patch = self.ct_arr[start_h:end_h, start_w:end_w, start_d:end_d]
        mri_patch = self.mri_arr[start_h:end_h, start_w:end_w, start_d:end_d]
        mask_patch = self.mask_arr[start_h:end_h, start_w:end_w, start_d:end_d]
        
        # Pad if necessary to match patch_size
        if ct_patch.shape != self.patch_size:
            pad_h = self.patch_size[0] - ct_patch.shape[0]
            pad_w = self.patch_size[1] - ct_patch.shape[1]
            pad_d = self.patch_size[2] - ct_patch.shape[2]
            
            ct_patch = np.pad(ct_patch, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            mri_patch = np.pad(mri_patch, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
        
        # Convert to tensors with explicit float32 dtype
        ct_tensor = torch.from_numpy(ct_patch).unsqueeze(0).float()  # [1, D, H, W]
        mri_tensor = torch.from_numpy(mri_patch).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0).float()
        
        # Determine class label based on directory name
        if 'AB' in self.root_dir:
            lbl = np.array([1, 0, 0])
        elif 'HN' in self.root_dir:
            lbl = np.array([0, 1, 0])
        elif 'TH' in self.root_dir:
            lbl = np.array([0, 0, 1])
        else:
            lbl = np.array([0, 0, 0])  # Default to HN
        
        class_tensor = torch.tensor(lbl, dtype=torch.long)
        
        return {
            "images": ct_tensor,
            "masks": mri_tensor,
            "masks_": mask_tensor,
            "classes": class_tensor,
            "coords": torch.tensor(patch_info['coords'], dtype=torch.long),
            "size": torch.tensor(patch_info['size'], dtype=torch.long),
            "ct_max": self.ct_max,
            "ct_min": self.ct_min,
            "mri_max": self.mri_max,
        }
    
    def reorient_to_RAS(self, img: sitk.Image) -> sitk.Image:
        """Take any SimpleITK Image and return an image whose axes are labeled R,A,S."""
        dicom_orient = sitk.DICOMOrientImageFilter()
        dicom_orient.SetDesiredCoordinateOrientation("RAS")
        return dicom_orient.Execute(img)
    
    def get_original_shape(self):
        """Return the original volume shape."""
        return self.ct_arr.shape
    
    def get_original_ct_image(self):
        """Return the original CT image for orientation preservation."""
        return self.original_ct_image

print("High-quality sampling function")
def high_quality_sampling(model, x_init, mri_patch, cond_tensor, device, num_steps=50):
    """
    High-quality multi-step sampling for better inference quality.
    
    Args:
        model: The trained model
        x_init: Initial noise [B, C, D, H, W]
        mri_patch: MRI conditioning [B, C, D, H, W]
        cond_tensor: Class conditioning [B, 1]
        device: Device to run on
        num_steps: Number of sampling steps (higher = better quality, slower)
    
    Returns:
        Final predicted image
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
    Calculate MAE, PSNR, SSIM for a single batch/patch.
    
    Args:
        pred_patch: Predicted CT patch [D, H, W] (normalized)
        ct_patch: Original CT patch [D, H, W] (normalized) 
        mask_patch: Mask patch [D, H, W] (1 for valid regions, 0 for padding)
        scale: Scaling method used
        ct_min, ct_max: Original CT min/max values
        npy_root, seed: For uniform scaling methods
    
    Returns:
        mae, psnr, ssim: Metrics calculated only on valid regions (mask > 0)
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
    """Denormalize CT values back to original range."""
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


def save_inference_result(output_volume, output_dir, exp_name, original_shape, original_image=None):
    """Save the inference result as MHA file, matching the original orientation."""
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
    """Calculate MAE, PSNR, SSIM between original and predicted CT images."""
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

print("Inference pipeline")
def inference_pipeline(model, dataset, device, solver_config, exp_name, output_dir, num_sampling_steps=50):
    """Run inference on the dataset and save results."""
    model.eval()
    
    # Get original volume shape
    original_shape = dataset.get_original_shape()
    print(f"Original volume shape: {original_shape}")
    print(f"Using {num_sampling_steps} sampling steps for {SAMPLING_QUALITY} quality")
    
    # Initialize output volume
    output_volume = np.zeros(original_shape, dtype=np.float32)
    count_volume = np.zeros(original_shape, dtype=np.float32)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Initialize metrics tracking
    batch_metrics = []
    running_mae = []
    running_psnr = []
    running_ssim = []
    
    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            # Move to device and ensure float32 dtype
            ct_patch = batch["images"].to(device, dtype=torch.float32)
            mri_patch = batch["masks"].to(device, dtype=torch.float32)
            mask_patch = batch["masks_"].to(device, dtype=torch.float32)  # Original mask for metrics
            classes_batch = batch["classes"].to(device)
            coords = batch["coords"][0]  # Get first (and only) item from batch
            size = batch["size"][0]  # Get first (and only) item from batch

            # Run high-quality inference
            x_0 = torch.randn_like(ct_patch, dtype=torch.float32)
            cond_tensor = classes_batch.unsqueeze(1).float()

            # Use high-quality multi-step sampling instead of single-step
            pred_patch = high_quality_sampling(
                model=model,
                x_init=x_0,
                mri_patch=mri_patch,
                cond_tensor=cond_tensor,
                device=device,
                num_steps=num_sampling_steps
            )

            # Convert prediction and mask to numpy for post-processing
            pred_patch_np = pred_patch.cpu().numpy()[0, 0]  # Remove batch and channel dimensions
            mask_patch_np = mask_patch.cpu().numpy()[0, 0]  # Remove batch and channel dimensions

            # Denormalize pred_patch before adding to output_volume
            pred_patch_np = denormalize_ct(pred_patch_np, dataset.ct_min, dataset.ct_max, dataset.scale, dataset.npy_root, dataset.seed)
            # Post-process: set pred_patch to CT_LOWER where mask==0
            pred_patch_np[mask_patch_np <= 0.5] = CT_LOWER

            # Calculate batch metrics before converting to numpy (use original pred_patch)
            mae, psnr, ssim = calculate_batch_metrics(
                pred_patch=pred_patch[0, 0],  # Remove batch and channel dims
                ct_patch=ct_patch[0, 0],      # Remove batch and channel dims  
                mask_patch=mask_patch[0, 0],  # Remove batch and channel dims
                scale=dataset.scale,
                ct_min=dataset.ct_min,
                ct_max=dataset.ct_max,
                npy_root=dataset.npy_root,
                seed=dataset.seed
            )

            # Store metrics (handle NaN values)
            if not np.isnan(mae) and not np.isnan(psnr) and not np.isnan(ssim):
                batch_metrics.append((mae, psnr, ssim))
                running_mae.append(mae)
                running_psnr.append(psnr)
                running_ssim.append(ssim)
                
                # Calculate running averages
                avg_mae = np.mean(running_mae)
                avg_psnr = np.mean(running_psnr)
                avg_ssim = np.mean(running_ssim)
                
                # Print batch metrics
                print(f"Batch {batch_idx+1:3d}: MAE={mae:6.2f}, PSNR={psnr:6.2f}dB, SSIM={ssim:6.4f} | "
                      f"Avg: MAE={avg_mae:6.2f}, PSNR={avg_psnr:6.2f}dB, SSIM={avg_ssim:6.4f}")
            else:
                print(f"Batch {batch_idx+1:3d}: Skipped (invalid metrics: MAE={mae:.2f}, PSNR={psnr:.2f}, SSIM={ssim:.4f})")

            # Extract coordinates (now they are tensors)
            start_h, start_w, start_d = coords.cpu().numpy()
            size_h, size_w, size_d = size.cpu().numpy()

            # Get the actual patch size that was used (may be smaller due to padding)
            actual_size_h = min(size_h, original_shape[0] - start_h)
            actual_size_w = min(size_w, original_shape[1] - start_w)
            actual_size_d = min(size_d, original_shape[2] - start_d)

            # Add to output volume (only the valid region, not the padded part)
            output_volume[start_h:start_h+actual_size_h, start_w:start_w+actual_size_w, start_d:start_d+actual_size_d] += pred_patch_np[:actual_size_h, :actual_size_w, :actual_size_d]
            count_volume[start_h:start_h+actual_size_h, start_w:start_w+actual_size_w, start_d:start_d+actual_size_d] += 1

            # # Compute blending weights using a 3D Hanning window to reduce seam artifacts
            # weight_h = np.hanning(actual_size_h) if actual_size_h > 1 else np.ones(1, dtype=np.float32)
            # weight_w = np.hanning(actual_size_w) if actual_size_w > 1 else np.ones(1, dtype=np.float32)
            # weight_d = np.hanning(actual_size_d) if actual_size_d > 1 else np.ones(1, dtype=np.float32)
            # weight_patch = weight_h[:, None, None] * weight_w[None, :, None] * weight_d[None, None, :]

            # # Add to output volume (only the valid region, not the padded part) with blending weights
            # output_volume[start_h:start_h+actual_size_h, start_w:start_w+actual_size_w, start_d:start_d+actual_size_d] += pred_patch_np[:actual_size_h, :actual_size_w, :actual_size_d] * weight_patch
            # count_volume[start_h:start_h+actual_size_h, start_w:start_w+actual_size_w, start_d:start_d+actual_size_d] += weight_patch
    
    # Print final summary
    if batch_metrics:
        final_mae = np.mean(running_mae)
        final_psnr = np.mean(running_psnr)
        final_ssim = np.mean(running_ssim)
        print(f"\n{'='*60}")
        print(f"INFERENCE COMPLETE - Patch-Level Metrics:")
        print(f"Processed {len(batch_metrics)} valid batches out of {len(dataloader)} total")
        print(f"Final MAE:  {final_mae:.2f}")
        print(f"Final PSNR: {final_psnr:.2f} dB")
        print(f"Final SSIM: {final_ssim:.4f}")
        print(f"{'='*60}")
    
    # Average overlapping regions
    count_volume = np.maximum(count_volume, 1)  # Avoid division by zero
    output_volume = output_volume / count_volume
    
    # Denormalize the output
    output_volume = denormalize_ct(output_volume, dataset.ct_min, dataset.ct_max, dataset.scale, dataset.npy_root, dataset.seed)
    
    # Save the result with original orientation metadata
    output_path = save_inference_result(output_volume, output_dir, exp_name, original_shape, dataset.get_original_ct_image())
    
    return output_volume, output_path, batch_metrics

# %% Run inference
print("Run inference")

# Create inference dataset
inference_dataset = MhaDataset_infer(
    root_dir=inference_path,
    patch_size=(16, 128, 128),
    overlap=OVERLAP_SIZE,
    scale=da["scale"],
    scale_mask=da["scale_mask"],
    npy_root=da["npy_root"],
    seed=seed
)

# Run inference
output_volume, output_path, batch_metrics = inference_pipeline(
    model=model,
    dataset=inference_dataset,
    device=device,
    solver_config=solver_config,
    exp_name=exp_name,
    output_dir=inference_path,
    num_sampling_steps=SAMPLING_STEPS[SAMPLING_QUALITY]
)

# Check GPU usage after inference
print("\n" + "="*50)
print("GPU USAGE AFTER INFERENCE:")
print("="*50)
os.system("nvidia-smi")

# %% Calculate and display metrics
print("Calculate and display metrics")

# Calculate full image metrics
original_ct_path = os.path.join(inference_path, "ct.mha")
predicted_ct_path = output_path
mask_path = os.path.join(inference_path, "mask.mha")

print("\nCalculating full image metrics...")
mae_full, psnr_full, ssim_score_full = calculate_metrics(original_ct_path, predicted_ct_path, mask_path)

# Calculate patch-level metrics summary
if batch_metrics:
    patch_mae_values = [m[0] for m in batch_metrics]
    patch_psnr_values = [m[1] for m in batch_metrics]
    patch_ssim_values = [m[2] for m in batch_metrics]
    
    patch_mae_mean = np.mean(patch_mae_values)
    patch_psnr_mean = np.mean(patch_psnr_values)
    patch_ssim_mean = np.mean(patch_ssim_values)
    
    patch_mae_std = np.std(patch_mae_values)
    patch_psnr_std = np.std(patch_psnr_values)
    patch_ssim_std = np.std(patch_ssim_values)
    
    print(f"\n{'='*80}")
    print(f"METRICS COMPARISON: PATCH-LEVEL vs FULL IMAGE")
    print(f"{'='*80}")
    print(f"{'Metric':<15} {'Patch-Level (Mean±Std)':<25} {'Full Image':<15} {'Difference':<15}")
    print(f"{'-'*80}")
    print(f"{'MAE':<15} {patch_mae_mean:6.2f}±{patch_mae_std:5.2f}{'':<15} {mae_full:6.2f}{'':<9} {mae_full - patch_mae_mean:+6.2f}")
    print(f"{'PSNR (dB)':<15} {patch_psnr_mean:6.2f}±{patch_psnr_std:5.2f}{'':<15} {psnr_full:6.2f}{'':<9} {psnr_full - patch_psnr_mean:+6.2f}")
    print(f"{'SSIM':<15} {patch_ssim_mean:6.4f}±{patch_ssim_std:5.4f}{'':<15} {ssim_score_full:6.4f}{'':<9} {ssim_score_full - patch_ssim_mean:+6.4f}")
    print(f"{'='*80}")
    
    # Additional statistics
    print(f"\nPatch-level statistics:")
    print(f"  Number of valid patches: {len(batch_metrics)}")
    print(f"  MAE range: [{np.min(patch_mae_values):.2f}, {np.max(patch_mae_values):.2f}]")
    print(f"  PSNR range: [{np.min(patch_psnr_values):.2f}, {np.max(patch_psnr_values):.2f}] dB")
    print(f"  SSIM range: [{np.min(patch_ssim_values):.4f}, {np.max(patch_ssim_values):.4f}]")
    
    # Check for significant differences
    mae_diff_pct = abs(mae_full - patch_mae_mean) / patch_mae_mean * 100
    psnr_diff_pct = abs(psnr_full - patch_psnr_mean) / patch_psnr_mean * 100
    ssim_diff_pct = abs(ssim_score_full - patch_ssim_mean) / patch_ssim_mean * 100
    
    print(f"\nRelative differences (Full vs Patch mean):")
    print(f"  MAE:  {mae_diff_pct:.1f}%")
    print(f"  PSNR: {psnr_diff_pct:.1f}%")
    print(f"  SSIM: {ssim_diff_pct:.1f}%")
    
    if mae_diff_pct > 10 or psnr_diff_pct > 10 or ssim_diff_pct > 10:
        print(f"\n⚠️  WARNING: Large differences detected (>10%) between patch-level and full image metrics!")
        print(f"   This may indicate issues with patch reconstruction or overlapping regions.")
else:
    print(f"\n{'='*80}")
    print(f"METRICS COMPARISON: PATCH-LEVEL vs FULL IMAGE")
    print(f"{'='*80}")
    print(f"No valid patch-level metrics available")
    print(f"Full Image Metrics:")
    print(f"  MAE:  {mae_full:.2f}")
    print(f"  PSNR: {psnr_full:.2f} dB")
    print(f"  SSIM: {ssim_score_full:.4f}")
    print(f"{'='*80}")

print(f"\nFinal Summary:")
print(f"Full Image - MAE: {mae_full:.4f}, PSNR: {psnr_full:.4f} dB, SSIM: {ssim_score_full:.4f}")
if batch_metrics:
    print(f"Patch Mean - MAE: {patch_mae_mean:.4f}, PSNR: {patch_psnr_mean:.4f} dB, SSIM: {patch_ssim_mean:.4f}")

# %% Plot
print("Plot")
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

# Diagnostic: plot a few high-MAE patches
if batch_metrics:
    patch_mae_values = [m[0] for m in batch_metrics]
    high_mae_indices = np.argsort(patch_mae_values)[-3:]
    print("\nDiagnostic: Plotting high-MAE patches (top 3):")
    for idx in high_mae_indices:
        # Get patch info from dataloader
        patch = inference_dataset[idx]
        pred_patch = output_volume[
            patch["coords"][0]:patch["coords"][0]+patch["size"][0],
            patch["coords"][1]:patch["coords"][1]+patch["size"][1],
            patch["coords"][2]:patch["coords"][2]+patch["size"][2]
        ]
        orig_patch = inference_dataset.ct_arr[
            patch["coords"][0]:patch["coords"][0]+patch["size"][0],
            patch["coords"][1]:patch["coords"][1]+patch["size"][1],
            patch["coords"][2]:patch["coords"][2]+patch["size"][2]
        ]
        # Use fixed color scale for normalized range [-1, 1]
        patch_vmin, patch_vmax = -1, 1
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        im1 = plt.imshow(pred_patch[pred_patch.shape[0]//2], cmap='gray', vmin=patch_vmin, vmax=patch_vmax)
        plt.title(f'Pred Patch (idx {idx})')
        plt.axis('off')
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        plt.subplot(1,2,2)
        im2 = plt.imshow(orig_patch[orig_patch.shape[0]//2], cmap='gray', vmin=patch_vmin, vmax=patch_vmax)
        plt.title('Original Patch')
        plt.axis('off')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.suptitle(f'MAE: {patch_mae_values[idx]:.2f}')
        plt.show()
        # Print orientation info
        print(f"Patch {idx} - Original CT direction: {inference_dataset.original_ct_image.GetDirection()}")
        print(f"Patch {idx} - Predicted CT (RAS assumed)")

# %%
