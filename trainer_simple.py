import argparse
import os
import sys
import time
import warnings

# Suppress most warnings for cleaner logs (comment out if debugging is needed)
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random

# Remove TensorBoard logging import
import shutil
from typing import Optional
from torch.utils.tensorboard.writer import SummaryWriter
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import HistGradientBoostingRegressor

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
is_interactive = sys.stdout.isatty()


#======== Dataset classes for MHA files ========
class MhaDataset_small(Dataset):
    """Represents the full dataset of MHA files."""
    def __init__(self, 
                 root_dir,
                 transform=None,
                 n_images=5,
                 n_patches=50,
                 patch_size=(16,128,128),
                 reverse=False,
                 scale='linear',
                 scale_mask='linear',
                 npy_root=None,
                 seed=None):
        self.n_patches = n_patches
        # create a reproducible RNG
        self.rng = np.random.default_rng(seed)
        anatomy_code = os.path.basename(root_dir)
        glob_pattern = f"1{anatomy_code}*"
        all_dirs = sorted(glob.glob(os.path.join(root_dir, glob_pattern)))
        n_max = len(all_dirs)
        self.n_images = min(n_images, n_max)
        # sample directories in a reproducible way
        idx_choices = self.rng.choice(n_max, size=self.n_images, replace=False)
        # NOTE: for the test
        print(idx_choices)
        self.sample_dirs = [all_dirs[i] for i in idx_choices]

        self.transform = transform
        self.patch_size = patch_size
        self.scale = scale
        self.scale_mask = scale_mask
        self.npy_root = npy_root 
        
        # NOTE: since the sampling data is small, we will do the preprocessing here
        self.ct_arrs = []
        self.mask_arrs = []
        self.mri_arrs = []
        self.lbls = []
        self.ct_maxs = []
        self.ct_mins = []
        self.mri_maxs = []
        
        print(f"Initializing MhaDataset_small with {len(self.sample_dirs)} images...")
        start_time = time.time()

        # fitting process
        if self.scale == 'sigmoid2':
            self.ct_sigmoid2 = Sigmoid2()
        elif self.scale == 'uniform':
            lower, upper = -1.0, 1.0
            self.ct_qt = QT(file_path=os.path.join(self.npy_root, "ct_mask1.npy"), seed=seed)
        elif self.scale == 'uniform2':
            self.ct_gbdt = GBoost(file_path=os.path.join(self.npy_root, "ct_mask1.npy"), seed=seed)

        if self.scale_mask == 'uniform':
            lower, upper = -1.0, 1.0
            self.mri_qt = QT(file_path=os.path.join(self.npy_root, "mri_mask1.npy"), seed=seed)
        elif self.scale_mask == 'uniform2':
            self.mri_gbdt = GBoost(file_path=os.path.join(self.npy_root, "mri_mask1.npy"), seed=seed)

        for i, d in enumerate(self.sample_dirs):
            # Calculate progress
            progress = (i / len(self.sample_dirs)) * 100
            elapsed_time = time.time() - start_time
            # Calculate average time per image
            if i > 0:
                avg_time_per_image = elapsed_time / i
                print(f"Processing image {i+1}/{len(self.sample_dirs)}, {progress:.1f}%, avg time per image = {avg_time_per_image:.2f}s")
            else:
                print(f"Processing image {i+1}/{len(self.sample_dirs)}, {progress:.1f}%, time = {elapsed_time:.2f}s")
            
            ct_path   = os.path.join(d, "ct.mha")
            mri_path  = os.path.join(d, "mr.mha")
            mask_path = os.path.join(d, "mask.mha")
            # determine 1-hot class ~ anatomy
            if root_dir.split('/')[-1] == 'AB':
                lbl = np.array([1, 0, 0])
            elif root_dir.split('/')[-1] == 'HN':
                lbl = np.array([0, 1, 0])
            elif root_dir.split('/')[-1] == 'TH':
                lbl = np.array([0, 0, 1])

            # path -> image -> reorient to RAS -> numpy array
            ct_arr   = sitk.GetArrayFromImage(
                self.reorient_to_RAS(sitk.ReadImage(ct_path))
                ).astype(np.float32)
            mask_arr = sitk.GetArrayFromImage(
                self.reorient_to_RAS(sitk.ReadImage(mask_path))
                ).astype(np.float32)
            mri_arr  = sitk.GetArrayFromImage(
                self.reorient_to_RAS(sitk.ReadImage(mri_path))
                ).astype(np.float32)

            # save original information for later use
            ct_max = np.max(ct_arr)
            ct_min = np.min(ct_arr)
            mri_max = np.max(mri_arr)
            
            if self.scale == 'linear':
                # NOTE: linear scale
                lower, upper = -1024.0, 3071.0
                # midpoint and half-range for symmetric [-1,1] mapping
                midpoint = (upper + lower) / 2.0
                range_half = (upper - lower) / 2.0
                ct_arr = np.clip(ct_arr, lower, upper)
                ct_arr = (ct_arr - midpoint) / range_half

            elif self.scale == 'sigmoid':
                # NOTE: sigmoid scale
                # Sigmoid-based normalization mapping HU [-1024,3071] to [-0.99,0.99]
                lower, upper = -1024.0, 3071.0
                ct_arr = np.clip(ct_arr, lower, upper)
                # Desired sigmoid outputs for endpoints: p_low=0.005 -> y_low≈-0.99, p_high=0.995->y_high≈0.99
                p_low = 0.005
                x0 = (upper + lower) / 2.0
                logit_high = np.log((1 - p_low) / p_low)
                k = 2 * logit_high / (upper - lower)
                s = 1 / (1 + np.exp(-k * (ct_arr - x0)))
                ct_arr = 2 * s - 1

            elif self.scale == 'sigmoid2':
                ct_arr = self.ct_sigmoid2.forward(ct_arr)

            elif self.scale == 'uniform': # transform x to U[-a, b]
                ct_arr = self.ct_qt.forward(ct_arr)
            
            elif self.scale == 'uniform2':
                ct_arr = self.ct_gbdt.forward(ct_arr)

            else: # stop and report error
                raise ValueError(f"Unknown scale type: {self.scale}. Choose 'linear', 'sigmoid', 'uniform', 'uniform2', or 'sigmoid2'.")

            # p99 = np.percentile(mri_arr, 99)
            mri_arr = np.clip(mri_arr, 0, 1357)
            if self.scale_mask == 'linear': 
                # mri clip to [0, 1357] and normalize to [-1, 1]
                mri_arr = (mri_arr - (1357 / 2)) / (1357 / 2)
            elif self.scale_mask == 'sigmoid':
                # Use fixed upper=1357 for MRI normalization
                lower, upper = 0.0, 1357.0
                p_low = 0.005
                x0 = (upper + lower) / 2.0
                # Calculate k to achieve desired output range
                k = -np.log((1/p_low) - 1) / ((upper - lower) / 2)
                # Apply sigmoid transformation
                mri_arr = 2.0 / (1.0 + np.exp(-k * (mri_arr - x0))) - 1.0
                # Scale to [-0.99, 0.99] range
                mri_arr = mri_arr * 0.99
            elif self.scale_mask == 'uniform':
                mri_arr = self.mri_qt.forward(mri_arr)
            elif self.scale_mask == 'uniform2':
                mri_arr = self.mri_gbdt.forward(mri_arr)
            else: # stop and report error
                raise ValueError(f"Unknown scale type: {self.scale_mask}. Use 'linear' or 'sigmoid' or 'uniform' or 'uniform2'.")

            # Ensure numpy arrays are float32 to avoid mixed-precision tensors
            ct_arr = ct_arr.astype(np.float32)
            mask_arr = mask_arr.astype(np.float32)
            mri_arr = mri_arr.astype(np.float32)
            # Preserve original arrays for patch extraction
            orig_ct_arr = ct_arr.copy()
            orig_mask_arr = mask_arr.copy()
            orig_mri_arr = mri_arr.copy()

            # crop random patch
            for j in range(self.n_patches):
                # extract random patch
                start_h, start_w, start_d = self.extract_random_patch(orig_ct_arr.shape, self.patch_size)
                ct_patch = orig_ct_arr[start_h:start_h+self.patch_size[0],
                                         start_w:start_w+self.patch_size[1],
                                         start_d:start_d+self.patch_size[2]]
                mask_patch = orig_mask_arr[start_h:start_h+self.patch_size[0],
                                           start_w:start_w+self.patch_size[1],
                                           start_d:start_d+self.patch_size[2]]
                mri_patch  = orig_mri_arr[start_h:start_h+self.patch_size[0],
                                         start_w:start_w+self.patch_size[1],
                                         start_d:start_d+self.patch_size[2]]
                # Skip patch if mask_patch.sum() == 0
                if mask_patch.sum() == 0:
                    print(f"[DEBUG] Skipping patch {j} of image {i}: mask_patch.sum() == 0")
                    continue
                # add to list
                self.ct_arrs.append(ct_patch)
                self.mask_arrs.append(mask_patch)
                self.mri_arrs.append(mri_patch)
                self.lbls.append(lbl)
                self.ct_maxs.append(ct_max)
                self.ct_mins.append(ct_min)
                self.mri_maxs.append(mri_max)

        total_time = time.time() - start_time
        print(f"Dataset initialization complete! Total time: {total_time:.2f}s")
        print(f"Processed {len(self.ct_arrs)} patches from {len(self.sample_dirs)} images")

    def __len__(self):
        return len(self.ct_arrs)
    def __getitem__(self, idx): # full image
        ct_arr   = self.ct_arrs[idx]
        mask_arr = self.mask_arrs[idx]
        mri_arr  = self.mri_arrs[idx]
        lbl      = self.lbls[idx]

        ct_tensor   = torch.from_numpy(ct_arr).unsqueeze(0) # [1, D, H, W] [channel, patch_size]
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0) # [1, D, H, W] [channel, patchsize]
        mri_tensor  = torch.from_numpy(mri_arr).unsqueeze(0)
        class_tensor= torch.tensor(lbl, dtype=torch.long)
        if self.transform:
            ct_tensor, mask_tensor = self.transform(ct_tensor, mask_tensor)
        return {
            "images": ct_tensor,
            "masks":  mri_tensor,
            "masks_":  mask_tensor,
            "classes": class_tensor,
            "ct_max": self.ct_maxs[idx],
            "ct_min": self.ct_mins[idx],
            "mri_max": self.mri_maxs[idx],
        }
    # helper functions
    def reorient_to_RAS(self, img: sitk.Image) -> sitk.Image:
        """
        Take any SimpleITK Image (in LPS, RAS, whatever) and
        return an image whose axes are labeled R,A,S.
        """
        dicom_orient = sitk.DICOMOrientImageFilter()
        dicom_orient.SetDesiredCoordinateOrientation("RAS")  # target orientation
        return dicom_orient.Execute(img)

    def extract_random_patch(self, volume_shape, patch_size):
        """Extract a random patch of size patch_size from a 3D volume."""
        H, W, D = volume_shape
        ph, pw, pd = patch_size
        if H < ph or W < pw or D < pd:
            raise ValueError("Volume shape is smaller than patch size.")
        start_h = self.rng.integers(0, H - ph + 1)
        start_w = self.rng.integers(0, W - pw + 1)
        start_d = self.rng.integers(0, D - pd + 1)
        # patch = volume[start_h:start_h+ph, start_w:start_w+pw, start_d:start_d+pd]
        # print(start_h, start_w, start_d)
        return start_h, start_w, start_d  # patch + its coordinates
    
    def show(self):
        """ return a dictionary of processed data"""
        return {'images': self.ct_arrs,
                'masks_': self.mask_arrs,
                'masks':self.mri_arrs,
                'ct_max': self.ct_maxs,
                'ct_min': self.ct_mins,
                'mri_max': self.mri_maxs,
                'classes': self.lbls,
                'n_data': f'{self.n_images} x {self.n_patches} = {self.n_images*self.n_patches}',
                'patch_size': self.patch_size,
        }
    

#--------done adding------------
def main():
    # Parse arguments and load config
    parser = argparse.ArgumentParser(description="Validate the flow matching model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="exp_configs/test_3090_1.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="some_checkpoints/test20-testseed1-nomaskloss/epoch_500",
        help="Path to the checkpoint directory.",
    )
    args = parser.parse_args()
    config_path = args.config_path
    checkpoint_dir = args.checkpoint_dir
    config = load_config(config_path)

    # Read core settings from config
    batch_size = config["train_args"]["batch_size"]    
    num_val_samples = config["train_args"].get("num_val_samples", 5)
    print(f"Validation settings - batch_size: {batch_size}, num_val_samples: {num_val_samples}")

    # ====== Single GPU Setup ======
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====== Logging ======
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    experiment_name = os.path.basename(checkpoint_dir.rstrip("/"))
    # Save config file for reproducibility
    exp_config_dir = os.path.join(os.path.dirname(__file__), "exp_configs")
    os.makedirs(exp_config_dir, exist_ok=True)
    exp_config_path = os.path.join(exp_config_dir, f"{experiment_name}_validation.yaml")
    shutil.copy2(config_path, exp_config_path)
    print("Validation config saved to:", exp_config_path)
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
    
    # ====== Build model ======
    model_cfg = config["model_args"].copy()
    model_cfg["mask_conditioning"] = config["general_args"]["mask_conditioning"]
    model = build_model(model_cfg, device=device)

    # NOTE: new data loader - same as inferer_ver7
    # Load sampling params from config
    da = config["data_args"]
    val_dataset = MhaDataset_small(
        root_dir=da["val_root"],
        n_images=da["val_n_images"],
        n_patches=da["val_n_patches"],
        reverse=da["val_reverse"],
        scale=da["scale"],
        scale_mask=da["scale_mask"],
        npy_root=da["npy_root"],
        seed=seed,  # Use the same seed for reproducibility
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2)
    

    # Create optimizer (needed for checkpoint loading)
    lr = config["train_args"]["lr"] * batch_size
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load the checkpoint
    start_epoch, loaded_config = load_checkpoint(
        model, optimizer, checkpoint_dir=checkpoint_dir, device=device, valid_only=False
    )
    print(f"Loaded checkpoint from epoch {start_epoch}")

    # Set model to evaluation mode
    model.eval()

    # Define path object (scheduler included)
    path = AffineProbPath(scheduler=CondOTScheduler())

    solver_config = config["solver_args"]

    print("Starting validation...")
    
    # Run validation
    val_mae, val_psnr, val_ssim = validate_and_save_samples(
        model=model,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epoch=start_epoch,
        solver_config=solver_config,
        writer=None,  # Pass None instead of writer
        max_samples=num_val_samples,
        class_map={0: "AB", 1: "HN", 2: "TH"},
        mask_conditioning=mask_conditioning,
        class_conditioning=class_conditioning,
        val=True,
        scale=da["scale"],
        print_batch_results=True,  # Enable batch-level result printing
    )
    
    print(f"Validation Results:")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  PSNR: {val_psnr:.4f} dB")
    print(f"  SSIM: {val_ssim:.4f}")
    
    print("Validation complete!")


if __name__ == "__main__":
    main() 

# python trainer_simple.py --config_path /media/prajbori/sda/private/github/proj_synthrad/algorithm-template/exp_configs/test_3090_1.yaml --checkpoint_dir /media/prajbori/sda/private/github/proj_synthrad/algorithm-template/some_checkpoints/apex-dist3-llf-HN/epoch_700