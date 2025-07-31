# %%
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

#---------added part--------
import glob
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
is_interactive = sys.stdout.isatty()
from matplotlib import pyplot as plt

print("Configuration and Paths")

from typing import Dict

import SimpleITK as sitk
# import torch
import numpy as np

from base_algorithm import BaseSynthradAlgorithm
from monai.inferers.utils import sliding_window_inference


class SynthradAlgorithm(BaseSynthradAlgorithm):
    """
    CT synthesis from MRI using flow matching with sliding window inference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Constants
        self.CT_UPPER = 3071.0
        self.CT_LOWER = -1024.0
        self.MRI_UPPER = 1357.0
        self.MRI_LOWER = 0.0
        
        # Sampling steps configuration
        self.SAMPLING_STEPS = {
            "fast": 10,
            "medium": 50, 
            "high": 100,
            "ultra": 200
        }
        
        # Default configuration paths - these should be set via environment or config
        self.config_path = os.getenv("CONFIG_PATH", "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/exp_configs/test20-testseed1-nomaskloss.yaml")
        self.checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/some_checkpoints/test20-testseed1-nomaskloss/epoch_500")
        self.sampling_quality = os.getenv("SAMPLING_QUALITY", "fast")
        
        # Load configuration and initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the flow matching model."""
        try:
            # Load configuration
            self.config = load_config(self.config_path)
            self.num_sampling_steps = self.SAMPLING_STEPS[self.sampling_quality]
            
            # Device setup
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Load model
            self._load_model()
            
            # Initialize scaling objects
            self._initialize_scaling()
            
            # Set model to evaluation mode
            if self.model is not None:
                self.model.eval()
            
        except Exception as e:
            print(f"Warning: Could not initialize flow matching model: {e}")
            print("Falling back to simple CT generation")
            self.model = None
            
    def _load_model(self):
        """Load and initialize the model."""
        try:
            # Model configuration
            model_cfg = self.config["model_args"].copy()
            model_cfg["mask_conditioning"] = self.config["general_args"]["mask_conditioning"]
            
            # Build model
            self.model = build_model(model_cfg, device=self.device)
            self.model = self.model.float()
            
            # Create optimizer (needed for checkpoint loading)
            lr = self.config["train_args"]["lr"] * self.config["train_args"]["batch_size"]
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            
            # Load checkpoint
            start_epoch, loaded_config = load_checkpoint(
                self.model, optimizer, checkpoint_dir=self.checkpoint_dir, 
                device=self.device, valid_only=False
            )
            
            # Define path object
            self.path = AffineProbPath(scheduler=CondOTScheduler())
            
            print(f"Model loaded from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
    def _initialize_scaling(self):
        """Initialize scaling objects based on configuration."""
        try:
            da = self.config["data_args"]
            self.scale_method = da["scale"]
            self.scale_mask_method = da["scale_mask"]
            self.npy_root = da["npy_root"]
            self.seed = self.config["general_args"].get("seed", 42)
            
            print(f"Scaling configuration: scale_method={self.scale_method}, scale_mask_method={self.scale_mask_method}")
            print(f"NPY root: {self.npy_root}")
            
            # Initialize scaling objects
            if self.scale_method in ['uniform', 'uniform2']:
                if self.scale_method == 'uniform':
                    self.ct_scaling = QT(file_path=os.path.join(self.npy_root, "ct_mask1.npy"), seed=self.seed)
                else:
                    self.ct_scaling = GBoost(file_path=os.path.join(self.npy_root, "ct_mask1.npy"), seed=self.seed)
            elif self.scale_method == 'sigmoid2':
                self.ct_scaling = Sigmoid2()
                
            if self.scale_mask_method in ['uniform', 'uniform2']:
                if self.scale_mask_method == 'uniform':
                    self.mri_scaling = QT(file_path=os.path.join(self.npy_root, "mri_mask1.npy"), seed=self.seed)
                else:
                    self.mri_scaling = GBoost(file_path=os.path.join(self.npy_root, "mri_mask1.npy"), seed=self.seed)
                    
            print("Scaling objects initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize scaling objects: {e}")
            print("Using default linear scaling")
            self.scale_method = 'linear'
            self.scale_mask_method = 'linear'
        
    def _reorient_to_RAS(self, img):
        """Reorient image to RAS coordinate system."""
        dicom_orient = sitk.DICOMOrientImageFilter()
        dicom_orient.SetDesiredCoordinateOrientation("RAS")
        return dicom_orient.Execute(img)
        
    def _scale_mri_array(self, arr):
        """Normalize MRI array to model input range."""
        arr = arr.astype(np.float32)
        arr = np.clip(arr, 0, self.MRI_UPPER)
        
        if self.scale_mask_method == 'linear':
            arr = (arr - (self.MRI_UPPER / 2)) / (self.MRI_UPPER / 2)
        elif self.scale_mask_method == 'sigmoid':
            lower, upper = 0.0, self.MRI_UPPER
            p_low = 0.005
            x0 = (upper + lower) / 2.0
            k = -np.log((1/p_low) - 1) / ((upper - lower) / 2)
            arr = 2.0 / (1.0 + np.exp(-k * (arr - x0))) - 1.0
            arr = arr * 0.99
        elif self.scale_mask_method in ['uniform', 'uniform2']:
            arr = self.mri_scaling.forward(arr)
            
        return arr
        
    def _denormalize_ct(self, ct_arr):
        """Denormalize CT values back to original HU range."""
        if self.scale_method == 'linear':
            lower, upper = self.CT_LOWER, self.CT_UPPER 
            midpoint = (upper + lower) / 2.0
            range_half = (upper - lower) / 2.0
            ct_arr = ct_arr * range_half + midpoint
        elif self.scale_method == 'sigmoid':
            lower, upper = self.CT_LOWER, self.CT_UPPER 
            p_low = 0.005
            x0 = (upper + lower) / 2.0
            logit_high = np.log((1 - p_low) / p_low)
            k = 2 * logit_high / (upper - lower)
            s = (ct_arr + 1) / 2
            ct_arr = x0 + (1/k) * np.log(s / (1 - s))
        elif self.scale_method in ['sigmoid2', 'uniform', 'uniform2']:
            ct_arr = self.ct_scaling.inverse(ct_arr)
            
        return ct_arr
        
    def _high_quality_sampling(self, x_init, mri_patch, cond_tensor):
        """Multi-step sampling for CT synthesis using flow matching."""
        if self.model is None:
            raise RuntimeError("Model is not initialized. Cannot perform high quality sampling.")
            
        # Create time grid for sampling
        t_grid = torch.linspace(0, 1, self.num_sampling_steps, device=self.device, dtype=torch.float32)
        
        # Initialize with noise
        x_t = x_init.clone()
        
        with torch.no_grad():
            for i in range(self.num_sampling_steps - 1):
                # Current time step
                t = t_grid[i].unsqueeze(0).expand(x_t.shape[0])
                
                # Ensure cond_tensor has the correct shape for the model: [B, 1, 3]
                if cond_tensor.dim() == 2:
                    cond_tensor = cond_tensor.unsqueeze(1).float()
                
                # Get velocity prediction
                v_pred = self.model(x=x_t, t=t, cond=cond_tensor, masks=mri_patch)
                
                # Time step size
                dt = t_grid[i+1] - t_grid[i]
                
                # Update x_t using Euler method
                x_t = x_t + dt * v_pred
        
        return x_t
        
    def _predictor(self, patch):
        """Process patch during sliding window inference."""
        # patch: [N, 3, z, y, x]
        x_init = patch[:, 0:1].to(self.device, dtype=torch.float32)
        mri_p = patch[:, 1:2].to(self.device, dtype=torch.float32)
        mask_p = patch[:, 2:3].to(self.device, dtype=torch.float32)
        cond = self.cond_tensor_full.to(self.device, dtype=torch.float32).expand(x_init.shape[0], -1)
        
        out = self._high_quality_sampling(x_init, mri_p, cond)
        
        # Zero outside body to stabilize blending
        out = out * (mask_p > 0.5)
        return out.detach().to(x_init.dtype)

    def predict(self, input_dict: Dict[str, sitk.Image]) -> sitk.Image:
        """
        Generates a synthetic CT image from the given input image and mask using flow matching.

        Parameters
        ----------
        input_dict : Dict[str, SimpleITK.Image]
            A dictionary containing keys: "image", "mask", and "region". 
            The values are SimpleITK.Image objects representing the input image and mask respectively.

        Returns
        -------
        SimpleITK.Image
            The generated synthetic CT image.
        """
        assert list(input_dict.keys()) == ["image", "mask", "region"]

        # Extract region information
        region = input_dict["region"]
        print(f"Processing region: {region}")
        mr_sitk = input_dict["image"]
        mask_sitk = input_dict["mask"]

        # If model is not available, fall back to simple generation
        if self.model is None:
            print("Using simple CT generation (fallback)")
            return self._simple_ct_generation(mr_sitk, mask_sitk)

        print("Using flow matching model for CT synthesis")
        
        # Convert sitk images to np arrays
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype("float32")
        mr_np = sitk.GetArrayFromImage(mr_sitk).astype("float32")

        print(f"MRI shape: {mr_np.shape}")

        # Reorient to RAS for processing
        mr_ras = self._reorient_to_RAS(mr_sitk)
        mr_arr = sitk.GetArrayFromImage(mr_ras).astype(np.float32)

        # Apply scaling
        mri_scaled = self._scale_mri_array(mr_arr.copy())

        # Prepare tensors
        mri_t = torch.from_numpy(mri_scaled)[None, None]  # [1, 1, Z, Y, X]
        mask_t = torch.from_numpy(mask_np)[None, None]   # [1, 1, Z, Y, X]

        # Random noise as starting point
        x_noise = torch.randn_like(mri_t, dtype=torch.float32)

        # Stack channels: [noise, mri, mask]
        inputs_stack = torch.cat([x_noise, mri_t, mask_t], dim=1)  # [1, 3, Z, Y, X]

        # Class conditioning based on region
        if region == 'AB':
            lbl = np.array([1, 0, 0], dtype=np.float32)
        elif region == 'HN':
            lbl = np.array([0, 1, 0], dtype=np.float32)
        elif region == 'TH':
            lbl = np.array([0, 0, 1], dtype=np.float32)
        else:
            lbl = np.array([0, 1, 0], dtype=np.float32)  # Default to HN

        self.cond_tensor_full = torch.from_numpy(lbl)[None, :]  # [1, 3]

        print(f"Running sliding window inference with {self.num_sampling_steps} steps...")

        # Sliding window inference
        roi_size = (16, 128, 128)
        overlap = 0.5

        pred_norm = sliding_window_inference(
            inputs=inputs_stack,
            roi_size=roi_size,
            sw_batch_size=2,
            predictor=self._predictor,
            overlap=overlap,
            mode="gaussian",
            sigma_scale=0.125,
            padding_mode="reflect",
            cval=0.0,
            sw_device=self.device,
            device=torch.device("cpu"),
            progress=True,
        )

        # Convert to numpy and denormalize - handle potential tuple/dict return
        if isinstance(pred_norm, (tuple, list)):
            pred_norm = pred_norm[0]
        elif isinstance(pred_norm, dict):
            pred_norm = pred_norm['pred']

        pred_norm = pred_norm.cpu().numpy()[0, 0]  # Remove batch and channel dims

        # Denormalize
        pred_denorm = self._denormalize_ct(pred_norm.copy())

        # Set background to CT_LOWER using the mask
        pred_denorm[mask_np <= 0.5] = self.CT_LOWER

        # Create SimpleITK image and copy metadata
        sCT_sitk = sitk.GetImageFromArray(pred_denorm)
        sCT_sitk.CopyInformation(mr_sitk)

        print(f"CT synthesis completed. Output range: [{np.min(pred_denorm):.2f}, {np.max(pred_denorm):.2f}]")
        return sCT_sitk

    def _simple_ct_generation(self, mr_sitk, mask_sitk):
        """Fallback simple CT generation when model is not available."""
        print("Using simple CT generation (fallback)")

        # convert sitk images to np arrays
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype("float32")
        mr_np = sitk.GetArrayFromImage(mr_sitk).astype("float32")

        # check if GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        # convert np arrays to tensors
        mr_tensor = torch.tensor(mr_np, device=device)
        mask_tensor = torch.tensor(mask_np, device=device)

        # sCT generation placeholder (set values inside mask to 0)
        mr_tensor[mask_tensor == 1] = 0
        mr_tensor[mask_tensor == 0] = -1000

        # convert tensor back to np array
        sCT = mr_tensor.cpu().numpy()

        sCT_sitk = sitk.GetImageFromArray(sCT)
        sCT_sitk.CopyInformation(mr_sitk)  # copies spatial metadata (origin, spacing, direction)

        return sCT_sitk

    def calculate_metrics(self, ground_truth_ct_path, predicted_ct_path, mask_path=None):
        """
        Calculate MAE, PSNR, SSIM between ground truth and predicted CT.
        
        Args:
            ground_truth_ct_path: Path to ground truth CT file
            predicted_ct_path: Path to predicted CT file
            mask_path: Optional path to mask file (if provided, only voxels with mask=1 are considered)
            
        Returns:
            dict: Dictionary containing MAE, PSNR, SSIM values
        """
        # Load images
        original_ct = sitk.ReadImage(ground_truth_ct_path)
        predicted_ct = sitk.ReadImage(predicted_ct_path)
        
        # Convert to numpy arrays
        original_array = sitk.GetArrayFromImage(original_ct)
        predicted_array = sitk.GetArrayFromImage(predicted_ct)
        
        # Load mask if provided
        if mask_path:
            mask_image = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask_image)
            mask_bool = mask_array > 0.5  # Consider mask=1 as True
            original_array_masked = original_array[mask_bool]
            predicted_array_masked = predicted_array[mask_bool]
        else:
            # If no mask provided, use all voxels
            original_array_masked = original_array.flatten()
            predicted_array_masked = predicted_array.flatten()
        
        # Calculate MAE (only on masked voxels)
        mae = np.mean(np.abs(original_array_masked - predicted_array_masked))
        
        # Calculate PSNR (only on masked voxels)
        mse = np.mean((original_array_masked - predicted_array_masked) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            max_val = np.max(original_array_masked)
            psnr = 20 * np.log10(max_val / np.sqrt(mse))
        
        # Calculate SSIM (on full volume for structural similarity)
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
        
        metrics = {
            'MAE': mae,
            'PSNR': psnr,
            'SSIM': ssim_score
        }
        
        print(f"Metrics - MAE: {mae:.4f}, PSNR: {psnr:.4f} dB, SSIM: {ssim_score:.4f}")
        if mask_path:
            print(f"Metrics calculated on {len(original_array_masked)} masked voxels (mask=1)")
        else:
            print(f"Metrics calculated on all {len(original_array_masked)} voxels")
        
        return metrics
        
    def plot_comparison(self, ground_truth_ct_path=None, predicted_ct_path=None, mask_path=None, 
                       mri_path=None, slice_indices=None, save_path=None):
        """
        Plot comparison between ground truth and predicted CT.
        
        Args:
            ground_truth_ct_path: Path to ground truth CT file
            predicted_ct_path: Path to predicted CT file
            mask_path: Optional path to mask file
            mri_path: Optional path to MRI file
            slice_indices: Dictionary with 'axial', 'coronal', 'sagittal' slice indices
            save_path: Optional path to save the plot (defaults to output_path/{case_id}.png)
        """
        # Check if at least one path is provided
        if not any([ground_truth_ct_path, predicted_ct_path, mask_path, mri_path]):
            raise ValueError("At least one image path must be provided (ground_truth_ct_path, predicted_ct_path, mask_path, or mri_path)")
        
        # Set default save path if not provided
        if save_path is None:
            # Extract case ID from predicted CT filename if available, otherwise use a default
            if predicted_ct_path:
                predicted_filename = os.path.basename(predicted_ct_path)
                case_id = os.path.splitext(predicted_filename)[0]  # Remove extension
            else:
                case_id = "comparison"
            save_path = os.path.join(str(self.output_path.parent), f"{case_id}.png")
        
        # Determine which images to load and their order
        images_to_load = []
        titles = []
        
        if predicted_ct_path:
            images_to_load.append(("predicted", predicted_ct_path))
            titles.append("Predicted CT")
            
        if ground_truth_ct_path:
            images_to_load.append(("ground_truth", ground_truth_ct_path))
            titles.append("Ground Truth CT")
            
        if mri_path:
            images_to_load.append(("mri", mri_path))
            titles.append("MRI")
            
        if mask_path:
            images_to_load.append(("mask", mask_path))
            titles.append("Mask")
        
        # Load images
        loaded_images = {}
        reference_shape = None
        
        for img_type, img_path in images_to_load:
            try:
                img = sitk.ReadImage(img_path)
                img_array = sitk.GetArrayFromImage(img)
                loaded_images[img_type] = img_array
                
                # Use the first loaded image as reference shape
                if reference_shape is None:
                    reference_shape = img_array.shape
                elif img_array.shape != reference_shape:
                    print(f"Warning: {img_type} image shape {img_array.shape} differs from reference shape {reference_shape}")
                    
            except Exception as e:
                print(f"Warning: Could not load {img_type} image from {img_path}: {e}")
                # Create a zero array with reference shape
                if reference_shape:
                    loaded_images[img_type] = np.zeros(reference_shape)
        
        # If no images were successfully loaded, raise error
        if not loaded_images:
            raise ValueError("No images could be loaded from the provided paths")
        
        # Default slice indices
        if slice_indices is None:
            if reference_shape is None:
                raise ValueError("No valid images loaded to determine slice indices")
            z, y, x = reference_shape
            slice_indices = {
                'axial': z // 2,
                'coronal': y // 2,
                'sagittal': x // 2
            }
        
        # Create subplots for each view - dynamic number of columns
        num_columns = len(loaded_images)
        fig, axes = plt.subplots(3, num_columns, figsize=(5*num_columns, 15))
        fig.suptitle('CT Synthesis Comparison', fontsize=16)
        
        # Handle single column case
        if num_columns == 1:
            axes = axes.reshape(-1, 1)
        
        views = ['axial', 'coronal', 'sagittal']
        
        for i, view in enumerate(views):
            for j, (img_type, img_array) in enumerate(loaded_images.items()):
                # Extract slice based on view
                if view == 'axial':
                    slice_data = img_array[slice_indices[view]]
                elif view == 'coronal':
                    slice_data = img_array[:, slice_indices[view]]
                else:  # sagittal
                    slice_data = img_array[:, :, slice_indices[view]]
                
                # Determine value range based on image type
                if img_type in ['predicted', 'ground_truth']:
                    # For CT images, use consistent scaling if both are present
                    if 'predicted' in loaded_images and 'ground_truth' in loaded_images:
                        pred_slice = loaded_images['predicted'][slice_indices[view]] if view == 'axial' else \
                                   loaded_images['predicted'][:, slice_indices[view]] if view == 'coronal' else \
                                   loaded_images['predicted'][:, :, slice_indices[view]]
                        gt_slice = loaded_images['ground_truth'][slice_indices[view]] if view == 'axial' else \
                                 loaded_images['ground_truth'][:, slice_indices[view]] if view == 'coronal' else \
                                 loaded_images['ground_truth'][:, :, slice_indices[view]]
                        vmin = min(pred_slice.min(), gt_slice.min())
                        vmax = max(pred_slice.max(), gt_slice.max())
                    else:
                        vmin = slice_data.min()
                        vmax = slice_data.max()
                else:
                    # For MRI and mask, use their own ranges
                    vmin = slice_data.min()
                    vmax = slice_data.max()
                
                # Plot image
                im = axes[i, j].imshow(slice_data, cmap="gray", vmin=vmin, vmax=vmax)
                axes[i, j].set_title(f"{titles[j]} ({view})")
                axes[i, j].axis('off')
                plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Run the algorithm on the default input and output paths specified in BaseSynthradAlgorithm.
    print("Starting SynthRAD algorithm...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Environment variables:")
    print(f"  TASK_TYPE: {os.getenv('TASK_TYPE', 'Not set')}")
    print(f"  INPUT_FOLDER: {os.getenv('INPUT_FOLDER', 'Not set')}")
    print(f"  OUTPUT_FOLDER: {os.getenv('OUTPUT_FOLDER', 'Not set')}")
    print(f"  CONFIG_PATH: {os.getenv('CONFIG_PATH', 'Not set')}")
    print(f"  CHECKPOINT_DIR: {os.getenv('CHECKPOINT_DIR', 'Not set')}")
    
    try:
        synthrad = SynthradAlgorithm()
        
        # Debug: Check what files are available
        print(f"\nChecking input directory: {synthrad.input_path}")
        if os.path.exists(synthrad.input_path):
            print(f"Input path exists. Contents:")
            for item in os.listdir(synthrad.input_path):
                print(f"  - {item}")
        else:
            print("Input path does not exist!")
            
        print(f"\nChecking mask directory: {synthrad.mask_path}")
        if os.path.exists(synthrad.mask_path):
            print(f"Mask path exists. Contents:")
            for item in os.listdir(synthrad.mask_path):
                print(f"  - {item}")
        else:
            print("Mask path does not exist!")
            
        print(f"\nChecking region file: {synthrad.region_path}")
        if os.path.exists(synthrad.region_path):
            print(f"Region file exists")
        else:
            print("Region file does not exist!")
        
        synthrad.process()
        print("Algorithm execution completed successfully")
    except Exception as e:
        print(f"Error during algorithm execution: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # synthrad.calculate_metrics(
    #     ground_truth_ct_path="./test_mha/images/ct/1HNA001.mha", 
    #     predicted_ct_path="./output/images/synthetic-ct/1HNA001.mha", 
    #     mask_path="./test_mha/images/body/1HNA001.mha")
    # synthrad.plot_comparison(
    #     ground_truth_ct_path="./test_mha/images/ct/1HNA001.mha",
    #     predicted_ct_path="./output/images/synthetic-ct/1HNA001.mha",
    #     mask_path="./test_mha/images/body/1HNA001.mha",
    #     mri_path="./test_mha/images/mri/1HNA001.mha",
    # )
