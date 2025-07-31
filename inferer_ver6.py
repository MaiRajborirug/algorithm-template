import os
import sys
import warnings
import torch
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from utils.general_utils import load_config, load_checkpoint
from utils.utils_fm import build_model
from utils.scaling import GBoost, Sigmoid2, QT
from monai.inferers.utils import sliding_window_inference


class SynthradAlgorithm:
    """
    CT synthesis from MRI using flow matching with sliding window inference.
    """
    
    def __init__(self, config_path, checkpoint_dir, sampling_quality="fast", device=None):
        """
        Initialize the SynthradAlgorithm.
        
        Args:
            config_path: Path to the configuration YAML file
            checkpoint_dir: Directory containing model checkpoint
            sampling_quality: Sampling quality ("fast", "medium", "high", "ultra")
            device: Device to use (None for auto-detection)
        """
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
        
        # Load configuration
        self.config = load_config(config_path)
        self.sampling_quality = sampling_quality
        self.num_sampling_steps = self.SAMPLING_STEPS[sampling_quality]
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        # Load model
        self._load_model(checkpoint_dir)
        
        # Initialize scaling objects
        self._initialize_scaling()
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _load_model(self, checkpoint_dir):
        """Load and initialize the model."""
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
            self.model, optimizer, checkpoint_dir=checkpoint_dir, 
            device=self.device, valid_only=False
        )
        
        # Define path object
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        
        print(f"Model loaded from epoch {start_epoch}")
        
    def _initialize_scaling(self):
        """Initialize scaling objects based on configuration."""
        da = self.config["data_args"]
        self.scale_method = da["scale"]
        self.scale_mask_method = da["scale_mask"]
        self.npy_root = da["npy_root"]
        self.seed = self.config["general_args"].get("seed", 42)
        
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
                
    def _reorient_to_RAS(self, img):
        """Reorient image to RAS coordinate system."""
        dicom_orient = sitk.DICOMOrientImageFilter()
        dicom_orient.SetDesiredCoordinateOrientation("RAS")
        return dicom_orient.Execute(img)
        
    def _scale_ct_array(self, arr):
        """Normalize CT array to model input range."""
        arr = arr.astype(np.float32)
        
        if self.scale_method == 'linear':
            lower, upper = self.CT_LOWER, self.CT_UPPER
            midpoint = (upper + lower) / 2.0
            range_half = (upper - lower) / 2.0
            arr = np.clip(arr, lower, upper)
            arr = (arr - midpoint) / range_half
        elif self.scale_method == 'sigmoid':
            lower, upper = self.CT_LOWER, self.CT_UPPER
            arr = np.clip(arr, lower, upper)
            p_low = 0.005
            x0 = (upper + lower) / 2.0
            logit_high = np.log((1 - p_low) / p_low)
            k = 2 * logit_high / (upper - lower)
            s = 1 / (1 + np.exp(-k * (arr - x0)))
            arr = 2 * s - 1
        elif self.scale_method in ['sigmoid2', 'uniform', 'uniform2']:
            arr = self.ct_scaling.forward(arr)
            
        return arr
        
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
        
    def predict(self, case_id, data_root, anatomical_region=None, use_real_mask=False):
        """
        Predict CT from MRI and save the result.
        
        Args:
            case_id: Case ID (e.g., "1HNA001")
            data_root: Root directory containing images/ folder with mri/, ct/, body/ subfolders
            anatomical_region: Anatomical region ("AB", "HN", "TH") - auto-detected if None
            use_real_mask: Whether to load real mask file (slower) or create simple mask (faster)
            
        Returns:
            Path to saved CT file
        """
        # Auto-detect anatomical region from case_id if not provided
        if anatomical_region is None:
            if case_id.startswith('1AB'):
                anatomical_region = 'AB'
            elif case_id.startswith('1HN'):
                anatomical_region = 'HN'
            elif case_id.startswith('1TH'):
                anatomical_region = 'TH'
            else:
                anatomical_region = 'HN'  # Default
                
        print(f"Processing case {case_id} for {anatomical_region} region")
        
        # Construct file paths
        mri_path = os.path.join(data_root, "images", "mri", f"{case_id}.mha")
        
        # Check if MRI file exists
        if not os.path.exists(mri_path):
            raise FileNotFoundError(f"MRI file not found: {mri_path}")
            
        # Set output directory for synthetic CT
        output_dir = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/output/images/synthetic-ct"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{case_id}.mha")
        
        print(f"Loading MRI from: {mri_path}")
        print(f"Saving CT to: {save_path}")
        
        # Load MRI
        mri_image = sitk.ReadImage(mri_path)
        original_mri = mri_image  # Keep original for metadata
        
        # Reorient to RAS for processing
        mri_ras = self._reorient_to_RAS(mri_image)
        mri_arr = sitk.GetArrayFromImage(mri_ras).astype(np.float32)
        
        print(f"MRI shape: {mri_arr.shape}")
        
        # Handle mask - use fast approach by default
        if use_real_mask:
            mask_path = os.path.join(data_root, "images", "body", f"{case_id}.mha")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            print(f"Loading mask from: {mask_path}")
            
            # Load and process real mask
            mask_image = sitk.ReadImage(mask_path)
            mask_ras = self._reorient_to_RAS(mask_image)
            mask_arr = sitk.GetArrayFromImage(mask_ras).astype(np.float32)
            print(f"Mask shape: {mask_arr.shape}")
        else:
            # Fast approach: create simple mask (assume full volume is valid)
            mask_arr = np.ones_like(mri_arr, dtype=np.float32)
            print("Using simple mask (full volume)")
        
        # Apply scaling
        mri_scaled = self._scale_mri_array(mri_arr.copy())
        
        # Prepare tensors
        mri_t = torch.from_numpy(mri_scaled)[None, None]  # [1, 1, Z, Y, X]
        mask_t = torch.from_numpy(mask_arr)[None, None]   # [1, 1, Z, Y, X]
        
        # Random noise as starting point
        x_noise = torch.randn_like(mri_t, dtype=torch.float32)
        
        # Stack channels: [noise, mri, mask]
        inputs_stack = torch.cat([x_noise, mri_t, mask_t], dim=1)  # [1, 3, Z, Y, X]
        
        # Class conditioning
        if anatomical_region == 'AB':
            lbl = np.array([1, 0, 0], dtype=np.float32)
        elif anatomical_region == 'HN':
            lbl = np.array([0, 1, 0], dtype=np.float32)
        elif anatomical_region == 'TH':
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
        
        # Get original MRI statistics for denormalization
        mri_min, mri_max = float(np.min(mri_arr)), float(np.max(mri_arr))
        pred_denorm = self._denormalize_ct(pred_norm.copy())
        
        # Set background to CT_LOWER using the mask
        pred_denorm[mask_arr <= 0.5] = self.CT_LOWER
        
        # Save result
        output_path = self._save_result(pred_denorm, save_path, original_mri)
        
        print(f"CT synthesis completed. Saved to: {output_path}")
        return output_path
        
    def _save_result(self, output_volume, save_path, original_image):
        """Save predicted CT volume as MHA file with proper metadata."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create SimpleITK image from array
        output_ras_img = sitk.GetImageFromArray(output_volume)
        
        # If we have the original image, resample to match its orientation
        if original_image is not None:
            # Prepare RAS-oriented reference
            orienter = sitk.DICOMOrientImageFilter()
            orienter.SetDesiredCoordinateOrientation("RAS")
            original_ras = orienter.Execute(original_image)
            output_ras_img.CopyInformation(original_ras)
            
            # Resample to match original orientation
            output_image = sitk.Resample(
                output_ras_img,
                original_image,
                sitk.Transform(),
                sitk.sitkLinear,
                self.CT_LOWER,
                sitk.sitkFloat32
            )
        else:
            output_image = output_ras_img
            output_image.SetSpacing([1.0, 1.0, 1.0])
            output_image.SetOrigin([0.0, 0.0, 0.0])
            
        # Save
        sitk.WriteImage(output_image, save_path)
        
        print(f"Output volume shape: {output_volume.shape}")
        print(f"Output volume range: [{np.min(output_volume):.2f}, {np.max(output_volume):.2f}]")
        
        return save_path
        
    def calculate_metrics(self, ground_truth_ct_path, predicted_ct_path, mask_path=None):
        """
        Calculate MAE, PSNR, SSIM between ground truth and predicted CT.
        
        Args:
            ground_truth_ct_path: Path to ground truth CT file
            predicted_ct_path: Path to predicted CT file
            mask_path: Optional path to mask file
            
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
        
        return metrics
        
    def plot_comparison(self, ground_truth_ct_path, predicted_ct_path, mask_path=None, 
                       slice_indices=None, save_path=None):
        """
        Plot comparison between ground truth and predicted CT.
        
        Args:
            ground_truth_ct_path: Path to ground truth CT file
            predicted_ct_path: Path to predicted CT file
            mask_path: Optional path to mask file
            slice_indices: Dictionary with 'axial', 'coronal', 'sagittal' slice indices
            save_path: Optional path to save the plot
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
        else:
            mask_array = np.ones_like(original_array)
        
        # Default slice indices
        if slice_indices is None:
            z, y, x = original_array.shape
            slice_indices = {
                'axial': z // 2,
                'coronal': y // 2,
                'sagittal': x // 2
            }
        
        # Create subplots for each view
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('CT Synthesis Comparison', fontsize=16)
        
        views = ['axial', 'coronal', 'sagittal']
        titles = ['Predicted CT', 'Ground Truth CT', 'Mask']
        
        for i, view in enumerate(views):
            if view == 'axial':
                pred_slice = predicted_array[slice_indices[view]]
                gt_slice = original_array[slice_indices[view]]
                mask_slice = mask_array[slice_indices[view]]
            elif view == 'coronal':
                pred_slice = predicted_array[:, slice_indices[view]]
                gt_slice = original_array[:, slice_indices[view]]
                mask_slice = mask_array[:, slice_indices[view]]
            else:  # sagittal
                pred_slice = predicted_array[:, :, slice_indices[view]]
                gt_slice = original_array[:, :, slice_indices[view]]
                mask_slice = mask_array[:, :, slice_indices[view]]
            
            # Determine value range for consistent scaling
            vmin = min(pred_slice.min(), gt_slice.min())
            vmax = max(pred_slice.max(), gt_slice.max())
            
            # Plot predicted CT
            im1 = axes[i, 0].imshow(pred_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f"{titles[0]} ({view})")
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Plot ground truth CT
            im2 = axes[i, 1].imshow(gt_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f"{titles[1]} ({view})")
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Plot mask
            im3 = axes[i, 2].imshow(mask_slice, cmap="gray")
            axes[i, 2].set_title(f"{titles[2]} ({view})")
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()


def main():
    """Example usage of SynthradAlgorithm with new data format."""
    # Configuration
    config_path = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/exp_configs/test20-testseed1-nomaskloss.yaml"
    checkpoint_dir = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/some_checkpoints/test20-testseed1-nomaskloss/epoch_500"
    
    # Initialize algorithm
    synthrad = SynthradAlgorithm(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        sampling_quality="fast"  # Options: "fast", "medium", "high", "ultra"
    )
    
    # New data format paths
    case_id = "1HNA001"
    data_root = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/test_mha"
    
    # Predict CT from MRI
    predicted_ct_path = synthrad.predict(
        case_id=case_id,
        data_root=data_root,
        anatomical_region=None,  # Will auto-detect from case_id
        use_real_mask=False # Use simple mask
    )
    
    # If you have ground truth CT for comparison
    ground_truth_ct_path = os.path.join(data_root, "images", "ct", f"{case_id}.mha")
    mask_path = os.path.join(data_root, "images", "body", f"{case_id}.mha")
    
    # Check if ground truth exists before calculating metrics
    if os.path.exists(ground_truth_ct_path):
        # Calculate metrics
        metrics = synthrad.calculate_metrics(
            ground_truth_ct_path=ground_truth_ct_path,
            predicted_ct_path=predicted_ct_path,
            mask_path=mask_path
        )
        
        # Plot comparison
        synthrad.plot_comparison(
            ground_truth_ct_path=ground_truth_ct_path,
            predicted_ct_path=predicted_ct_path,
            mask_path=mask_path,
            slice_indices={'axial': 50, 'coronal': 100, 'sagittal': 150},
            save_path="/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/output/comparison.png"
        )
    else:
        print(f"Ground truth CT not found at {ground_truth_ct_path}")
        print("Skipping metrics calculation and comparison plotting")


if __name__ == "__main__":
    main() 