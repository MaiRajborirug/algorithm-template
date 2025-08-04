import os
import sys
import warnings
import torch
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from tqdm import tqdm
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
    
    def __init__(self, config_path, checkpoint_dir, sampling_quality="fast", device='cuda:1',
                 ct_upper=3071.0, ct_lower=-1024.0, mri_upper=1357.0, mri_lower=0.0,
                 padding_mode="reflect", sigma_scale=0.125, mode="gaussian", overlap=0.5):
        """
        Initialize the SynthradAlgorithm.
        
        Args:
            config_path: Path to the configuration YAML file
            checkpoint_dir: Directory containing model checkpoint
            sampling_quality: Sampling quality ("fast", "medium", "high", "ultra")
            device: Device to use (default: 'cuda:1')
            ct_upper: Upper bound for CT values
            ct_lower: Lower bound for CT values
            mri_upper: Upper bound for MRI values
            mri_lower: Lower bound for MRI values
            padding_mode: Padding mode for sliding window
            sigma_scale: Sigma scale for sliding window
            mode: Mode for sliding window inference
            overlap: Overlap for sliding window
        """
        # Constants from parameters
        self.CT_UPPER = ct_upper
        self.CT_LOWER = ct_lower
        self.MRI_UPPER = mri_upper
        self.MRI_LOWER = mri_lower
        self.padding_mode = padding_mode
        self.sigma_scale = sigma_scale
        self.mode = mode
        self.overlap = overlap
        
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
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load model
        self._load_model(checkpoint_dir)
        
        # Initialize scaling objects
        self._initialize_scaling()
        
        # Set model to evaluation mode
        self.model.eval()
        
    def save_config(self, exp_name):
        """Save configuration parameters to a YAML file."""
        import yaml
        import os
        
        config_data = {
            'CT_UPPER': self.CT_UPPER,
            'CT_LOWER': self.CT_LOWER,
            'MRI_UPPER': self.MRI_UPPER,
            'MRI_LOWER': self.MRI_LOWER,
            'padding_mode': self.padding_mode,
            'sigma_scale': self.sigma_scale,
            'mode': self.mode,
            'overlap': self.overlap,
            'sampling_quality': self.sampling_quality,
            'num_sampling_steps': self.num_sampling_steps,
            'device': str(self.device)
        }
        
        config_path = os.path.join(exp_name, "other", "config.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        print(f"Configuration saved to: {config_path}")
        
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
        
    def predict(self, mri_path, save_path, mask_path=None, anatomical_region="HN"):
        """
        Predict CT from MRI and save the result.
        
        Args:
            mri_path: Path to input MRI file (.mha)
            save_path: Path to save the predicted CT (.mha)
            mask_path: Path to mask file (.mha) - if None, will use full volume mask
            anatomical_region: Anatomical region ("AB", "HN", "TH")
            
        Returns:
            tuple: (Path to saved CT file, dict with timing and patch info)
        """
        import time
        
        start_time = time.time()
        
        # print(f"Loading MRI from: {mri_path}")
        
        # Load MRI
        mri_image = sitk.ReadImage(mri_path)
        original_mri = mri_image  # Keep original for metadata
        
        # Reorient to RAS for processing
        mri_ras = self._reorient_to_RAS(mri_image)
        mri_arr = sitk.GetArrayFromImage(mri_ras).astype(np.float32)
        
        print(f"Load MRI: {mri_path.split('/')[-2]}, shape: {mri_arr.shape}")
        
        # Load or create mask
        if mask_path is not None:
            # print(f"Loading mask from: {mask_path}")
            mask_image = sitk.ReadImage(mask_path)
            # Reorient mask to RAS to match MRI
            mask_ras = self._reorient_to_RAS(mask_image)
            mask_arr = sitk.GetArrayFromImage(mask_ras).astype(np.float32)
            
            # Ensure mask has same shape as MRI
            if mask_arr.shape != mri_arr.shape:
                print(f"Warning: Mask shape {mask_arr.shape} doesn't match MRI shape {mri_arr.shape}")
                print("Resampling mask to match MRI dimensions...")
                # Resample mask to match MRI dimensions
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(mri_ras)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                mask_ras = resampler.Execute(mask_ras)
                mask_arr = sitk.GetArrayFromImage(mask_ras).astype(np.float32)
        else:
            # print("No mask provided, using full volume mask")
            # Create mask (assume full volume is valid)
            mask_arr = np.ones_like(mri_arr, dtype=np.float32)
        
        # Ensure mask is binary (0 or 1)
        mask_arr = (mask_arr > 0.5).astype(np.float32)
        
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
        
        # Sliding window inference
        roi_size = (16, 128, 128)
        
        # Count patches (approximate calculation)
        z, y, x = mri_arr.shape
        roi_z, roi_y, roi_x = roi_size
        overlap_z = int(roi_z * self.overlap)
        overlap_y = int(roi_y * self.overlap)
        overlap_x = int(roi_x * self.overlap)
        
        step_z = roi_z - overlap_z
        step_y = roi_y - overlap_y
        step_x = roi_x - overlap_x
        
        num_patches_z = max(1, (z - roi_z) // step_z + 1)
        num_patches_y = max(1, (y - roi_y) // step_y + 1)
        num_patches_x = max(1, (x - roi_x) // step_x + 1)
        total_patches = num_patches_z * num_patches_y * num_patches_x
        
        pred_norm = sliding_window_inference(
            inputs=inputs_stack,
            roi_size=roi_size,
            sw_batch_size=2,
            predictor=self._predictor,
            overlap=self.overlap,
            mode=self.mode,
            sigma_scale=self.sigma_scale,
            padding_mode=self.padding_mode,
            cval=0.0,
            sw_device=self.device,
            device=self.device,
            progress=True,
        )
        
        # Convert to numpy and denormalize
        pred_norm = pred_norm.cpu().numpy()[0, 0]  # Remove batch and channel dims
        
        # Get original MRI statistics for denormalization
        mri_min, mri_max = float(np.min(mri_arr)), float(np.max(mri_arr))
        pred_denorm = self._denormalize_ct(pred_norm.copy())
        
        # Set background to CT_LOWER
        pred_denorm[mask_arr <= 0.5] = self.CT_LOWER
        
        # Save result
        output_path = self._save_result(pred_denorm, save_path, original_mri)
        
        # Calculate timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60.0
        
        # Get image name
        image_name = os.path.basename(output_path)
        
        # Create info dict
        info = {
            'elapsed_time': elapsed_time,
            'elapsed_minutes': elapsed_minutes,
            'ct_shape': pred_denorm.shape,
            'mri_shape': mri_arr.shape,
            'image_name': image_name,
            'total_patches': total_patches,
            'time_warning': elapsed_minutes > 15
        }
        
        # print(f"CT synthesis completed. Saved to: {output_path}")
        print(f"CT synthesis completed. Saved to: {output_path.split('/')[-1]}")
        
        return output_path, info
        
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
        
    def process_cases(self, data_path, exp_name, anatomical_region="HN"):
        """
        Process multiple cases from a data directory.
        
        Args:
            data_path: Path to data directory containing case folders
            exp_name: Path to output directory for results
            anatomical_region: Anatomical region ("AB", "HN", "TH")
            
        Returns:
            dict: Dictionary containing metrics for all processed cases
        """
        import os
        import glob
        import time
        import random
        
        # Create output directories
        os.makedirs(exp_name, exist_ok=True)
        other_dir = os.path.join(exp_name, "other")
        os.makedirs(other_dir, exist_ok=True)
        
        # Create unique summary file name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        random_suffix = random.randint(1000, 9999)
        summary_filename = f"summary_metrics_{timestamp}_{random_suffix}.txt"
        summary_path = os.path.join(other_dir, summary_filename)
        
        # Save configuration
        self.save_config(exp_name)
        
        # Get all case directories
        case_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        case_dirs.sort()  # Sort for consistent processing order
        
        print(f"Found {len(case_dirs)} cases in {data_path}")
        print(f"Output directory: {exp_name}")
        print(f"Summary file: {summary_path}")
        
        all_metrics = {}
        
        # Write header to summary file
        with open(summary_path, 'w') as f:
            f.write("CASE METRICS SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Data path: {data_path}\n")
            f.write(f"Anatomical region: {anatomical_region}\n")
            f.write(f"Total cases: {len(case_dirs)}\n\n")
            f.write("PER-CASE METRICS:\n")
            f.write("-"*30 + "\n")
            f.flush()  # Ensure header is written immediately
        
        for case_dir in tqdm(case_dirs):
            case_path = os.path.join(data_path, case_dir)
            # print(f"Processing: {case_dir}")
            
            # Define file paths for this case
            mri_path = os.path.join(case_path, "mr.mha")
            mask_path = os.path.join(case_path, "mask.mha")
            gt_ct_path = os.path.join(case_path, "ct.mha")
            
            # Check if required files exist
            if not os.path.exists(mri_path):
                print(f"    MRI file not found: {mri_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"    Mask file not found: {mask_path}")
                continue
            if not os.path.exists(gt_ct_path):
                print(f"    Ground truth CT file not found: {gt_ct_path}")
                continue
            
            # Define output paths
            sct_filename = f"sct_{case_dir}.mha"
            sct_path = os.path.join(exp_name, sct_filename)
            comparison_plot_path = os.path.join(other_dir, f"sct_{case_dir}.png")
            
            try:
                # Predict CT from MRI
                predicted_ct_path, info = self.predict(
                    mri_path=mri_path,
                    save_path=sct_path,
                    mask_path=mask_path,
                    anatomical_region=anatomical_region
                )
                
                # Calculate metrics
                metrics = self.calculate_metrics(
                    ground_truth_ct_path=gt_ct_path,
                    predicted_ct_path=predicted_ct_path,
                    mask_path=mask_path
                )
                
                # Store metrics with additional info
                all_metrics[case_dir] = {**metrics, **info}
                
                # Create comparison plot
                self.plot_comparison(
                    ground_truth_ct_path=gt_ct_path,
                    predicted_ct_path=predicted_ct_path,
                    mask_path=mask_path,
                    slice_indices=None,  # Use default middle slices
                    save_path=comparison_plot_path
                )
                
                # Print and write metrics immediately
                time_warning = "!" if info['time_warning'] else ""
                metrics_line = f"   MAE: {metrics['MAE']:.4f}, PSNR: {metrics['PSNR']:.4f}dB, SSIM: {metrics['SSIM']:.4f}, Time: {info['elapsed_minutes']:.1f}min{time_warning}, Patches: {info['total_patches']}, Shape: {info['ct_shape']}, File: {info['image_name']}"
                print(metrics_line)
                
                # Write to summary file immediately
                with open(summary_path, 'a') as f:
                    f.write(f"{case_dir}: MAE={metrics['MAE']:.4f}, PSNR={metrics['PSNR']:.4f}dB, SSIM={metrics['SSIM']:.4f}, Time={info['elapsed_minutes']:.1f}min{time_warning}, Patches={info['total_patches']}, Shape={info['ct_shape']}, File={info['image_name']}\n")
                    f.flush()  # Ensure it's written immediately
                
            except Exception as e:
                error_msg = f"   Error: {str(e)}"
                print(error_msg)
                all_metrics[case_dir] = {"error": str(e)}
                
                # Write error to summary file immediately
                with open(summary_path, 'a') as f:
                    f.write(f"{case_dir}: ERROR - {str(e)}\n")
                    f.flush()
                continue
        
        # Calculate and print summary statistics
        successful_cases = {k: v for k, v in all_metrics.items() if "error" not in v}
        if successful_cases:
            print(f"\n📊 SUMMARY: {len(successful_cases)}/{len(case_dirs)} cases processed")
            
            mae_values = [metrics['MAE'] for metrics in successful_cases.values()]
            psnr_values = [metrics['PSNR'] for metrics in successful_cases.values()]
            ssim_values = [metrics['SSIM'] for metrics in successful_cases.values()]
            time_values = [metrics['elapsed_minutes'] for metrics in successful_cases.values()]
            patch_values = [metrics['total_patches'] for metrics in successful_cases.values()]
            
            print(f"   MAE: {np.mean(mae_values):.4f}±{np.std(mae_values):.4f}")
            print(f"   PSNR: {np.mean(psnr_values):.4f}±{np.std(psnr_values):.4f} dB")
            print(f"   SSIM: {np.mean(ssim_values):.4f}±{np.std(ssim_values):.4f}")
            print(f"   Time: {np.mean(time_values):.1f}±{np.std(time_values):.1f} min")
            print(f"   Patches: {np.mean(patch_values):.0f}±{np.std(patch_values):.0f}")
            
            # Count time warnings
            time_warnings = sum(1 for metrics in successful_cases.values() if metrics['time_warning'])
            if time_warnings > 0:
                print(f"   ⚠️  {time_warnings} cases took >15 minutes")
            
            # Append summary statistics to file
            with open(summary_path, 'a') as f:
                f.write(f"\nSUMMARY STATISTICS:\n")
                f.write("-"*30 + "\n")
                f.write(f"Successful cases: {len(successful_cases)}/{len(case_dirs)}\n")
                f.write(f"MAE - Mean: {np.mean(mae_values):.4f}, Std: {np.std(mae_values):.4f}\n")
                f.write(f"PSNR - Mean: {np.mean(psnr_values):.4f} dB, Std: {np.std(psnr_values):.4f}\n")
                f.write(f"SSIM - Mean: {np.mean(ssim_values):.4f}, Std: {np.std(ssim_values):.4f}\n")
                f.write(f"Time - Mean: {np.mean(time_values):.1f} min, Std: {np.std(time_values):.1f} min\n")
                f.write(f"Patches - Mean: {np.mean(patch_values):.0f}, Std: {np.std(patch_values):.0f}\n")
                if time_warnings > 0:
                    f.write(f"Time warnings (>15min): {time_warnings} cases\n")
            
            print(f"📄 Summary saved to: {summary_path}")
        
        return all_metrics
        
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
    """Example usage of SynthradAlgorithm."""
    # Configuration parameters
    # config_path = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/exp_configs/test20-testseed1-nomaskloss.yaml"
    config_path = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/exp_configs/test_apex3.yaml"
    # checkpoint_dir = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/some_checkpoints/apex-dist3-llf-AB/epoch_700"
    checkpoint_dir = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/some_checkpoints/test20-testseed1-nomaskloss/epoch_500"
    sampling_quality = "fast"  # Options: "fast", "medium", "high", "ultra"
    
    # Data and output paths
    data_path = "/media/prajbori/sda/private/dataset/proj_synthrad/training/synthRAD2025_Task1_Train_ABCD/Task1/AB"
    exp_name = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/output_llfAB1-500"
    
    # CT and MRI value bounds
    CT_UPPER = 3071.0
    CT_LOWER = -1024.0
    MRI_UPPER = 1357.0
    MRI_LOWER = 0.0
    
    # Sliding window parameters
    padding_mode = "reflect"
    sigma_scale = 0.125
    mode = "gaussian"
    overlap = 0.5
    
    # Initialize algorithm with all parameters
    synthrad = SynthradAlgorithm(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        sampling_quality=sampling_quality,
        device='cuda:0',
        ct_upper=CT_UPPER,
        ct_lower=CT_LOWER,
        mri_upper=MRI_UPPER,
        mri_lower=MRI_LOWER,
        padding_mode=padding_mode,
        sigma_scale=sigma_scale,
        mode=mode,
        overlap=overlap
    )
    
    # # Example 1: Process single case
    # print("="*60)
    # print("EXAMPLE 1: Processing single case")
    # print("="*60)
    
    # mri_path = "/media/prajbori/sda/private/dataset/proj_synthrad/training/synthRAD2025_Task1_Train_D/Task1/HN_x/1HND001/mr.mha"
    # mask_path = "/media/prajbori/sda/private/dataset/proj_synthrad/training/synthRAD2025_Task1_Train_D/Task1/HN_x/1HND001/mask.mha"
    # save_path = "/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/output_ver5/predicted_ct.mha"
    
    # # Predict CT from MRI with actual mask
    # predicted_ct_path = synthrad.predict(
    #     mri_path=mri_path,
    #     save_path=save_path,
    #     mask_path=mask_path,  # Use actual mask file
    #     anatomical_region="HN"  # Options: "AB", "HN", "TH"
    # )
    
    # # If you have ground truth CT for comparison
    # ground_truth_ct_path = "/media/prajbori/sda/private/dataset/proj_synthrad/training/synthRAD2025_Task1_Train_D/Task1/HN_x/1HND001/ct.mha"
    
    # # Calculate metrics
    # metrics = synthrad.calculate_metrics(
    #     ground_truth_ct_path=ground_truth_ct_path,
    #     predicted_ct_path=predicted_ct_path,
    #     mask_path=mask_path
    # )
    
    # # Plot comparison
    # synthrad.plot_comparison(
    #     ground_truth_ct_path=ground_truth_ct_path,
    #     predicted_ct_path=predicted_ct_path,
    #     mask_path=mask_path,
    #     slice_indices={'axial': 50, 'coronal': 100, 'sagittal': 150},
    #     save_path="/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/output_ver5/comparison.png"
    # )
    
    # Example 2: Process multiple cases
    print("\n" + "="*60)
    print("EXAMPLE 2: Processing multiple cases")
    print("="*60)
    
    # Process all cases in the directory
    all_metrics = synthrad.process_cases(
        data_path=data_path,
        exp_name=exp_name,
        anatomical_region="AB"  # Use "AB" for abdominal cases
    )
    
    print(f"\nProcessing completed! Check the output directory: {exp_name}")
    print("The directory contains:")
    print("  - sct_*.mha files: Synthetic CT images")
    print("  - other/sct_*.png files: Comparison plots")
    print("  - other/config.yaml: Configuration parameters")
    print("  - other/summary_metrics_*.txt: Summary of all metrics")


if __name__ == "__main__":
    main() 