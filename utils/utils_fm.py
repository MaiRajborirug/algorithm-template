import os
import json
import torch
import matplotlib.pyplot as plt
from torch import nn
from generative.networks.nets import DiffusionModelUNet, ControlNet
from flow_matching.solver import ODESolver
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import imageio

from .general_utils import normalize_zero_to_one, save_image

###############################################################################
# Model Building
###############################################################################
class MergedModel(nn.Module):
    """
    Merged model that wraps a UNet and an optional ControlNet.
    Takes in x, time in [0,1], and (optionally) a ControlNet condition.
    """

    def __init__(self, unet: DiffusionModelUNet, controlnet: ControlNet = None, max_timestep=1000):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.max_timestep = max_timestep

        # If controlnet is None, we won't do anything special in forward.
        self.has_controlnet = controlnet is not None
        self.has_conditioning = unet.with_conditioning

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        masks: torch.Tensor = None,
    ):
        """
        Args:
            x: input image tensor [B, C, H, W].
            t: timesteps in [0,1], will be scaled to [0, max_timestep - 1].
            cond: [B,1 , conditions_dim].
            masks: [B, C, H, W] masks for conditioning.

        Returns:
            The network output (e.g. velocity, noise, or predicted epsilon).
        """
        # Scale continuous t -> discrete timesteps(If you dont want to change the embedding function in the UNet)
        t = t * (self.max_timestep - 1)
        t = t.floor().long()

        # If t is scalar, expand to batch size
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        # t's shape should be [B]

        if self.has_controlnet:
            # cond is expected to be a ControlNet conditioning, e.g. mask
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x=x, timesteps=t, controlnet_cond=masks, context=cond
            )
            output = self.unet(
                x=x,
                timesteps=t,
                context=cond,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
        else:
            # If no ControlNet, cond might be cross-attention or None
            output = self.unet(x=x, timesteps=t, context=cond)

        return output


def build_model(model_config: dict, device: torch.device = None) -> MergedModel:
    """
    Builds a model (UNet only, or UNet+ControlNet) based on the provided model_config.

    Args:
        model_config: Dictionary containing model configuration.
        device: Device to move the model to.

    Returns:
        A MergedModel instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make a copy so the original config remains unaltered.
    mc = model_config.copy()

    # Pop out keys that are not needed by the model constructors.
    mask_conditioning = mc.pop("mask_conditioning", False)
    max_timestep = mc.pop("max_timestep", 1000)
    # Pop out ControlNet specific key, if present.
    cond_embed_channels = mc.pop("conditioning_embedding_num_channels", None)

    # Build the base UNet by passing all remaining items as kwargs.
    unet = DiffusionModelUNet(**mc)

    controlnet = None
    if mask_conditioning:
        # Ensure the controlnet has its specific key.
        if cond_embed_channels is None:
            cond_embed_channels = (16,)
        # Pass the same config kwargs to ControlNet plus the controlnet-specific key.
        # Prepare kwargs for ControlNet by excluding unsupported arguments
        controlnet_kwargs = mc.copy()
        controlnet_kwargs.pop("out_channels", None)
        controlnet = ControlNet(**controlnet_kwargs, conditioning_embedding_num_channels=cond_embed_channels)
        controlnet.load_state_dict(unet.state_dict(), strict=False)

    model = MergedModel(unet=unet, controlnet=controlnet, max_timestep=max_timestep)

    # Print number of trainable parameters.
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")
    model_size_mb = num_params * 4 / (1024**2)
    print(f"Model size: {model_size_mb:.2f} MB")

    return model.to(device)


import torch

def sample_with_solver(
    model,
    x_init,
    solver_config,
    cond=None,
    masks=None,
    return_intermediates=True,
):
    """
    Uses ODESolver (flow-matching) to sample from x_init -> final output.
    solver_config might contain keys:
        {
          "method": "midpoint"/"rk4"/etc.,
          "step_size": float,
          "time_points": int,
        }

    Returns either the full trajectory [time_points, B, C, H, W] if return_intermediates=True
    or just the final state [B, C, H, W].
    """
    # If model is wrapped in DataParallel, unwrap it for sampling - use single GPU
    if isinstance(model, torch.nn.DataParallel):
        velocity_model = model.module
    else:
        velocity_model = model

    solver = ODESolver(velocity_model=velocity_model)

    time_points = solver_config.get("time_points", 10)
    T = torch.linspace(0, 1, time_points, device=x_init.device)

    method = solver_config.get("method", "midpoint")
    step_size = solver_config.get("step_size", 0.02)

    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        method=method,
        step_size=step_size,
        return_intermediates=return_intermediates,
        cond=cond,
        masks=masks,
    )

    sol = solver.sample(time_grid=T, x_init=x_init,method=method,step_size=step_size, return_intermediates=True,cond=cond,masks=masks,)
    return sol


def plot_solver_steps(sol, im_batch, mask_batch, class_batch, class_map, outdir, max_plot=4):
    if sol.dim() != 5:  # No intermediates to plot
        return
    n_samples = min(sol.shape[1], max_plot)
    n_steps = sol.shape[0]
    if mask_batch is not None:
        fig, axes = plt.subplots(n_samples, n_steps + 2, figsize=(20, 8))
    else:
        fig, axes = plt.subplots(n_samples, n_steps + 1, figsize=(20, 8))
    if n_samples == 1:
        axes = [axes]
    for i in range(n_samples):
        for t in range(n_steps):
            axes[i][t].imshow(sol[t, i].cpu().numpy().squeeze(), cmap="gray")
            axes[i][t].axis("off")
            if i == 0:
                axes[i][t].set_title(f"Step {t}")
        col = n_steps
        if mask_batch is not None:
            axes[i][col].imshow(mask_batch[i].cpu().numpy().squeeze(), cmap="gray")
            axes[i][col].axis("off")
            if i == 0:
                axes[i][col].set_title("Mask")
            col += 1
        axes[i][col].imshow(im_batch[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i][col].axis("off")
        if i == 0:
            axes[i][col].set_title("Real")
        if class_map and class_batch is not None:
            idx = class_batch[i].argmax().item()
            cls = class_map[idx] if idx < len(class_map) else str(idx)
            axes[i][col].text(
                0.5,
                -0.15,
                f"Class: {cls}",
                ha="center",
                va="top",
                transform=axes[i][col].transAxes,
                color="red",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "solver_steps.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

def save_image_np(img_arr_, out_path):
    """
    Saves a single 2D image slice from a 4D numpy array with shape [C, D, H, W],
    selecting the middle slice along the D axis, and squeezing the channel dimension.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if img_arr_.ndim == 4 and img_arr_.shape[0] == 1:
        # Shape: [1, D, H, W] → [D, H, W]
        img_arr_ = img_arr_.squeeze(0)

        # Select middle slice along depth axis D
        d_mid = img_arr_.shape[0] // 2
        img_arr_ = img_arr_[d_mid]  # shape becomes [H, W]

    elif img_arr_.ndim == 3 and img_arr_.shape[0] != 1:
        # Already shape [D, H, W], pick middle slice
        d_mid = img_arr_.shape[0] // 2
        img_arr_ = img_arr_[d_mid]

    elif img_arr_.ndim == 3 and img_arr_.shape[0] == 1:
        # [1, H, W] → [H, W]
        img_arr_ = img_arr_.squeeze(0)

    plt.figure()
    plt.imshow(img_arr_, cmap="gray")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

@torch.no_grad()
def validate_and_save_samples(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_dir: str,
    epoch: int,
    solver_config: dict,
    writer: SummaryWriter = None,
    max_samples=16,
    class_map=None,
    mask_conditioning=True,
    class_conditioning=False,
    val=True,
    scale='linear',
    print_batch_results=False
):
    """Run your model on a handful of validation batches, 
    generate samples via the flow-matching solver, 
    save both generated and real images (and masks/classes),
    and report simple metrics (MAE, PSNR)."""
    import time
    start_time = time.time()
    model.eval()
    outdir = os.path.join(checkpoint_dir, f"val_samples_epoch_{epoch}")
    os.makedirs(outdir, exist_ok=True)
    
    # # Initialize FID metric only for validation
    # fid_metric = None
    # if writer is not None and val:
    #     try:
    #         from torchmetrics.image.fid import FrechetInceptionDistance
    #         fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    #     except Exception as e:
    #         print(f"[Validation] Warning: Could not initialize FID metric: {e}")
    #         fid_metric = None
            
    # Initialize metric lists
    mae_list = []
    psnr_list = []
    ssim_list = []
    count, step_plot_done = 0, False # sample counter
    batch_count = 0
    total_batches = len(val_loader)
    
    for batch in val_loader:
        batch_count += 1
        batch_mae_list = []
        batch_psnr_list = []
        batch_ssim_list = []
        
        imgs = batch["images"].to(device)
        cond = batch["classes"].to(device).unsqueeze(1).float() if class_conditioning else None
        # cond = batch["classes"].to(device).float() if class_conditioning else None
        masks = batch["masks"].to(device) if mask_conditioning else None

        x_init = torch.randn_like(imgs)
        sol = sample_with_solver(
            model,  # velocity model
            x_init, # [B,C,D,H,W]
            solver_config,  # test1.yaml - solver_args
            cond=cond,  # [B,1,classes_dim]
            masks=masks,    # [B,C,D,H,W]
            return_intermediates=True
        )
        final_imgs = sol[-1] if sol.dim() == 6 else sol
        for i in range(final_imgs.size(0)):
            if count >= max_samples:
                break

            # Compute MAE and PSNR on normalized images
            gen_arr = final_imgs[i].cpu().numpy() # [channel, D, H, W]
            real_arr = imgs[i].cpu().numpy()
            # Clip to [-1, 1]
            gen_arr = np.clip(gen_arr, -1, 1)
            real_arr = np.clip(real_arr, -1, 1)
            # Remove channel dimension for real_arr to match gen_arr
            if real_arr.ndim == 4 and real_arr.shape[0] == 1:
                real_arr = real_arr[0]

            # Restrict metrics to mask==1 region
            mask_np = batch["masks_"].cpu().numpy()  # shape [B,1,D,H,W]
            mask_vol = mask_np[i]
            if mask_vol.ndim == 4:
                mask_vol = mask_vol[0]  # drop channel dim -> [D,H,W]
            mask_bool = mask_vol > 0.5

            if scale == 'linear':
                # NOTE: linear scale
                # mapping from [-1, 1] to HU range [-1024, 3071]
                lower, upper = -1024.0, 3071.0
                range_half = (upper - lower) / 2.0
                midpoint = (upper + lower) / 2.0
                gen_arr = gen_arr * range_half + midpoint
                real_arr = real_arr * range_half + midpoint
            elif scale == 'sigmoid':
                # NOTE: sigmoid scale
                # Sigmoid inverse mapping from [-0.99,0.99] to HU range [-1024,3071]
                gen_arr = np.clip(gen_arr, -0.99, 0.99)
                real_arr = np.clip(real_arr, -0.99, 0.99)
                lower, upper = -1024.0, 3071.0
                p_low = 0.005
                x0 = (upper + lower) / 2.0
                logit_high = np.log((1 - p_low) / p_low)
                k = 2 * logit_high / (upper - lower)
                # invert the sigmoid normalization
                s_gen = (gen_arr + 1) / 2.0
                gen_arr = x0 + (1.0 / k) * np.log(s_gen / (1.0 - s_gen))
                s_real = (real_arr + 1) / 2.0
                real_arr = x0 + (1.0 / k) * np.log(s_real / (1.0 - s_real))
                # Clip to valid HU range
                gen_arr = np.clip(gen_arr, lower, upper)
                real_arr = np.clip(real_arr, lower, upper)
            elif scale == 'uniform':
                # Use QuantileTransformer objects from the dataset attached to val_loader
                qt_ct = val_loader.dataset.ct_qt
                # The QT class now handles shape preservation internally
                gen_arr = qt_ct.inverse(gen_arr)
                real_arr = qt_ct.inverse(real_arr)
            elif scale == 'uniform2':
                # Use GBoost objects from the dataset attached to val_loader
                gbdt_ct = val_loader.dataset.ct_gbdt
                gen_arr = gbdt_ct.inverse(gen_arr)
                real_arr = gbdt_ct.inverse(real_arr)
            elif scale == 'sigmoid2':
                # Use Sigmoid2 objects from the dataset attached to val_loader
                sig2_ct = val_loader.dataset.ct_sigmoid2
                gen_arr = sig2_ct.inverse(gen_arr)
                real_arr = sig2_ct.inverse(real_arr)
            else:
                raise ValueError("Invalid scale type. Choose 'linear', 'sigmoid', 'uniform', 'uniform2', or 'sigmoid2'.")

            # Set values outside mask to -1024 (lower bound)
            if gen_arr.ndim == 4 and gen_arr.shape[0] == 1:
                gen_arr = gen_arr[0]  # Remove channel dimension if present
            gen_arr[~mask_bool] = -1024.0  # Set non-mask regions to lower bound

            # Compute MAE and MSE within masked region
            diff = gen_arr - real_arr
            # Remove channel dimension for masking if needed
            if diff.ndim == 4 and diff.shape[0] == 1:
                diff = diff[0]  # shape now [D, H, W]

            # ---- BEGIN DIAGNOSTIC BLOCK ----
            # print(f"[Diagnostic][Sample {i}] gen_arr.shape={gen_arr.shape}, real_arr.shape={real_arr.shape}")
            # print(f"[Diagnostic][Sample {i}] gen_arr: min={np.nanmin(gen_arr)}, max={np.nanmax(gen_arr)}, mean={np.nanmean(gen_arr)}")
            # print(f"[Diagnostic][Sample {i}] real_arr: min={np.nanmin(real_arr)}, max={np.nanmax(real_arr)}, mean={np.nanmean(real_arr)}")
            # print(f"[Diagnostic][Sample {i}] gen_arr has nan: {np.isnan(gen_arr).any()}, inf: {np.isinf(gen_arr).any()}")
            # print(f"[Diagnostic][Sample {i}] real_arr has nan: {np.isnan(real_arr).any()}, inf: {np.isinf(real_arr).any()}")
            # print(f"[Diagnostic][Sample {i}] diff has nan: {np.isnan(diff).any()}, inf: {np.isinf(diff).any()}")
            # print(f"[Diagnostic][Sample {i}] mask_bool sum: {mask_bool.sum()}, unique values: {np.unique(mask_bool)}")
            if mask_bool.sum() == 0:
                print(f"\t--[WARNING][Sample {i}] mask_bool is all zeros; metrics will be NaN.")
            if np.isnan(gen_arr).any() or np.isnan(real_arr).any():
                print(f"\t--[WARNING][Sample {i}] NaN detected in gen_arr or real_arr!")
            if np.isinf(gen_arr).any() or np.isinf(real_arr).any():
                print(f"\t--[WARNING][Sample {i}] Inf detected in gen_arr or real_arr!")
            # ---- END DIAGNOSTIC BLOCK ----

            # Guard against division by zero for MAE/MSE, and skip metric logging if mask is all zeros
            if mask_bool.sum() == 0:
                print(f"\t--[WARNING][Sample {i}] Zero valid voxels in mask, skipping metric logging.")
                continue  # Don't append to lists
            else:
                mae = np.mean(np.abs(diff)[mask_bool])
                mse = np.mean((diff**2)[mask_bool])
            max_value = 4095.0  # Maximum possible value in HU range
            # Prevent log10 of zero by clamping MSE
            mse_clamped = max(mse, 1e-8)
            psnr = 10 * np.log10((max_value ** 2) / mse_clamped)

            # print(f"\t--[Diagnostic][Sample {i}] MAE: {mae}, MSE: {mse}, PSNR: {psnr}")

            # — extract a middle slice along depth (D) before computing SSIM —
            if gen_arr.ndim == 4:
                # gen_arr shape: [C, D, H, W]
                mid = gen_arr.shape[1] // 2
                gen_slice  = gen_arr[0, mid, :, :]
                real_slice = real_arr[0, mid, :, :] if real_arr.ndim == 4 else real_arr[mid, :, :]
            elif gen_arr.ndim == 3 and gen_arr.shape[0] != 1:
                # gen_arr shape: [D, H, W]
                mid = gen_arr.shape[0] // 2
                gen_slice  = gen_arr[mid, :, :]
                real_slice = real_arr[mid, :, :]
            else:
                # shapes [1,H,W] or [H,W]
                gen_slice  = np.squeeze(gen_arr)
                real_slice = np.squeeze(real_arr)

            # Log images to TensorBoard
            if writer is not None:
                # Normalize CT slices for visualization in [0,1]
                gen_slice_vis = (gen_slice + 1024.0) / 4095.0
                real_slice_vis = (real_slice + 1024.0) / 4095.0
                writer.add_image(f"{'Val' if val else 'Train'}/gen_sample_{count}", gen_slice_vis, epoch, dataformats='HW')
                writer.add_image(f"{'Val' if val else 'Train'}/real_sample_{count}", real_slice_vis, epoch, dataformats='HW')
                # Extract and normalize MRI slice for visualization
                mri_arr_np = masks[i].cpu().numpy()
                if mri_arr_np.ndim == 4:
                    mri_slice = mri_arr_np[0, mri_arr_np.shape[1] // 2]
                elif mri_arr_np.ndim == 3:
                    mri_slice = mri_arr_np[mri_arr_np.shape[0] // 2]
                else:
                    mri_slice = mri_arr_np.squeeze()
                mri_slice_vis = (np.clip(mri_slice, -1.0, 1.0) + 1.0) / 2.0
                writer.add_image(f"{'Val' if val else 'Train'}/mri_sample_{count}", mri_slice_vis, epoch, dataformats='HW')

            # Prepare SSIM on masked slice
            # Determine mask slice at mid-depth
            if gen_arr.ndim == 4:
                mid = gen_arr.shape[1] // 2
            else:
                mid = gen_arr.shape[0] // 2
            mask_slice = mask_vol[mid]
            ys, xs = np.where(mask_slice)
            if ys.size > 0:
                ymin, ymax = ys.min(), ys.max() + 1
                xmin, xmax = xs.min(), xs.max() + 1
                gen_crop = gen_slice[ymin:ymax, xmin:xmax]
                real_crop = real_slice[ymin:ymax, xmin:xmax]
                if min(gen_crop.shape) >= 7:
                    ssim_val = ssim(gen_crop, real_crop, data_range=4095)
                else:
                    ssim_val = np.nan
            else:
                ssim_val = np.nan

            mae_list.append(mae)
            psnr_list.append(psnr)
            ssim_list.append(ssim_val)
            
            # Add to batch lists for batch-level reporting
            batch_mae_list.append(mae)
            batch_psnr_list.append(psnr)
            batch_ssim_list.append(ssim_val)

            if class_map and "classes" in batch:
                idx = batch["classes"][i].argmax().item()
                with open(os.path.join(outdir, "class.json"), "w") as f:
                    json.dump(
                        {
                            "class_index": idx,
                            "class_name": class_map[idx] if idx < len(class_map) else str(idx),
                            "class_map": class_map,
                            "class_coditioning": class_conditioning,
                            "mask_conditioning": mask_conditioning,
                        },
                        f,
                        indent=4,
                    )
            count += 1
            
        # Print batch results if enabled
        if print_batch_results and batch_mae_list:
            batch_avg_mae = np.mean(batch_mae_list)
            batch_avg_psnr = np.mean(batch_psnr_list)
            batch_avg_ssim = np.nanmean(batch_ssim_list)
            print(f"[Batch {batch_count}/{total_batches}] Avg MAE: {batch_avg_mae:.4f}, Avg PSNR: {batch_avg_psnr:.2f} dB, Avg SSIM: {batch_avg_ssim:.4f}")
            
        if not step_plot_done:
            clz = batch["classes"] if class_map and "classes" in batch else None
            plot_solver_steps(sol, imgs, masks, clz, class_map, outdir)
            step_plot_done = True
        if count >= max_samples:
            break
    # After saving samples, report average metrics
    if mae_list and psnr_list and ssim_list:
        avg_mae = np.mean(mae_list)
        avg_psnr = np.mean(psnr_list)
        # Use nanmean to ignore nan values in SSIM
        avg_ssim = np.nanmean(ssim_list)
        num_total = len(ssim_list)
        num_valid = np.sum(~np.isnan(ssim_list))
        num_skipped = num_total - num_valid
        elapsed = time.time() - start_time
        if val:
            print(f"[Validation] Avg MAE: {avg_mae:.4f}, Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f} "
                  f"(valid: {num_valid}/{num_total}, skipped: {num_skipped}), Time: {elapsed:.2f}s")
        else:
            print(f"[Train] Avg MAE: {avg_mae:.4f}, Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f} "
                  f"(valid: {num_valid}/{num_total}, skipped: {num_skipped}), Time: {elapsed:.2f}s")

    return avg_mae, avg_psnr, avg_ssim


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    solver_config: dict,
    batch: torch.Tensor,
    device: torch.device,
    class_conditioning: bool = False,
    mask_conditioning: bool = False,
):
    model.eval()
    imgs = batch["images"].to(device)
    cond = batch["classes"].to(device).unsqueeze(1) if class_conditioning else None
    masks = batch["masks"].to(device) if mask_conditioning else None

    x_init = torch.randn_like(imgs)
    sol = sample_with_solver(
        model=model, solver_config=solver_config, x_init=x_init, cond=cond, masks=masks
    )
    # final_imgs = sol[-1] if sol.dim() == 5 else sol
    
    # sol: [T, B, C, D, H, W]  or  [T, B, C, H, W]
    if sol.dim() >= 5:
        final_imgs = sol[-1]       # -> [B, C, D, H, W] or [B, C, H, W]
    else:
        final_imgs = sol           # fallback, but for 2D case you'll still get [B, C, H, W]
    # after final_imgs is [B, C, D, H, W]:
    if final_imgs.dim() == 5:
        D = final_imgs.shape[2]
        mid = D // 2
        # now shrink to [B, C, H, W]:
        final_imgs = final_imgs[:, :, mid, :, :]
    return final_imgs
