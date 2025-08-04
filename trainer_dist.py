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

# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # os.path.dirname(__file__): parent dir ../MOTFM/  + join(MOTFM, ..) = any directory in MOTFM

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
class MhaDataset(Dataset):
    """ for full dataset """
    def __init__(self, root_dir, transform=None):
        anatomy_code = os.path.basename(root_dir)
        glob_pattern = f"1{anatomy_code}*"
        self.sample_dirs = sorted(glob.glob(os.path.join(root_dir, glob_pattern))) # list of all sibdor in root_dir
        self.transform = transform
    def __len__(self):
        return len(self.ct_arrs)
    def __getitem__(self, idx): # full image
        d = self.sample_dirs[idx]
        ct_path   = os.path.join(d, "ct.mha")
        mri_path  = os.path.join(d, "mr.mha")
        mask_path = os.path.join(d, "mask.mha")
        # label_path= 
        # with open(label_path) as f:
        #     lbl = int(f.read().strip())
        lbl = int(0)
        
        # see test1.ipynb to learn more about 'sitk'
        ct_arr   = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
        mri_arr  = sitk.GetArrayFromImage(sitk.ReadImage(mri_path)).astype(np.float32)

        ct_tensor   = torch.from_numpy(ct_arr)
        mask_tensor = torch.from_numpy(mask_arr)
        mri_tensor  = torch.from_numpy(mri_arr)
        class_tensor= torch.tensor(lbl, dtype=torch.long)
        if self.transform:
            ct_tensor, mask_tensor = self.transform(ct_tensor, mask_tensor)
        return {
            "ct": ct_tensor,
            "mri":  mri_tensor,
            "masks":  mask_tensor,
            "classes": class_tensor,
        }
        
class MhaDataset_small(Dataset):
    """Represents the full dataset of MHA files."""
    def __init__(self, 
                 root_dir,
                 transform=None,
                 n_images=65,
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
    parser = argparse.ArgumentParser(description="Train the flow matching model.")
    parser.add_argument(
        "--config_path",
        type=str,
        # default="configs/default.yaml",
        default="configs/test1.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config_path = args.config_path
    config = load_config(config_path)

    # Read core settings from config
    num_epochs = config["train_args"]["num_epochs"]
    batch_size = config["train_args"]["batch_size"]    
    lr = config["train_args"]["lr"] * batch_size
    print_every = config["train_args"].get("print_every", 1)
    val_freq = config["train_args"].get("val_freq", 5)
    num_val_samples = config["train_args"].get("num_val_samples", 5)
    root_ckpt_dir = config["train_args"]["checkpoint_dir"]
    use_masked_loss = config["train_args"].get("use_masked_loss", False)
    print(f"train_args - use_masked_loss: {use_masked_loss}")

    # ====== DDP ENVIRONMENT VARIABLES ======
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    # ====== DDP INIT ======
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ====== Logging & TensorBoard (rank 0 only) ======
    if rank == 0:
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        experiment_name = os.path.basename(root_ckpt_dir.rstrip("/"))
        # Save config file for reproducibility
        exp_config_dir = os.path.join(os.path.dirname(__file__), "exp_configs")
        os.makedirs(exp_config_dir, exist_ok=True)
        exp_config_path = os.path.join(exp_config_dir, f"{experiment_name}.yaml")
        shutil.copy2(config_path, exp_config_path)
        print("Experiment config saved to:", exp_config_path)
        writer = SummaryWriter(log_dir=os.path.join(logs_dir, experiment_name))
        print("TensorBoard log files in:", writer.log_dir)
        # Log config as text to TensorBoard
        with open(config_path, "r") as f:
            config_text = f.read()
        writer.add_text("experiment_config", f"```yaml\n{config_text}\n```", 0)
        print("Using device:", device)
    else:
        writer = None

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
    
    # ====== Build model and wrap with DDP ======
    model_cfg = config["model_args"].copy()
    model_cfg["mask_conditioning"] = config["general_args"]["mask_conditioning"]
    model = build_model(model_cfg, device=device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


    # # Prepare data -- removed for now -- see 'trainer.py'
    
    # NOTE: new data loader
    # Load sampling params from config
    da = config["data_args"]
    train_dataset = MhaDataset_small(
        root_dir=da["train_root"],
        n_images=da["train_n_images"],
        n_patches=da["train_n_patches"],
        reverse=da["train_reverse"],
        scale=da["scale"],
        scale_mask=da["scale_mask"],
        npy_root=da["npy_root"],
        seed=seed,  # Use the same seed for reproducibility
    )
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
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=4)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=2)
    

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load the latest checkpoint if available
    latest_ckpt_dir = os.path.join(root_ckpt_dir, "latest")
    start_epoch, loaded_config = load_checkpoint(
        model, optimizer, checkpoint_dir=latest_ckpt_dir, device=device, valid_only=False
    )

    # Define path object (scheduler included)
    path = AffineProbPath(scheduler=CondOTScheduler())

    solver_config = config["solver_args"]

    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs", disable=not is_interactive):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        # Use tqdm for the train loader to get a per-batch progress bar
        if seed is not None:
            dl_gen.manual_seed(seed + epoch) # epoch i: use = seed + i (for reproducibility)
        for batch in train_loader:
            im_batch = batch["images"].to(device) # [B, 1, Z, H, W]
            mri_batch = batch["masks"].to(device)
            classes_batch = batch["classes"].to(device)
            mask_batch = batch["masks_"].to(device)

            # Sample random initial noise, and random t
            x_0 = torch.randn_like(im_batch)
            t = torch.rand(im_batch.shape[0], device=device)

            # Sample the path from x_0 to the conditional image (CT+MRI)
            sample_info = path.sample(t=t, x_0=x_0, x_1=im_batch)
            cond_tensor = classes_batch.unsqueeze(1).float()

            # Model forward pass with debug prints and try-except
            try:
                v_pred = model(x=sample_info.x_t, t=sample_info.t, cond=cond_tensor, masks=mri_batch)
            except Exception as e:
                print(f"[Rank {rank}] Model forward error: {e}")
                raise

            if use_masked_loss:
                # build binary mask and broadcast to match prediction shape
                mb = (mask_batch > 0.5).float()
                if mb.dim() < v_pred.dim():
                    mb = mb.expand_as(v_pred)
                # compute masked MSE using optimized mse_loss
                loss_sum = F.mse_loss(v_pred * mb, sample_info.dx_t * mb, reduction='sum')
                loss = loss_sum / mb.sum().clamp(min=1.0)
            else:
                loss = F.mse_loss(v_pred, sample_info.dx_t)

            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Print max gradient value after clipping (rank 0 only)
            if rank == 0:
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm = max(grad_norm, p.grad.data.abs().max().item())
                # print(f"[Rank {rank}] Max grad after clipping: {grad_norm}")
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % print_every == 0 and rank == 0:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}, Time: {elapsed:.2f}s")
        if writer is not None:
            writer.add_scalar("Loss/train", avg_loss, epoch + 1)

        # Validation & checkpoint saving
        if (epoch + 1) % val_freq == 0:
            epoch_ckpt_dir = os.path.join(root_ckpt_dir, f"epoch_{epoch+1}")
            # NOTE: To save the disk space, we save only the latest checkpoint
            # if (epoch + 1) % (val_freq) == 0 and rank == 0:
            #     save_checkpoint(model, optimizer, epoch + 1, config, epoch_ckpt_dir)
            #     save_checkpoint(model, optimizer, epoch + 1, config, latest_ckpt_dir)
            #     print(f"Checkpoint saved at {epoch_ckpt_dir} and latest checkpoint updated.")

            # Compute & log training metrics (rank 0 only)
            if rank == 0:
                save_checkpoint(model, optimizer, epoch + 1, config, epoch_ckpt_dir)
                save_checkpoint(model, optimizer, epoch + 1, config, latest_ckpt_dir)
                print(f"Checkpoint saved at {epoch_ckpt_dir} and latest checkpoint updated.")
                train_mae, train_psnr, train_ssim = validate_and_save_samples(
                    model=model,
                    val_loader=train_loader,
                    device=device,
                    checkpoint_dir=epoch_ckpt_dir,
                    epoch=epoch + 1,
                    solver_config=solver_config,
                    writer=writer,
                    max_samples=num_val_samples,
                    class_map=None,
                    mask_conditioning=mask_conditioning,
                    class_conditioning=class_conditioning,
                    val=False,
                    scale=da["scale"],
                )
                writer.add_scalar("MAE/train", train_mae, epoch + 1)
                writer.add_scalar("PSNR/train", train_psnr, epoch + 1)
                writer.add_scalar("SSIM/train", train_ssim, epoch + 1)

                val_mae, val_psnr, val_ssim = validate_and_save_samples(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    checkpoint_dir=epoch_ckpt_dir,
                    epoch=epoch + 1,
                    solver_config=solver_config,
                    writer=writer,
                    max_samples=num_val_samples,
                    class_map={0: "AB", 1: "HN", 2: "TH"},
                    mask_conditioning=mask_conditioning,
                    class_conditioning=class_conditioning,
                    val=True,
                    scale=da["scale"],
                )
                writer.add_scalar("MAE/val", val_mae, epoch + 1)
                writer.add_scalar("PSNR/val", val_psnr, epoch + 1)
                writer.add_scalar("SSIM/val", val_ssim, epoch + 1)

    if writer is not None:
        writer.close()
    if rank == 0:
        print("Training complete!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
