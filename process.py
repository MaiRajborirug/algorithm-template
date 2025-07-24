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

# %%

# Experiment configuration
exp_name = "x"
SAMPLING_QUALITY = "fast"  # Options: "fast" (10 steps), "medium" (50 steps), "high" (100 steps), "ultra" (200 steps)
SAMPLING_STEPS = {
    "fast": 10,
    "medium": 50, 
    "high": 100,
    "ultra": 200
}
# Paths - docker location

# config_path = "/opt/algorithm/exp_configs/test_apex_1.yaml"
# inference_path = "/input"  # (if you want to use the mounted test data, adjust as needed)
# latest_ckpt_dir = "/opt/algorithm/some_checkpoints/apex-dist3-u2lf-HN"

config_path = "./exp_configs/test_apex_1.yaml"
inference_path = "./test_mha/images"
latest_ckpt_dir = "./some_checkpoints/apex-dist3-llf-HN"
npy_root = "./some_checkpoints"


# %% Model initialization
CT_UPPER = 3071.0
CT_LOWER = -1024.0
MRI_UPPER = 1357.0
MRI_LOWER = 0.0
PATCH_SIZE = (16, 128, 128)
OVERLAP_SIZE = (4, 32, 32)

print("Initiate model and load weights")

# %% Inference classes

# %% original code -------------
class SynthradAlgorithm(BaseSynthradAlgorithm):
    """
    This class implements a simple synthetic CT generation algorithm that segments all values greater than 2 in the input image.

    Author: Suraj Pai (b.pai@maastrichtuniversity.nl)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model and weight initialization moved here
        print("Initiate model and load weights (in __init__)")
        self.config = load_config(config_path)
        # Read core settings from config
        self.num_epochs = self.config["train_args"]["num_epochs"]
        self.batch_size = self.config["train_args"]["batch_size"]    
        self.lr = self.config["train_args"]["lr"] * self.batch_size
        # Device selection
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        print("Using device:", self.device)
        # Model configuration flags
        self.mask_conditioning = self.config["general_args"]["mask_conditioning"]
        self.class_conditioning = self.config["general_args"]["class_conditioning"]
        # Random seed for reproducibility
        seed = self.config["general_args"].get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Create one Generator for DataLoader shuffling:
            self.dl_gen = torch.Generator()
            self.dl_gen.manual_seed(seed)
        else:
            self.dl_gen = None
        # Build model with condition
        model_cfg = self.config["model_args"].copy()
        model_cfg["mask_conditioning"] = self.config["general_args"]["mask_conditioning"]
        self.model = build_model(model_cfg, device=self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Load the latest checkpoint if available
        self.start_epoch, self.loaded_config = load_checkpoint(
            self.model, self.optimizer, checkpoint_dir=latest_ckpt_dir, device=self.device, valid_only=False
        )
        # Ensure model is in float32
        self.model = self.model.float()
        # Define path object (scheduler included)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.solver_config = self.config["solver_args"]
        da = self.config['data_args']
        self.scale = da["scale"]
        self.scale_mask = da["scale_mask"]
        self.npy_root = npy_root
        self.seed = seed


        self.patch_size = PATCH_SIZE 
        self.overlap = OVERLAP_SIZE 

        # initialize data scaling
        self._apply_scaling_init()

    
    def _apply_scaling_init(self):
        """Apply the same scaling as in training."""
        # CT scaling
        if self.scale == 'uniform':
            self.ct_scale = QT(file_path=os.path.join(self.npy_root, "ct_mask1.npy"), seed=self.seed)
            # self.ct_arr = ct_qt.forward(self.ct_arr)
        elif self.scale == 'uniform2':
            self.ct_scale = GBoost(file_path=os.path.join(self.npy_root, "ct_mask1.npy"), seed=self.seed)
            # self.ct_arr = ct_gbdt.forward(self.ct_arr)
        
        # MRI scaling
        if self.scale_mask == 'uniform':
            self.mri_scale = QT(file_path=os.path.join(self.npy_root, "mri_mask1.npy"), seed=self.seed)
            # self.mri_arr = mri_qt.forward(self.mri_arr)
        elif self.scale_mask == 'uniform2':
            self.mri_scale = GBoost(file_path=os.path.join(self.npy_root, "mri_mask1.npy"), seed=self.seed)
            # self.mri_arr = mri_gbdt.forward(self.mri_arr)

    def _generate_overlapping_patches(self, volume_shape, patch_size, overlap):
        """Generate overlapping patch coordinates covering the entire volume."""
        H, W, D = volume_shape
        ph, pw, pd = patch_size
        oh, ow, od = overlap

        # Calculate step sizes
        step_h = ph - oh
        step_w = pw - ow
        step_d = pd - od

        patches = []

        for h in range(0, H, step_h):
            for w in range(0, W, step_w):
                for d in range(0, D, step_d):
                    patches.append({
                        'coords': (h, w, d),
                        'size': (ph, pw, pd)
                    })
        return patches

    def predict(self, input_dict: Dict[str, sitk.Image]) -> sitk.Image:
        mr_np = sitk.GetArrayFromImage(input_dict["image"]).astype("float32")
        mask_np = sitk.GetArrayFromImage(input_dict["mask"]).astype("float32")

        patches = self._generate_overlapping_patches(mr_np.shape, self.patch_size, self.overlap)

        output_np = np.zeros_like(mr_np)
        count_np = np.zeros_like(mr_np)

        for patch_info in patches:
            h, w, d = patch_info['coords']
            ph, pw, pd = patch_info['size']

            end_h = min(h + ph, mr_np.shape[0])
            end_w = min(w + pw, mr_np.shape[1])
            end_d = min(d + pd, mr_np.shape[2])

            mr_patch = mr_np[h:end_h, w:end_w, d:end_d]
            mask_patch = mask_np[h:end_h, w:end_w, d:end_d]

            # Pad if necessary
            pad_h = ph - mr_patch.shape[0]
            pad_w = pw - mr_patch.shape[1]
            pad_d = pd - mr_patch.shape[2]
            if pad_h > 0 or pad_w > 0 or pad_d > 0:
                mr_patch = np.pad(mr_patch, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
                mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')

            # Predict patch
            pred_patch = self._predict_patch(mr_patch, mask_patch)

            # Remove padding if any
            pred_patch = pred_patch[:end_h-h, :end_w-w, :end_d-d]

            # Place predicted patch in output array
            output_np[h:end_h, w:end_w, d:end_d] += pred_patch
            count_np[h:end_h, w:end_w, d:end_d] += 1

        output_np = output_np / np.maximum(count_np, 1)
        output_sitk = sitk.GetImageFromArray(output_np)
        output_sitk.CopyInformation(input_dict["image"])
        return output_sitk

    def _predict_patch(self, mr_patch, mask_patch):
        # Convert to torch tensors and add batch/channel dimensions
        mr_tensor = torch.tensor(mr_patch, device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        mask_tensor = torch.tensor(mask_patch, device=self.device).unsqueeze(0).unsqueeze(0)

        # Model prediction (adjust as needed for your model)
        with torch.no_grad():
            pred_patch = self.model(mr_tensor, mask_tensor)  # adjust if your model expects different input
            pred_patch = pred_patch.squeeze().cpu().numpy()
        return pred_patch

    def predict_simple(self, input_dict: Dict[str, sitk.Image]) -> sitk.Image:
        """
        Generates a synthetic CT image from the given input image and mask.

        Parameters
        ----------
        input_dict : Dict[str, SimpleITK.Image]
            A dictionary containing two keys: "image" and "mask". The value for each key is a SimpleITK.Image object representing the input image and mask respectively.

        Returns
        -------
        SimpleITK.Image
            The generated synthetic CT image.

        Raises
        ------
        AssertionError:
            If the keys of `input_dict` are not ["image", "mask"]
        """
        assert list(input_dict.keys()) == ["image", "mask", "region"]

        # You may use the region information to generate the synthetic CT image if needed 
        region = input_dict["region"]
        print("Scan region: ", region)
        mr_sitk = input_dict["image"]
        mask_sitk = input_dict["mask"]

        # convert sitk images to np arrays
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype("float32")
        mr_np = sitk.GetArrayFromImage(mr_sitk).astype("float32")



        # NOTE: To test using pytorch, uncomment the following lines and comment the lines below

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

        
        # # NOTE: Comment the following lines if using pytorch
        # sCT = np.zeros(mr_np.shape)
        # sCT[mask_np == 1] = 0
        # sCT[mask_np == 0] = -1000


        sCT_sitk = sitk.GetImageFromArray(sCT)
        sCT_sitk.CopyInformation(mr_sitk) # copies spatial metadata (origin, spacing, direction)

        return sCT_sitk




if __name__ == "__main__":
    # Run the algorithm on the default input and output paths specified in BaseSynthradAlgorithm.
    SynthradAlgorithm().process()
