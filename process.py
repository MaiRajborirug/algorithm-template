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
config_path = "/opt/algorithm/exp_configs/test_apex_1.yaml"
inference_path = "/input"  # (if you want to use the mounted test data, adjust as needed)
latest_ckpt_dir = "/opt/algorithm/some_checkpoints/apex-dist3-u2lf-HN"

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
    torch.device('cuda') if torch.cuda.is_available()
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




# %% original code -------------
class SynthradAlgorithm(BaseSynthradAlgorithm):
    """
    This class implements a simple synthetic CT generation algorithm that segments all values greater than 2 in the input image.

    Author: Suraj Pai (b.pai@maastrichtuniversity.nl)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, input_dict: Dict[str, sitk.Image]) -> sitk.Image:
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

        ## check if GPU is available
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("Using device: ", device)

        ## convert np arrays to tensors
        # mr_tensor = torch.tensor(mr_np, device=device)
        # mask_tensor = torch.tensor(mask_np, device=device)

        ## sCT generation placeholder (set values inside mask to 0)
        # mr_tensor[mask_tensor == 1] = 0
        # mr_tensor[mask_tensor == 0] = -1000

        ## convert tensor back to np array
        # sCT = mr_tensor.cpu().numpy()

        
        # NOTE: Comment the following lines if using pytorch
        sCT = np.zeros(mr_np.shape)
        sCT[mask_np == 1] = 0
        sCT[mask_np == 0] = -1000


        sCT_sitk = sitk.GetImageFromArray(sCT)
        sCT_sitk.CopyInformation(mr_sitk)

        return sCT_sitk


if __name__ == "__main__":
    # Run the algorithm on the default input and output paths specified in BaseSynthradAlgorithm.
    SynthradAlgorithm().process()
