import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
import glob
import json
import multiprocessing as mp
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.general_utils import load_config

# Reuse scaling helpers if needed
from utils.scaling import GBoost, Sigmoid2, QT

# ----------------------- Patch extraction worker -----------------------
import SimpleITK as sitk

def process_single_dir_worker(d, scale, scale_mask, npy_root, seed, n_patches, patch_size):
    import SimpleITK as sitk
    import numpy as np
    import os

    local_rng = np.random.default_rng(seed)

    root_dir = os.path.dirname(d)
    if root_dir.split('/')[-1] == 'AB':
        lbl = np.array([1, 0, 0], dtype=np.float32)
    elif root_dir.split('/')[-1] == 'HN':
        lbl = np.array([0, 1, 0], dtype=np.float32)
    elif root_dir.split('/')[-1] == 'TH':
        lbl = np.array([0, 0, 1], dtype=np.float32)
    else:
        raise ValueError(f"Unknown organ in path: {root_dir}")

    ct_path   = os.path.join(d, "ct.mha")
    mri_path  = os.path.join(d, "mr.mha")
    mask_path = os.path.join(d, "mask.mha")

    def reorient_to_RAS(img):
        dicom_orient = sitk.DICOMOrientImageFilter()
        dicom_orient.SetDesiredCoordinateOrientation("RAS")
        return dicom_orient.Execute(img)

    ct_arr   = sitk.GetArrayFromImage(reorient_to_RAS(sitk.ReadImage(ct_path))).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(reorient_to_RAS(sitk.ReadImage(mask_path))).astype(np.float32)
    mri_arr  = sitk.GetArrayFromImage(reorient_to_RAS(sitk.ReadImage(mri_path))).astype(np.float32)

    ct_max = float(np.max(ct_arr))
    ct_min = float(np.min(ct_arr))
    mri_max = float(np.max(mri_arr))

    if scale == 'linear':
        lower, upper = -1024.0, 3071.0
        midpoint = (upper + lower) / 2.0
        range_half = (upper - lower) / 2.0
        ct_arr = np.clip(ct_arr, lower, upper)
        ct_arr = (ct_arr - midpoint) / range_half
    elif scale == 'sigmoid':
        lower, upper = -1024.0, 3071.0
        ct_arr = np.clip(ct_arr, lower, upper)
        p_low = 0.005
        x0 = (upper + lower) / 2.0
        logit_high = np.log((1 - p_low) / p_low)
        k = 2 * logit_high / (upper - lower)
        s = 1 / (1 + np.exp(-k * (ct_arr - x0)))
        ct_arr = 2 * s - 1
    elif scale in ['sigmoid2', 'uniform', 'uniform2']:
        pass
    else:
        raise ValueError(f"Unknown scale type: {scale}")

    MRI_MAX = np.percentile(mri_arr, 99.0)
    mri_arr = np.clip(mri_arr, 0, MRI_MAX)
    if scale_mask == 'linear':
        mri_arr = (mri_arr - (MRI_MAX / 2)) / (MRI_MAX / 2)
    elif scale_mask == 'sigmoid':
        lower, upper = 0.0, MRI_MAX
        p_low = 0.005
        x0 = (upper + lower) / 2.0
        k = -np.log((1/p_low) - 1) / ((upper - lower) / 2)
        mri_arr = 2.0 / (1.0 + np.exp(-k * (mri_arr - x0))) - 1.0
        mri_arr = mri_arr * 0.99
    elif scale_mask in ['uniform', 'uniform2']:
        pass
    else:
        raise ValueError(f"Unknown scale type: {scale_mask}")

    ct_arr = ct_arr.astype(np.float32)
    mask_arr = mask_arr.astype(np.float32)
    mri_arr = mri_arr.astype(np.float32)

    def extract_random_patch_local(volume_shape, patch_size, rng):
        H, W, D = volume_shape
        ph, pw, pd = patch_size
        if H < ph or W < pw or D < pd:
            raise ValueError("Volume shape is smaller than patch size.")
        start_h = rng.integers(0, H - ph + 1)
        start_w = rng.integers(0, W - pw + 1)
        start_d = rng.integers(0, D - pd + 1)
        return start_h, start_w, start_d

    ct_patches, mask_patches, mri_patches, labels = [], [], [], []

    for _ in range(n_patches):
        sh, sw, sd = extract_random_patch_local(ct_arr.shape, patch_size, local_rng)
        ct_patch = ct_arr[sh:sh+patch_size[0], sw:sw+patch_size[1], sd:sd+patch_size[2]]
        mask_patch = mask_arr[sh:sh+patch_size[0], sw:sw+patch_size[1], sd:sd+patch_size[2]]
        mri_patch = mri_arr[sh:sh+patch_size[0], sw:sw+patch_size[1], sd:sd+patch_size[2]]
        if mask_patch.sum() == 0:
            continue
        ct_patches.append(ct_patch.astype(np.float32))
        mask_patches.append(mask_patch.astype(np.float32))
        mri_patches.append(mri_patch.astype(np.float32))
        labels.append(lbl)

    return {
        "ct_patches": ct_patches,
        "mask_patches": mask_patches,
        "mri_patches": mri_patches,
        "labels": labels,
        "ct_maxs": [ct_max] * len(ct_patches),
        "ct_mins": [ct_min] * len(ct_patches),
        "mri_maxs": [mri_max] * len(ct_patches),
    }

# ----------------------- Discovery utilities -----------------------

def discover_sample_dirs(root_dir: str, organs: List[str]) -> Dict[str, List[str]]:
    root_basename = os.path.basename(root_dir.rstrip("/"))
    possible = ["AB", "HN", "TH"]
    if organs is None:
        if root_basename in possible:
            organ_root_map = {root_basename: root_dir}
        else:
            present = [o for o in possible if os.path.isdir(os.path.join(root_dir, o))]
            organ_root_map = {o: os.path.join(root_dir, o) for o in present}
    else:
        organ_root_map = {}
        for o in organs:
            if root_basename in possible:
                organ_root_map[o] = root_dir if o == root_basename else None
            else:
                organ_root_map[o] = os.path.join(root_dir, o)
        organ_root_map = {o: p for o, p in organ_root_map.items() if p and os.path.isdir(p)}

    result = {}
    for organ, organ_path in organ_root_map.items():
        pattern = os.path.join(organ_path, f"1{organ}*")
        result[organ] = sorted(glob.glob(pattern))
    return result

# ----------------------- Saving helpers -----------------------

def save_shard(out_file: str, data: dict):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # Convert to torch tensors and stack
    def to_tensor_stack(list_np):
        if len(list_np) == 0:
            return torch.empty(0)
        return torch.from_numpy(np.stack(list_np, axis=0)).contiguous()

    shard = {
        "ct": to_tensor_stack(data["ct_patches"]),
        "mask": to_tensor_stack(data["mask_patches"]),
        "mri": to_tensor_stack(data["mri_patches"]),
        "labels": torch.from_numpy(np.stack(data["labels"], axis=0)).float() if len(data["labels"]) > 0 else torch.empty(0, 3),
        "ct_maxs": torch.tensor(data["ct_maxs"], dtype=torch.float32),
        "ct_mins": torch.tensor(data["ct_mins"], dtype=torch.float32),
        "mri_maxs": torch.tensor(data["mri_maxs"], dtype=torch.float32),
    }
    torch.save(shard, out_file)

# ----------------------- Main preprocess -----------------------

def preprocess_split(split_name: str, root_dir: str, organs: List[str], n_images: int, n_patches: int, scale: str, scale_mask: str, npy_root: str, seed: int, patch_size=(16,128,128), out_root: str=None):
    rng = np.random.default_rng(seed)
    organ_to_dirs = discover_sample_dirs(root_dir, organs)

    index = {"shards": [], "split": split_name, "patch_size": list(patch_size)}
    out_split_dir = os.path.join(out_root, split_name)
    os.makedirs(out_split_dir, exist_ok=True)

    # Build list of all dirs across organs
    all_dirs = []
    for organ, dirs in organ_to_dirs.items():
        all_dirs.extend([(organ, d) for d in dirs])

    if len(all_dirs) == 0:
        raise RuntimeError(f"No sample directories found under {root_dir}")

    n_select = min(n_images, len(all_dirs))
    chosen_idx = rng.choice(len(all_dirs), size=n_select, replace=False)
    chosen = [all_dirs[i] for i in chosen_idx]

    # Parallel map per directory
    args_list = []
    for organ, d in chosen:
        args_list.append((d, scale, scale_mask, npy_root, int(rng.integers(0, 1_000_000)), n_patches, patch_size))

    n_workers = min(mp.cpu_count(), max(1, len(args_list)))
    print(f"[{split_name}] Using {n_workers} workers to preprocess {len(args_list)} images...")

    start = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(process_single_dir_worker, args_list)

    for (organ, d), data in zip(chosen, results):
        base = os.path.basename(d.rstrip('/'))
        shard_path = os.path.join(out_split_dir, organ, f"{base}.pt")
        save_shard(shard_path, data)
        num_patches = int(data["labels"].__len__())
        index["shards"].append({
            "file": shard_path,
            "num_patches": num_patches,
            "organ": organ,
            "image_dir": d,
            "basename": base,
        })

    index_path = os.path.join(out_root, f"{split_name}_index.json")
    with open(index_path, 'w') as f:
        json.dump(index, f)

    elapsed = time.time() - start
    total_patches = sum(s["num_patches"] for s in index["shards"])
    print(f"[{split_name}] Done. {len(index['shards'])} shards, {total_patches} patches in {elapsed:.2f}s. Index -> {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess 3D MHA into patch shards and save index")
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)

    seed = config["general_args"].get("seed", 0)

    da = config["data_args"]
    organs = da.get("organs", None)
    pre_dir = da.get("preprocessed_dir", None)
    if pre_dir is None:
        raise ValueError("data_args.preprocessed_dir must be set in config for preprocessing output")

    os.makedirs(pre_dir, exist_ok=True)

    # Train split
    preprocess_split(
        split_name="train",
        root_dir=da["train_root"],
        organs=organs,
        n_images=da["train_n_images"],
        n_patches=da["train_n_patches"],
        scale=da["scale"],
        scale_mask=da["scale_mask"],
        npy_root=da["npy_root"],
        seed=seed,
        patch_size=(16,128,128),
        out_root=pre_dir,
    )

    # Val split
    preprocess_split(
        split_name="val",
        root_dir=da["val_root"],
        organs=organs,
        n_images=da["val_n_images"],
        n_patches=da["val_n_patches"],
        scale=da["scale"],
        scale_mask=da["scale_mask"],
        npy_root=da["npy_root"],
        seed=seed + 1 if seed is not None else 0,
        patch_size=(16,128,128),
        out_root=pre_dir,
    )

    print("Preprocessing complete. Shards and indexes saved to:", pre_dir)

if __name__ == "__main__":
    main() 