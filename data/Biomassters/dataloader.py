# dataloader.py

import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from skimage import io

from data_transforms import calculate_veg_indices_uint8, train_aug

# --- constants for normalization ---
L = 0.5
epsilon = 1e-6

s1_min = np.array([-25, -62, -25, -60], dtype="float32")
s1_max = np.array([ 29,  28,  30,  22], dtype="float32")
s1_mm  = s1_max - s1_min

# s2_max = np.array([
#     19616., 18400., 17536., 17097., 16928., 16768., 16593., 16492.,
#     15401., 15226.,    1., 10316.,  8406.,    1.,   255.
# ], dtype="float32")

s2_max = np.array(
    [19616., 18400., 17536., 17097., 16928., 16768., 16593., 16492., 15401., 15226., 255.],
    dtype="float32",
)

IMG_SIZE = (256, 256)

# --- helper to read and normalize Sentinel-1 & Sentinel-2 chips ---
def read_imgs(chip_id, data_dir, veg_indices=False):
    # Ensure data_dir is a Path object
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    imgs, mask = [], []
    for month in range(12):
        img_s1 = io.imread(data_dir / f"{chip_id}_S1_{month:0>2}.tif")
        m = img_s1 == -9999
        img_s1 = img_s1.astype("float32")
        img_s1 = (img_s1 - s1_min) / s1_mm
        img_s1 = np.where(m, 0, img_s1)
        filepath = data_dir / f"{chip_id}_S2_{month:0>2}.tif"
        if filepath.is_file():
            img_s2 = io.imread(filepath)
            img_s2 = img_s2.astype("float32")

            main_channels = img_s2[:, :, :-1]
            transparency_channel = img_s2[:, :, -1:]

            if veg_indices:
                veg_indices_uint8 = calculate_veg_indices_uint8(img_s2)
                img_s2 = main_channels
                for index_name, index_array in veg_indices_uint8.items():
                    if np.isnan(index_array).any():
                        index_array = np.nan_to_num(index_array, nan=0.0)
                    index_array = np.expand_dims(index_array, axis=2)
                    img_s2 = np.concatenate([img_s2, index_array], axis=2)
                    # print(f"{index_name} max: {np.max(index_array)}, {np.count_nonzero(np.isnan(index_array))}")

                img_s2 = np.concatenate([img_s2, transparency_channel], axis=2)
                # print(f"Before Normalisation: {np.max(img_s2)}")

            img_s2 = img_s2 / s2_max

            # print(f"After Normalisation: {np.max(img_s2)}")
            
        else:
            if veg_indices:
                img_s2 = np.zeros(IMG_SIZE + (15,), dtype="float32")
            else:
                img_s2 = np.zeros(IMG_SIZE + (11,), dtype="float32")

        img = np.concatenate([img_s1, img_s2], axis=2)
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)
        mask.append(False)

    mask = np.array(mask)



    imgs = np.stack(imgs, axis=0)  # [t, c, h, w]

    return imgs, mask


class SatImDataset(Dataset):
    """
    Expects a DataFrame with a 'chip_id' column,
    a features folder, optional labels folder,
    and flags for augmentations & veg‐indices.
    """

    def __init__(self, df, dir_features, dir_labels=None,
                 augs=False, veg_indices=False):
        self.df = df.reset_index(drop=True)
        self.dir_features = dir_features
        self.dir_labels   = dir_labels
        self.augs         = augs
        self.veg_indices  = veg_indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        imgs, mask = read_imgs(item.chip_id, self.dir_features, self.veg_indices)

        if self.dir_labels:
            tgt = io.imread(Path(self.dir_labels) / f"{item.chip_id}_agbm.tif")
        else:
            tgt = item.chip_id  # for pseudo‐test mode

        if self.augs:
            imgs, mask, tgt = train_aug(imgs, mask, tgt)

        return imgs, mask, tgt


def get_dataloader(df, dir_features, dir_labels=None,
                   augs=False, veg_indices=False,
                   batch_size=32, num_workers=4,
                   shuffle=True):
    ds = SatImDataset(df, dir_features, dir_labels, augs, veg_indices)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True
    )


def get_distributed_dataloader(df, dir_features, dir_labels=None,
                               augs=False, veg_indices=False,
                               rank=0, world_size=1,
                               batch_size=32, num_workers=4,
                               shuffle=True):
    ds = SatImDataset(df, dir_features, dir_labels, augs, veg_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=world_size, rank=rank
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, sampler=sampler
    )


if __name__ == "__main__":
    # This is a test block to demonstrate how to use the SatImDataset and get_dataloader.

    # The project root is 4 levels up from this script.
    # biomassters/DeepSatModels/data/Biomassters/dataloader.py
    project_root = Path(__file__).resolve().parents[3]

    # As per user instruction:
    # metadata: data/features_metadata.csv
    # train_features: data/train_feature/train_features
    # train_agbm: data/train_agbm/train_agbm
    # test_features: data/test_features/test_features
    # test_agbm: data/test_agbm/test_agbm

    features_metadata_path = project_root / "data" / "features_metadata.csv"
    train_dir_features = project_root / "data" / "train_feature" / "train_features"
    train_dir_labels = project_root / "data" / "train_agbm" / "train_agbm"
    test_dir_features = project_root / "data" / "test_features" / "test_features"

    # Read metadata
    metadata = pd.read_csv(features_metadata_path)

    # We assume there is a 'split' column in the metadata.
    try:
        train_df = metadata[metadata.split == "train"].copy()
        test_df = metadata[metadata.split == "test"].copy()
        print(f"Found {len(train_df)} training samples and {len(test_df)} test samples.")
    except (AttributeError, KeyError):
        # If there is no 'split' column, let's assume the whole file is for training for this test.
        print("No 'split' column in metadata, using the whole file for training set test.")
        train_df = metadata
        test_df = pd.DataFrame(columns=metadata.columns)  # empty df for test

    # --- Test Training Dataset ---
    print("\n--- Testing Training Dataset ---")
    if not train_df.empty:
        train_dataset = SatImDataset(
            df=train_df,
            dir_features=train_dir_features,
            dir_labels=train_dir_labels,
            augs=True,
            veg_indices=False,
        )
        print(f"Train dataset length: {len(train_dataset)}")
        print("Getting a sample from the training dataset...")
        imgs, mask, tgt = train_dataset[0]

        print("\nSample shapes and types:")
        print(f"  Images: {imgs.shape}, dtype: {imgs.dtype}")
        print(f"  Mask: {mask.shape}, dtype: {mask.dtype}")
        print(f"  Target: {tgt.shape}, dtype: {tgt.dtype}")

        train_loader = get_dataloader(
            df=train_df, dir_features=train_dir_features, dir_labels=train_dir_labels, batch_size=4, num_workers=0
        )
        print("\nGetting a batch from the training dataloader...")
        imgs_batch, mask_batch, tgt_batch = next(iter(train_loader))
        print("Batch shapes:")
        print(f"  Images batch: {imgs_batch.shape}")
        print(f"  Mask batch: {mask_batch.shape}")
        print(f"  Target batch: {tgt_batch.shape}")
    else:
        print("No training data to test.")

    # --- Test Test Dataset ---
    print("\n--- Testing Test Dataset ---")
    if not test_df.empty:
        # For the test set, labels are not available.
        test_dataset = SatImDataset(
            df=test_df,
            dir_features=test_dir_features,
            dir_labels=None,  # No labels for test set
            augs=False,
            veg_indices=False,
        )
        print(f"Test dataset length: {len(test_dataset)}")
        print("Getting a sample from the test dataset...")
        imgs, mask, chip_id = test_dataset[0]

        print("\nSample shapes and types:")
        print(f"  Images: {imgs.shape}, dtype: {imgs.dtype}")
        print(f"  Mask: {mask.shape}, dtype: {mask.dtype}")
        print(f"  Target (chip_id): {chip_id}")

        test_loader = get_dataloader(
            df=test_df, dir_features=test_dir_features, dir_labels=None, batch_size=4, num_workers=0, shuffle=False
        )
        print("\nGetting a batch from the test dataloader...")
        imgs_batch, mask_batch, chip_id_batch = next(iter(test_loader))
        print("Batch shapes:")
        print(f"  Images batch: {imgs_batch.shape}")
        print(f"  Mask batch: {mask_batch.shape}")
        print(f"  Target batch (chip_ids): {len(chip_id_batch)}")
    else:
        print("No test data to test.")

    print("\nTest finished successfully!")