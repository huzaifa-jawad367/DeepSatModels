# dataloader.py

import argparse
import os
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from skimage import io
import torch.distributed as dist

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
    import argparse
    import os
    from pathlib import Path

    import pandas as pd
    import torch
    import torch.distributed as dist

    # -----------------------
    # 1) PARSE ARGS & DDP SETUP
    # -----------------------
    parser = argparse.ArgumentParser(
        description="Test SatImDataset + (Distributed) DataLoader"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="per-GPU batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader num_workers"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="use torch.distributed data loaders",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="torch.distributed backend",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="rank of this process (set by torchrun)",
    )
    args = parser.parse_args()

    if args.distributed:
        dist.init_process_group(backend=args.backend, init_method="env://")
        torch.cuda.set_device(args.local_rank)
        print(f"[GPU {args.local_rank}] Initialized DDP (world size = {dist.get_world_size()})")

    device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Running on device: {device}")

    # -----------------------
    # 2) LOCATE DATA + READ META
    # -----------------------
    project_root = Path(__file__).resolve().parents[3]
    features_metadata_path = project_root / "data" / "features_metadata.csv"
    train_dir_features = project_root / "data" / "train_feature" / "train_features"
    train_dir_labels = project_root / "data" / "train_agbm" / "train_agbm"
    test_dir_features = project_root / "data" / "test_features" / "test_features"

    metadata = pd.read_csv(features_metadata_path)
    try:
        train_df = metadata[metadata.split == "train"].copy()
        test_df = metadata[metadata.split == "test"].copy()
    except (KeyError, AttributeError):
        train_df = metadata.copy()
        test_df = pd.DataFrame(columns=metadata.columns)

    print(f"Found {len(train_df)} training samples, {len(test_df)} test samples.")

    # -----------------------
    # 3) TEST TRAINING DATASET & DATALOADER
    # -----------------------
    if len(train_df):
        print("\n--- TRAINING SET ---")
        # sample-level test
        ds = SatImDataset(
            df=train_df,
            dir_features=train_dir_features,
            dir_labels=train_dir_labels,
            augs=True,
            veg_indices=False,
        )
        imgs, mask, tgt = ds[0]
        print(" Sample 0 shapes:", imgs.shape, mask.shape, tgt.shape)

        # choose single‐GPU vs DDP dataloader
        if args.distributed:
            loader = get_distributed_dataloader(
                df=train_df,
                dir_features=train_dir_features,
                dir_labels=train_dir_labels,
                augs=True,
                veg_indices=False,
                rank=args.local_rank,
                world_size=torch.cuda.device_count(),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
        else:
            loader = get_dataloader(
                df=train_df,
                dir_features=train_dir_features,
                dir_labels=train_dir_labels,
                augs=True,
                veg_indices=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )

        imgs_b, mask_b, tgt_b = next(iter(loader))
        print(" Batch shapes:", imgs_b.shape, mask_b.shape, tgt_b.shape)
    else:
        print("No training data to test.")

    # -----------------------
    # 4) TEST TEST SET
    # -----------------------
    if len(test_df):
        print("\n--- TEST SET ---")
        ds = SatImDataset(
            df=test_df,
            dir_features=test_dir_features,
            dir_labels=None,
            augs=False,
            veg_indices=False,
        )
        imgs, mask, chip_id = ds[0]
        print(" Sample 0 shapes:", imgs.shape, mask.shape, chip_id)

        if args.distributed:
            loader = get_distributed_dataloader(
                df=test_df,
                dir_features=test_dir_features,
                dir_labels=None,
                augs=False,
                veg_indices=False,
                rank=args.local_rank,
                world_size=torch.cuda.device_count(),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
        else:
            loader = get_dataloader(
                df=test_df,
                dir_features=test_dir_features,
                dir_labels=None,
                augs=False,
                veg_indices=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )

        imgs_b, mask_b, ids_b = next(iter(loader))
        print(" Batch shapes:", imgs_b.shape, mask_b.shape, len(ids_b))
    else:
        print("No test data to test.")

    print("\nAll tests passed!")
