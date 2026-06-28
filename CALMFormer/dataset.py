from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DATASET_ROOT = Path(__file__).parent / "Datasets"

DATASET_CONFIG = {
    "ORACLE": {
        "path": "ORACLE",
        "default_classes": 16,
        "signal_length": 6000,
    },
    "WiSig_ManyTx": {
        "path": "WiSig_ManyTx",
        "default_classes": 100,
        "signal_length": 256,
    },
    "WiSig": {
        "path": "WiSig",
        "default_classes": 6,
        "signal_length": 256,
    },
    "ADS-B": {
        "path": "ADS-B",
        "default_classes": 10,
        "signal_length": 4800,
    },
}


class SEIDataset(Dataset):
    """Numpy-backed SEI dataset. Returns x with shape [2, L] and integer y."""

    def __init__(self, dataset_name="ORACLE", split="train", n_classes=16, normalize=True):
        if dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        cfg = DATASET_CONFIG[dataset_name]
        data_dir = DATASET_ROOT / cfg["path"]
        x_path = data_dir / f"X_{split}_{n_classes}Class.npy"
        y_path = data_dir / f"Y_{split}_{n_classes}Class.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing dataset files:\n  {x_path}\n  {y_path}\n"
                "Place numpy files under CALMFormer/Datasets/<dataset>/."
            )

        x_raw = np.load(x_path)
        y = np.load(y_path).astype(np.int64)

        x = x_raw.astype(np.float32)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.shape[-1] == 2 and x.shape[1] != 2:
            x = x.transpose(0, 2, 1)
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected X with shape [N, 2, L] or [N, L, 2], got {x.shape}")

        if normalize:
            mean = x.mean(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
            std = x.std(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32) + 1e-8
            x = (x - mean) / std
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite values after normalization: {x_path}")

        self.X = x
        self.Y = y
        self.n_classes = n_classes
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), int(self.Y[idx])


def get_dataloader(dataset_name="ORACLE", split="train", n_classes=16,
                   batch_size=64, shuffle=None, num_workers=2):
    shuffle = (split == "train") if shuffle is None else shuffle
    dataset = SEIDataset(dataset_name, split, n_classes)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset


if __name__ == "__main__":
    for split in ["train", "test"]:
        _, ds = get_dataloader("ORACLE", split, n_classes=16, batch_size=64)
        x, y = ds[0]
        labels, counts = np.unique(ds.Y, return_counts=True)
        print(f"[ORACLE-{split}] samples={len(ds)}, x={tuple(x.shape)}, y={y}")
        print(dict(zip(labels.tolist(), counts.tolist())))
