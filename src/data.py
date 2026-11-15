from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_client_csv(path, feature_columns, gnss_cols, gt_cols,
                    high_threshold=2.0, medium_threshold=5.0):
    df = pd.read_csv(path)
    required = list(feature_columns) + list(gnss_cols) + list(gt_cols)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    df = df.replace([np.inf, -np.inf], np.nan)
    # Interpolate sensor gaps; discard rows where position/ground truth is unavailable.
    df[list(feature_columns)] = df[list(feature_columns)].interpolate(limit_direction="both")
    df = df.dropna(subset=list(gnss_cols) + list(gt_cols) + list(feature_columns)).reset_index(drop=True)
    gnss = df[list(gnss_cols)].to_numpy(dtype=np.float32)
    gt = df[list(gt_cols)].to_numpy(dtype=np.float32)
    error = np.linalg.norm(gnss - gt, axis=1).astype(np.float32)
    quality = np.where(error <= high_threshold, 0,
              np.where(error <= medium_threshold, 1, 2)).astype(np.int64)
    x = df[list(feature_columns)].to_numpy(dtype=np.float32)
    return x, error, quality

def chronological_split(x, error, quality, train_fraction=0.70, val_fraction=0.15):
    n = len(x)
    a, b = int(n * train_fraction), int(n * (train_fraction + val_fraction))
    if a < 2 or b <= a or n - b < 1:
        raise ValueError("Not enough rows for train/validation/test split.")
    return (x[:a], error[:a], quality[:a]), (x[a:b], error[a:b], quality[a:b]), (x[b:], error[b:], quality[b:])

class SequenceDataset(Dataset):
    def __init__(self, x, error, quality, sequence_length):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.error = torch.tensor(error, dtype=torch.float32)
        self.quality = torch.tensor(quality, dtype=torch.long)
        self.seq = sequence_length
    def __len__(self):
        return max(0, len(self.x) - self.seq + 1)
    def __getitem__(self, idx):
        end = idx + self.seq
        return self.x[idx:end], self.error[end - 1], self.quality[end - 1]

def fit_normalizer(train_arrays):
    joined = np.concatenate(train_arrays, axis=0)
    mean = joined.mean(axis=0)
    std = joined.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def normalize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)
