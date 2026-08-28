import pandas as pd
import torch
from torch.utils.data import DataLoader
from . import sample_dataset as ds


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform = None,
    val_transform = None,
):
    train_dataset = ds.SegDataset(train_df, train_transform)
    val_dataset = ds.SegDataset(val_df, val_transform)
    test_dataset = ds.SegDataset(test_df, val_transform)

    return train_dataset, val_dataset, test_dataset

def build_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size = 8,
    num_workers = 4,
    pin_memory=True
):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    valid_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, valid_loader, test_loader