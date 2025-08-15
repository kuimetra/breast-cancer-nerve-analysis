from transforms import ComposeTransformations, Random90Rotation, RandomFlip
from data_io import load_data

from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, List, Any
import numpy as np
import torch


class IMCDataset(Dataset):
    """
    A PyTorch Dataset for handling peripherin marker images and binary nerve masks.

    Attributes:
    - protein_markers: Dictionary containing protein marker data with case IDs as keys.
    - bin_nerve_mask: Dictionary containing binary nerve mask data with case IDs as keys.
    - transform: Optional list of transformations to be applied on a sample.
    """

    def __init__(self, peripherin_marker, bin_nerve_mask, transform=None):
        self.peripherin_marker = peripherin_marker
        self.bin_nerve_mask = bin_nerve_mask
        self.transform = transform

    def __len__(self):
        return len(self.peripherin_marker)

    def __getitem__(self, idx):
        case_id = list(self.peripherin_marker.keys())[idx]
        peripherin_marker = torch.tensor(self.peripherin_marker[case_id], dtype=torch.float32).unsqueeze(0)
        bin_nerve_mask = torch.tensor(self.bin_nerve_mask[case_id], dtype=torch.float32).unsqueeze(0)

        if self.transform:
            peripherin_marker, bin_nerve_mask = self.transform(peripherin_marker, bin_nerve_mask)

        return case_id, peripherin_marker, bin_nerve_mask


def create_dataloader(data: dict[str, dict[str, np.ndarray]],
                      transform,
                      batch_size: int,
                      shuffle: bool) -> DataLoader:
    """
    Creates a DataLoader for a specific dataset.

    Parameters:
    - data: Dictionary containing peripherin marker and nerve masks.
    - transform: Transformation to apply to the dataset.
    - batch_size: Number of samples per batch.
    - shuffle: Whether to shuffle the data.

    Returns:
    - DataLoader for the given data.
    """
    dataset = IMCDataset(data["peripherin_marker"], data["bin_nerve_mask"], transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataloaders(
        base_path: str = "data/datasets",
        batch_size: int = 16,
        mode: str = "train",
        data_augmentation: bool = False,
        transform_list: Optional[List[Any]] = None) -> Dict[str, DataLoader]:
    """
    Constructs DataLoaders for the IMC dataset based on the specified mode.

    Parameters:
    - base_path: Path to the dataset directory.
    - batch_size: Number of samples per batch.
    - mode: Determines which DataLoaders to create ('train', 'val', 'test', 'all').
    - data_augmentation: Whether to apply augmentation transformations.
    - transform_list: Additional transformations to apply.

    Returns:
    - Dictionary containing DataLoaders for 'train', 'val', 'test' or 'all'.
    """
    transform_list = transform_list or []
    augmentations = [Random90Rotation(), RandomFlip()] if data_augmentation else []

    train_transform = ComposeTransformations(augmentations + transform_list)
    val_test_transform = ComposeTransformations(transform_list)

    data = load_data(base_path, mode)

    return {
        subset: create_dataloader(data[subset],
                                  train_transform if subset == "train" else val_test_transform,
                                  batch_size,
                                  shuffle=(subset == "train"))
        for subset in data
    }
