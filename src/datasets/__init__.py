"""
Datasets Module
===============

This module provides PyTorch-compatible Dataset classes for EEG data,
enabling seamless integration with PyTorch DataLoader for training
deep learning models.

Dataset Classes:
---------------
- EEGDataset: Base dataset for EEG trials
- BCICIV2aDataset: Specialized for BCI Competition IV-2a

Utility Functions:
-----------------
- train_val_test_split: Split dataset into train/val/test
- create_cv_folds: Create cross-validation folds

Features:
---------
- PyTorch DataLoader compatibility
- Trial extraction from continuous EEG
- Optional preprocessing transforms
- Train/val/test splitting
- Cross-validation support
- Subject-wise data loading

PyTorch Integration:
    ```python
    from torch.utils.data import DataLoader
    from src.datasets import EEGDataset
    
    dataset = EEGDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for batch_x, batch_y in loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
    ```

BCI Competition IV-2a Usage:
    ```python
    from src.datasets import BCICIV2aDataset
    
    # Load single subject
    dataset = BCICIV2aDataset.from_subject(
        subject_id=1,
        session='T',
        data_dir='/content/drive/MyDrive/BCI_IV_2a'
    )
    
    # Load multiple subjects
    dataset = BCICIV2aDataset.from_subjects(
        subjects=[1, 2, 3],
        session='T',
        data_dir=data_dir
    )
    
    # Create train/val split
    from src.datasets import train_val_test_split
    train_ds, val_ds = train_val_test_split(dataset, val_ratio=0.2)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

# Import dataset classes
from src.datasets.eeg_dataset import (
    EEGDataset,
    BCICIV2aDataset,
    train_val_test_split,
    create_cv_folds,
)

# Define public API
__all__ = [
    # Dataset classes
    'EEGDataset',
    'BCICIV2aDataset',
    
    # Utility functions
    'train_val_test_split',
    'create_cv_folds',
]
