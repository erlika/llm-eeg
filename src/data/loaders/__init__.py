"""
Data Loaders Module
===================

This module provides data loading functionality for the EEG-BCI framework.
It includes loaders for various EEG data formats and a factory for creating
appropriate loaders.

Available Loaders:
-----------------
- MATLoader: Loads MATLAB .mat files (BCI Competition IV-2a format)
- BaseDataLoader: Base class for implementing custom loaders

Factory Methods:
---------------
- DataLoaderFactory.create(): Create loader by type name
- DataLoaderFactory.create_for_file(): Auto-detect loader from file extension
- create_loader(): Convenience function for quick loader creation
- load_eeg_file(): Load file with automatic loader selection

BCI Competition IV-2a Support:
-----------------------------
The loaders are optimized for the BCI Competition IV-2a dataset:
- 9 subjects (A01-A09)
- 2 sessions per subject (Training 'T', Evaluation 'E')
- 4 motor imagery classes
- 22 EEG + 3 EOG channels
- 250 Hz sampling rate

Usage Examples:
    ```python
    # Method 1: Use factory with auto-detection
    from src.data.loaders import DataLoaderFactory
    
    loader = DataLoaderFactory.create_for_file('data/A01T.mat')
    eeg_data = loader.load('data/A01T.mat')
    
    # Method 2: Use specific loader directly
    from src.data.loaders import MATLoader
    
    loader = MATLoader()
    loader.initialize({'include_eog': False})
    eeg_data = loader.load('data/A01T.mat')
    
    # Method 3: Quick one-liner
    from src.data.loaders import load_eeg_file
    
    eeg_data = load_eeg_file('data/A01T.mat', include_eog=False)
    ```

Google Colab Usage:
    ```python
    # In Google Colab after mounting Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    from src.data.loaders import load_eeg_file
    
    # Load from Google Drive
    eeg_data = load_eeg_file(
        '/content/drive/MyDrive/BCI_Competition_IV_2a/A01T.mat'
    )
    ```

Author: EEG-BCI Framework
Date: 2024
"""

# Base loader class
from src.data.loaders.base_loader import BaseDataLoader

# MAT file loader
from src.data.loaders.mat_loader import (
    MATLoader,
    create_mat_loader,
    # Constants
    BCI_IV_2A_EEG_CHANNELS,
    BCI_IV_2A_EOG_CHANNELS,
    BCI_IV_2A_ALL_CHANNELS,
    BCI_IV_2A_EVENT_CODES,
    BCI_IV_2A_CLASS_MAPPING,
    BCI_IV_2A_SAMPLING_RATE,
    BCI_IV_2A_TRIALS_PER_SESSION,
)

# Factory and convenience functions
from src.data.loaders.factory import (
    DataLoaderFactory,
    create_loader,
    load_eeg_file,
)

# Define public API
__all__ = [
    # Base classes
    'BaseDataLoader',
    
    # Loaders
    'MATLoader',
    
    # Factory
    'DataLoaderFactory',
    
    # Convenience functions
    'create_loader',
    'create_mat_loader',
    'load_eeg_file',
    
    # Constants
    'BCI_IV_2A_EEG_CHANNELS',
    'BCI_IV_2A_EOG_CHANNELS',
    'BCI_IV_2A_ALL_CHANNELS',
    'BCI_IV_2A_EVENT_CODES',
    'BCI_IV_2A_CLASS_MAPPING',
    'BCI_IV_2A_SAMPLING_RATE',
    'BCI_IV_2A_TRIALS_PER_SESSION',
]
