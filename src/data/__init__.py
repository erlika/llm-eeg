"""
Data Module
===========

This module provides comprehensive data loading and validation functionality
for the EEG-BCI framework.

Sub-modules:
-----------
- loaders: Data loading from various file formats
- validators: Data validation and quality assessment

Loaders:
--------
- MATLoader: Load MATLAB .mat files (BCI Competition IV-2a)
- DataLoaderFactory: Factory for creating loaders
- load_eeg_file: Convenience function for loading files

Validators:
----------
- DataValidator: Validate data structure and integrity
- QualityChecker: Assess signal quality metrics

Usage Examples:
    ```python
    # Load EEG data from file
    from src.data import load_eeg_file
    
    eeg_data = load_eeg_file('data/A01T.mat', include_eog=False)
    print(f"Loaded: {eeg_data.shape}")
    
    # Extract trials
    X, y = eeg_data.get_trials_array(trial_length_sec=4.0)
    
    # Validate data
    from src.data import DataValidator
    
    validator = DataValidator()
    result = validator.validate(eeg_data)
    print(f"Valid: {result.is_valid}")
    
    # Check signal quality
    from src.data import QualityChecker
    
    checker = QualityChecker()
    checker.initialize({'sampling_rate': 250})
    report = checker.assess_quality(eeg_data)
    print(f"Quality: {report['overall_score']:.2%}")
    ```

Google Colab Usage:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    
    from src.data import load_eeg_file
    
    eeg_data = load_eeg_file(
        '/content/drive/MyDrive/BCI_Competition_IV_2a/A01T.mat'
    )
    ```

Author: EEG-BCI Framework
Date: 2024
"""

# Import from loaders module
from src.data.loaders import (
    # Base and loaders
    BaseDataLoader,
    MATLoader,
    
    # Factory
    DataLoaderFactory,
    
    # Convenience functions
    create_loader,
    create_mat_loader,
    load_eeg_file,
    
    # Constants
    BCI_IV_2A_EEG_CHANNELS,
    BCI_IV_2A_EOG_CHANNELS,
    BCI_IV_2A_ALL_CHANNELS,
    BCI_IV_2A_EVENT_CODES,
    BCI_IV_2A_CLASS_MAPPING,
    BCI_IV_2A_SAMPLING_RATE,
    BCI_IV_2A_TRIALS_PER_SESSION,
)

# Import from validators module
from src.data.validators import (
    DataValidator,
    ValidationResult,
    ValidationError,
    QualityChecker,
)

# Define public API
__all__ = [
    # Loaders
    'BaseDataLoader',
    'MATLoader',
    'DataLoaderFactory',
    'create_loader',
    'create_mat_loader',
    'load_eeg_file',
    
    # Validators
    'DataValidator',
    'ValidationResult',
    'ValidationError',
    'QualityChecker',
    
    # Constants
    'BCI_IV_2A_EEG_CHANNELS',
    'BCI_IV_2A_EOG_CHANNELS',
    'BCI_IV_2A_ALL_CHANNELS',
    'BCI_IV_2A_EVENT_CODES',
    'BCI_IV_2A_CLASS_MAPPING',
    'BCI_IV_2A_SAMPLING_RATE',
    'BCI_IV_2A_TRIALS_PER_SESSION',
]
