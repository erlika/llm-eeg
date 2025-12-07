"""
Preprocessing Module
====================

This module provides EEG signal preprocessing functionality for the BCI framework.
It includes individual preprocessing steps and a composable pipeline system.

Module Structure:
----------------
- steps/: Individual preprocessing steps (filters, normalization)
- pipeline.py: Composable preprocessing pipeline

Preprocessing Steps:
-------------------
- BandpassFilter: Butterworth bandpass filter
- NotchFilter: IIR notch filter for line noise
- Normalization: Signal normalization methods

Pipeline:
---------
- PreprocessingPipeline: Compose multiple steps
- create_standard_pipeline: Factory for common BCI pipeline
- create_pipeline_from_config: Create pipeline from config dict

Standard BCI Preprocessing:
--------------------------
For motor imagery BCI (BCI Competition IV-2a):

1. **Notch Filter**: Remove 50/60 Hz power line interference
2. **Bandpass Filter**: Isolate 8-30 Hz (mu + beta rhythms)
3. **Normalization**: Standardize signal amplitudes

Usage Examples:
    ```python
    # Method 1: Use standard pipeline factory
    from src.preprocessing import create_standard_pipeline
    
    pipeline = create_standard_pipeline(
        sampling_rate=250,
        notch_freq=50,
        bandpass_low=8,
        bandpass_high=30
    )
    processed = pipeline.process(raw_eeg)
    
    # Method 2: Build custom pipeline
    from src.preprocessing import PreprocessingPipeline
    from src.preprocessing.steps import BandpassFilter, NotchFilter
    
    pipeline = PreprocessingPipeline()
    pipeline.add_step(NotchFilter(), {'notch_freq': 50})
    pipeline.add_step(BandpassFilter(), {'low_freq': 8, 'high_freq': 30})
    pipeline.initialize({'sampling_rate': 250})
    processed = pipeline.process(raw_eeg)
    
    # Method 3: Use individual steps
    from src.preprocessing.steps import BandpassFilter
    
    bandpass = BandpassFilter()
    bandpass.initialize({'sampling_rate': 250, 'low_freq': 8, 'high_freq': 30})
    filtered = bandpass.process(raw_eeg)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

# Import preprocessing steps
from src.preprocessing.steps import (
    BandpassFilter,
    NotchFilter,
    Normalization,
)

# Import pipeline components
from src.preprocessing.pipeline import (
    PreprocessingPipeline,
    create_standard_pipeline,
    create_pipeline_from_config,
)

# Define public API
__all__ = [
    # Steps
    'BandpassFilter',
    'NotchFilter',
    'Normalization',
    
    # Pipeline
    'PreprocessingPipeline',
    'create_standard_pipeline',
    'create_pipeline_from_config',
]
