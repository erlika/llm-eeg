"""
Preprocessing Steps Module
==========================

This module provides individual preprocessing steps for EEG signals.
Each step implements the IPreprocessor interface and can be used
standalone or composed into pipelines.

Available Steps:
---------------
- BandpassFilter: Butterworth bandpass filter (e.g., 8-30 Hz for MI)
- NotchFilter: IIR notch filter for line noise removal (50/60 Hz)
- Normalization: Signal normalization (z-score, minmax, robust)

Motor Imagery Standard Processing:
---------------------------------
For BCI Competition IV-2a (motor imagery), typical settings:

1. Notch filter: 50 Hz (Europe) or 60 Hz (US)
2. Bandpass filter: 8-30 Hz (mu + beta rhythms)
3. Normalization: z-score, channel-wise

Usage Examples:
    ```python
    # Import steps
    from src.preprocessing.steps import BandpassFilter, NotchFilter, Normalization
    
    # Create and use bandpass filter
    bandpass = BandpassFilter()
    bandpass.initialize({
        'sampling_rate': 250,
        'low_freq': 8.0,
        'high_freq': 30.0
    })
    filtered = bandpass.process(raw_eeg)
    
    # Create and use notch filter
    notch = NotchFilter()
    notch.initialize({
        'sampling_rate': 250,
        'notch_freq': 50.0,
        'quality_factor': 30
    })
    clean = notch.process(filtered)
    
    # Create and use normalization
    norm = Normalization()
    norm.initialize({
        'method': 'zscore',
        'axis': 'channel'
    })
    normalized = norm.process(clean)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

# Import preprocessing steps
from src.preprocessing.steps.bandpass_filter import BandpassFilter
from src.preprocessing.steps.notch_filter import NotchFilter
from src.preprocessing.steps.normalization import Normalization

# Define public API
__all__ = [
    'BandpassFilter',
    'NotchFilter',
    'Normalization',
]
