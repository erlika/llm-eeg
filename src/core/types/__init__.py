"""
Core Types Module
=================

This module exports all core data types for the EEG-BCI framework.

Available Types:
---------------
- EEGData: Container for continuous EEG recordings
- TrialData: Single trial with signal, label, and metadata
- EventMarker: Event/stimulus marker
- DatasetInfo: Dataset-level metadata

Example Usage:
    ```python
    from src.core.types import EEGData, TrialData, EventMarker
    
    # Create EEG data
    eeg_data = EEGData(
        signals=signals,
        sampling_rate=250,
        channel_names=['C3', 'C4', 'Cz']
    )
    
    # Extract trials
    trials = eeg_data.extract_trials(trial_length_sec=4.0)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from src.core.types.eeg_data import (
    EEGData,
    TrialData,
    EventMarker,
    DatasetInfo
)

__all__ = [
    'EEGData',
    'TrialData',
    'EventMarker',
    'DatasetInfo'
]
