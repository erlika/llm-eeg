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

Author: EEG-BCI Framework
Date: 2024
"""

from .eeg_data import (
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
