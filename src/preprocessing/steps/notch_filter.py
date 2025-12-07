"""
Notch Filter Preprocessor
=========================

This module implements notch (band-stop) filtering for removing
power line interference from EEG signals.

Power Line Interference:
-----------------------
EEG recordings are commonly contaminated by electromagnetic interference
from power lines:
- 50 Hz: Europe, Asia, Africa, Australia (most countries)
- 60 Hz: North America, parts of South America

This interference appears as a sharp peak at the fundamental frequency
and may have harmonics (100/120 Hz, 150/180 Hz, etc.).

Notch Filter Design:
-------------------
This implementation uses an IIR notch filter (second-order sections)
with configurable:
- Center frequency (50 or 60 Hz)
- Quality factor (Q) - controls the width of the notch
- Option to remove harmonics as well

Quality Factor:
- Higher Q = narrower notch = less signal distortion but may not remove all noise
- Lower Q = wider notch = better noise removal but more signal distortion
- Typical Q for EEG: 25-35

Usage Example:
    ```python
    from src.preprocessing.steps import NotchFilter
    
    # Create 50 Hz notch filter (European power line)
    notch = NotchFilter()
    notch.initialize({
        'sampling_rate': 250,
        'notch_freq': 50.0,
        'quality_factor': 30,
        'remove_harmonics': False  # Set True to also remove 100 Hz, 150 Hz
    })
    
    # Apply to EEG data
    filtered_data = notch.process(raw_eeg)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, Any, Union, Optional, List
import numpy as np
from scipy import signal as scipy_signal
import logging

from src.core.interfaces.i_preprocessor import IPreprocessor
from src.core.types.eeg_data import EEGData


# Configure module logger
logger = logging.getLogger(__name__)


class NotchFilter(IPreprocessor):
    """
    IIR notch filter for removing power line interference.
    
    This preprocessor applies a notch (band-stop) filter to remove
    specific frequency components, primarily power line noise.
    
    Attributes:
        _notch_freq (float): Center frequency of the notch in Hz
        _quality_factor (float): Q factor controlling notch bandwidth
        _remove_harmonics (bool): Whether to also filter harmonics
        _max_harmonic (int): Maximum harmonic to remove
        _sampling_rate (float): Signal sampling rate in Hz
        _sos_list (List): List of SOS coefficients for each notch
    
    Filter Characteristics:
        - Type: IIR notch (second-order sections)
        - Application: Zero-phase (forward-backward)
        - Bandwidth: ~notch_freq/Q (e.g., 50/30 ≈ 1.67 Hz)
    
    Default Configuration:
        - notch_freq: 50.0 Hz (European power line)
        - quality_factor: 30 (bandwidth ~1.67 Hz)
        - remove_harmonics: False
    
    Example:
        >>> notch = NotchFilter()
        >>> notch.initialize({'sampling_rate': 250, 'notch_freq': 50})
        >>> filtered = notch.process(raw_signals)
    """
    
    def __init__(self):
        """Initialize the notch filter."""
        # Filter parameters
        self._notch_freq: float = 50.0
        self._quality_factor: float = 30.0
        self._remove_harmonics: bool = False
        self._max_harmonic: int = 3  # Up to 3rd harmonic (150/180 Hz)
        self._sampling_rate: float = 250.0
        
        # Filter coefficients (list for multiple notches)
        self._sos_list: List[np.ndarray] = []
        
        # State tracking
        self._is_initialized: bool = False
        
        logger.debug("NotchFilter instantiated")
    
    # =========================================================================
    # ABSTRACT PROPERTY IMPLEMENTATIONS
    # =========================================================================
    
    @property
    def name(self) -> str:
        """
        Unique identifier for this preprocessor.
        
        Returns:
            str: 'notch_filter'
        """
        return "notch_filter"
    
    @property
    def is_trainable(self) -> bool:
        """
        Indicates this preprocessor doesn't require training.
        
        Returns:
            bool: False (stateless preprocessor)
        """
        return False
    
    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the notch filter with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - 'sampling_rate' (float, required): Signal sampling rate in Hz
                - 'notch_freq' (float, optional): Notch frequency (default: 50.0 Hz)
                - 'quality_factor' (float, optional): Q factor (default: 30)
                - 'remove_harmonics' (bool, optional): Remove harmonics (default: False)
                - 'max_harmonic' (int, optional): Max harmonic order (default: 3)
        
        Raises:
            ValueError: If parameters are invalid
        
        Example:
            >>> notch.initialize({
            ...     'sampling_rate': 250,
            ...     'notch_freq': 60.0,  # US power line
            ...     'quality_factor': 30,
            ...     'remove_harmonics': True
            ... })
        """
        logger.info("Initializing NotchFilter")
        
        # Extract sampling rate (required)
        if 'sampling_rate' not in config:
            raise ValueError("sampling_rate is required for notch filter")
        self._sampling_rate = float(config['sampling_rate'])
        
        # Extract filter parameters
        self._notch_freq = float(config.get('notch_freq', 50.0))
        self._quality_factor = float(config.get('quality_factor', 30.0))
        self._remove_harmonics = config.get('remove_harmonics', False)
        self._max_harmonic = int(config.get('max_harmonic', 3))
        
        # Validate parameters
        self._validate_parameters()
        
        # Design filter(s)
        self._design_filters()
        
        self._is_initialized = True
        
        harmonics_info = f", harmonics up to {self._max_harmonic}x" if self._remove_harmonics else ""
        logger.info(
            f"NotchFilter initialized: {self._notch_freq} Hz, "
            f"Q={self._quality_factor}{harmonics_info}"
        )
    
    def process(
        self,
        data: Union[np.ndarray, EEGData],
        **kwargs
    ) -> Union[np.ndarray, EEGData]:
        """
        Apply notch filter to the input data.
        
        Removes power line interference at the configured frequency
        (and optionally harmonics).
        
        Args:
            data: Input data to filter
                - numpy array: Shape (channels, samples) or (trials, channels, samples)
                - EEGData: EEG data container
            **kwargs: Additional options (currently unused)
        
        Returns:
            Filtered data in the same format as input
        
        Raises:
            RuntimeError: If filter not initialized
            ValueError: If data format is invalid
        """
        if not self._is_initialized:
            raise RuntimeError(
                "NotchFilter not initialized. Call initialize() first."
            )
        
        # Handle EEGData input
        if isinstance(data, EEGData):
            return self._process_eegdata(data)
        
        # Handle numpy array input
        return self._process_array(data)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current filter parameters.
        
        Returns:
            Dict containing all filter parameters
        """
        return {
            'notch_freq': self._notch_freq,
            'quality_factor': self._quality_factor,
            'remove_harmonics': self._remove_harmonics,
            'max_harmonic': self._max_harmonic,
            'sampling_rate': self._sampling_rate
        }
    
    def set_params(self, **params) -> 'NotchFilter':
        """
        Set filter parameters.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        """
        if 'notch_freq' in params:
            self._notch_freq = float(params['notch_freq'])
        if 'quality_factor' in params:
            self._quality_factor = float(params['quality_factor'])
        if 'remove_harmonics' in params:
            self._remove_harmonics = bool(params['remove_harmonics'])
        if 'max_harmonic' in params:
            self._max_harmonic = int(params['max_harmonic'])
        if 'sampling_rate' in params:
            self._sampling_rate = float(params['sampling_rate'])
        
        # Re-validate and re-design filter
        if self._is_initialized:
            self._validate_parameters()
            self._design_filters()
            logger.debug("Notch filter(s) re-designed with new parameters")
        
        return self
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Raises:
            ValueError: If parameters are invalid
        """
        nyquist = self._sampling_rate / 2.0
        
        # Check notch frequency
        if self._notch_freq <= 0:
            raise ValueError(f"notch_freq must be positive, got {self._notch_freq}")
        
        if self._notch_freq >= nyquist:
            raise ValueError(
                f"notch_freq ({self._notch_freq}) must be less than "
                f"Nyquist frequency ({nyquist})"
            )
        
        # Check quality factor
        if self._quality_factor <= 0:
            raise ValueError(
                f"quality_factor must be positive, got {self._quality_factor}"
            )
        
        # Check harmonics
        if self._remove_harmonics:
            max_freq = self._notch_freq * self._max_harmonic
            if max_freq >= nyquist:
                logger.warning(
                    f"Maximum harmonic frequency ({max_freq} Hz) exceeds "
                    f"Nyquist ({nyquist} Hz). Will only filter valid harmonics."
                )
    
    def _design_filters(self) -> None:
        """
        Design notch filter(s).
        
        Creates one filter for the fundamental frequency and additional
        filters for harmonics if requested.
        """
        nyquist = self._sampling_rate / 2.0
        self._sos_list = []
        
        # Frequencies to notch
        frequencies = [self._notch_freq]
        
        if self._remove_harmonics:
            for harmonic in range(2, self._max_harmonic + 1):
                freq = self._notch_freq * harmonic
                if freq < nyquist * 0.95:  # Leave some margin
                    frequencies.append(freq)
        
        # Design a notch filter for each frequency
        for freq in frequencies:
            # Normalize frequency
            w0 = freq / nyquist
            
            # Design notch filter
            b, a = scipy_signal.iirnotch(w0, self._quality_factor)
            
            # Convert to SOS for numerical stability
            sos = scipy_signal.tf2sos(b, a)
            self._sos_list.append(sos)
            
            logger.debug(f"Designed notch filter at {freq} Hz")
    
    def _process_array(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter(s) to numpy array.
        
        Args:
            data: Input array
        
        Returns:
            Filtered array
        """
        self.validate_input(data)
        
        if data.ndim == 2:
            return self._filter_2d(data)
        elif data.ndim == 3:
            return self._filter_3d(data)
        else:
            raise ValueError(f"Unexpected data dimensions: {data.ndim}")
    
    def _filter_2d(self, data: np.ndarray) -> np.ndarray:
        """
        Filter 2D array (channels, samples).
        
        Args:
            data: Shape (n_channels, n_samples)
        
        Returns:
            Filtered array
        """
        n_channels, n_samples = data.shape
        filtered = data.copy()
        
        # Apply each notch filter in sequence
        for sos in self._sos_list:
            for ch in range(n_channels):
                try:
                    filtered[ch] = scipy_signal.sosfiltfilt(sos, filtered[ch])
                except ValueError as e:
                    logger.warning(
                        f"Channel {ch}: Could not apply notch filter ({e})"
                    )
        
        return filtered
    
    def _filter_3d(self, data: np.ndarray) -> np.ndarray:
        """
        Filter 3D array (trials, channels, samples).
        
        Args:
            data: Shape (n_trials, n_channels, n_samples)
        
        Returns:
            Filtered array
        """
        n_trials = data.shape[0]
        filtered = np.zeros_like(data)
        
        for trial in range(n_trials):
            filtered[trial] = self._filter_2d(data[trial])
        
        return filtered
    
    def _process_eegdata(self, eeg_data: EEGData) -> EEGData:
        """
        Apply notch filter to EEGData object.
        
        Args:
            eeg_data: EEGData object
        
        Returns:
            New EEGData with filtered signals
        """
        filtered_signals = self._filter_2d(eeg_data.signals)
        
        return EEGData(
            signals=filtered_signals,
            sampling_rate=eeg_data.sampling_rate,
            channel_names=eeg_data.channel_names.copy(),
            events=eeg_data.events.copy(),
            subject_id=eeg_data.subject_id,
            session_id=eeg_data.session_id,
            recording_date=eeg_data.recording_date,
            source_file=eeg_data.source_file,
            metadata={
                **eeg_data.metadata,
                'notch_filtered': True,
                'notch_params': self.get_params()
            }
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_notch_frequencies(self) -> List[float]:
        """
        Get all frequencies being notched.
        
        Returns:
            List of notch frequencies in Hz
        """
        nyquist = self._sampling_rate / 2.0
        frequencies = [self._notch_freq]
        
        if self._remove_harmonics:
            for harmonic in range(2, self._max_harmonic + 1):
                freq = self._notch_freq * harmonic
                if freq < nyquist * 0.95:
                    frequencies.append(freq)
        
        return frequencies
    
    def get_bandwidth(self) -> float:
        """
        Get the approximate -3dB bandwidth of each notch.
        
        Returns:
            Bandwidth in Hz
        """
        return self._notch_freq / self._quality_factor
    
    def __repr__(self) -> str:
        """String representation."""
        if self._is_initialized:
            harmonics = f", +harmonics" if self._remove_harmonics else ""
            return (
                f"NotchFilter({self._notch_freq} Hz, "
                f"Q={self._quality_factor}{harmonics})"
            )
        return "NotchFilter(not initialized)"
