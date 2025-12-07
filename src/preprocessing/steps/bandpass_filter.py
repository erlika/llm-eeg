"""
Bandpass Filter Preprocessor
============================

This module implements bandpass filtering for EEG signals.

Bandpass filtering is essential for EEG preprocessing because:
1. It removes low-frequency drift (DC offset, movement artifacts)
2. It removes high-frequency noise (muscle activity, line noise harmonics)
3. It isolates the frequency band of interest for motor imagery (typically 8-30 Hz)

Motor Imagery Frequency Bands:
-----------------------------
- Mu rhythm: 8-12 Hz (sensorimotor cortex, suppressed during motor imagery)
- Beta rhythm: 13-30 Hz (motor planning and execution)
- Combined: 8-30 Hz (commonly used for motor imagery BCI)

Filter Design:
-------------
This implementation uses Butterworth IIR filters, which provide:
- Maximally flat passband response
- Good computational efficiency
- Smooth frequency response

The filter is applied forward-backward (filtfilt) to achieve:
- Zero phase distortion
- Doubled filter order effect

Usage Example:
    ```python
    from src.preprocessing.steps import BandpassFilter
    
    # Create filter for motor imagery band (8-30 Hz)
    bandpass = BandpassFilter()
    bandpass.initialize({
        'sampling_rate': 250,
        'low_freq': 8.0,
        'high_freq': 30.0,
        'filter_order': 5
    })
    
    # Apply to EEG data
    filtered_data = bandpass.process(raw_eeg)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, Any, Union, Optional
import numpy as np
from scipy import signal as scipy_signal
import logging

from src.core.interfaces.i_preprocessor import IPreprocessor
from src.core.types.eeg_data import EEGData


# Configure module logger
logger = logging.getLogger(__name__)


class BandpassFilter(IPreprocessor):
    """
    Butterworth bandpass filter for EEG signals.
    
    This preprocessor applies a zero-phase Butterworth bandpass filter
    to EEG signals, preserving the time alignment while removing
    frequencies outside the passband.
    
    Attributes:
        _low_freq (float): Lower cutoff frequency in Hz
        _high_freq (float): Upper cutoff frequency in Hz
        _filter_order (int): Filter order (effective order is doubled due to filtfilt)
        _sampling_rate (float): Signal sampling rate in Hz
        _sos (np.ndarray): Second-order sections filter coefficients
    
    Filter Characteristics:
        - Type: Butterworth (maximally flat passband)
        - Application: Zero-phase (forward-backward)
        - Effective order: 2 × filter_order
        - Edge behavior: Padded to reduce edge effects
    
    Default Configuration (Motor Imagery):
        - low_freq: 8.0 Hz (below mu rhythm)
        - high_freq: 30.0 Hz (above beta rhythm)
        - filter_order: 5 (effective order 10)
    
    Example:
        >>> filter = BandpassFilter()
        >>> filter.initialize({'sampling_rate': 250, 'low_freq': 8, 'high_freq': 30})
        >>> filtered = filter.process(raw_signals)
    """
    
    def __init__(self):
        """Initialize the bandpass filter."""
        # Filter parameters (set during initialization)
        self._low_freq: float = 8.0
        self._high_freq: float = 30.0
        self._filter_order: int = 5
        self._sampling_rate: float = 250.0
        
        # Filter coefficients (computed during initialization)
        self._sos: Optional[np.ndarray] = None
        
        # State tracking
        self._is_initialized: bool = False
        
        logger.debug("BandpassFilter instantiated")
    
    # =========================================================================
    # ABSTRACT PROPERTY IMPLEMENTATIONS
    # =========================================================================
    
    @property
    def name(self) -> str:
        """
        Unique identifier for this preprocessor.
        
        Returns:
            str: 'bandpass_filter'
        """
        return "bandpass_filter"
    
    @property
    def is_trainable(self) -> bool:
        """
        Indicates this preprocessor doesn't require training.
        
        Bandpass filtering is a stateless operation - the same input
        always produces the same output.
        
        Returns:
            bool: False (stateless preprocessor)
        """
        return False
    
    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the bandpass filter with configuration.
        
        Validates parameters and computes filter coefficients.
        
        Args:
            config: Configuration dictionary with keys:
                - 'sampling_rate' (float, required): Signal sampling rate in Hz
                - 'low_freq' (float, optional): Lower cutoff frequency (default: 8.0 Hz)
                - 'high_freq' (float, optional): Upper cutoff frequency (default: 30.0 Hz)
                - 'filter_order' (int, optional): Filter order (default: 5)
        
        Raises:
            ValueError: If parameters are invalid
        
        Example:
            >>> filter.initialize({
            ...     'sampling_rate': 250,
            ...     'low_freq': 8.0,
            ...     'high_freq': 30.0,
            ...     'filter_order': 5
            ... })
        """
        logger.info("Initializing BandpassFilter")
        
        # Extract sampling rate (required)
        if 'sampling_rate' not in config:
            raise ValueError("sampling_rate is required for bandpass filter")
        self._sampling_rate = float(config['sampling_rate'])
        
        # Extract filter parameters
        self._low_freq = float(config.get('low_freq', 8.0))
        self._high_freq = float(config.get('high_freq', 30.0))
        self._filter_order = int(config.get('filter_order', 5))
        
        # Validate parameters
        self._validate_parameters()
        
        # Design filter
        self._design_filter()
        
        self._is_initialized = True
        logger.info(
            f"BandpassFilter initialized: {self._low_freq}-{self._high_freq} Hz, "
            f"order={self._filter_order}, fs={self._sampling_rate} Hz"
        )
    
    def process(
        self,
        data: Union[np.ndarray, EEGData],
        **kwargs
    ) -> Union[np.ndarray, EEGData]:
        """
        Apply bandpass filter to the input data.
        
        Filters the signal to retain only frequencies within the passband.
        Uses zero-phase filtering (filtfilt) to avoid phase distortion.
        
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
        
        Example:
            >>> filtered = filter.process(raw_signals)
            >>> filtered.shape == raw_signals.shape
            True
        """
        if not self._is_initialized:
            raise RuntimeError(
                "BandpassFilter not initialized. Call initialize() first."
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
            Dict containing:
                - 'low_freq': Lower cutoff frequency
                - 'high_freq': Upper cutoff frequency
                - 'filter_order': Filter order
                - 'sampling_rate': Sampling rate
        
        Example:
            >>> params = filter.get_params()
            >>> print(f"Passband: {params['low_freq']}-{params['high_freq']} Hz")
        """
        return {
            'low_freq': self._low_freq,
            'high_freq': self._high_freq,
            'filter_order': self._filter_order,
            'sampling_rate': self._sampling_rate
        }
    
    def set_params(self, **params) -> 'BandpassFilter':
        """
        Set filter parameters.
        
        After setting parameters, the filter is re-designed.
        
        Args:
            **params: Parameters to update:
                - low_freq: Lower cutoff frequency
                - high_freq: Upper cutoff frequency
                - filter_order: Filter order
                - sampling_rate: Sampling rate
        
        Returns:
            Self for method chaining
        
        Example:
            >>> filter.set_params(low_freq=4, high_freq=40)
        """
        if 'low_freq' in params:
            self._low_freq = float(params['low_freq'])
        if 'high_freq' in params:
            self._high_freq = float(params['high_freq'])
        if 'filter_order' in params:
            self._filter_order = int(params['filter_order'])
        if 'sampling_rate' in params:
            self._sampling_rate = float(params['sampling_rate'])
        
        # Re-validate and re-design filter
        if self._is_initialized:
            self._validate_parameters()
            self._design_filter()
            logger.debug("Filter re-designed with new parameters")
        
        return self
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _validate_parameters(self) -> None:
        """
        Validate filter parameters.
        
        Ensures parameters are physically meaningful and won't cause
        numerical issues.
        
        Raises:
            ValueError: If parameters are invalid
        """
        nyquist = self._sampling_rate / 2.0
        
        # Check frequency bounds
        if self._low_freq <= 0:
            raise ValueError(f"low_freq must be positive, got {self._low_freq}")
        
        if self._high_freq <= self._low_freq:
            raise ValueError(
                f"high_freq ({self._high_freq}) must be greater than "
                f"low_freq ({self._low_freq})"
            )
        
        if self._high_freq >= nyquist:
            raise ValueError(
                f"high_freq ({self._high_freq}) must be less than "
                f"Nyquist frequency ({nyquist})"
            )
        
        # Check filter order
        if self._filter_order < 1:
            raise ValueError(f"filter_order must be >= 1, got {self._filter_order}")
        
        if self._filter_order > 10:
            logger.warning(
                f"High filter order ({self._filter_order}) may cause "
                "numerical instability. Consider using order <= 10."
            )
    
    def _design_filter(self) -> None:
        """
        Design the Butterworth bandpass filter.
        
        Uses second-order sections (SOS) representation for numerical
        stability, especially for higher-order filters.
        """
        nyquist = self._sampling_rate / 2.0
        
        # Normalize frequencies to Nyquist
        low_normalized = self._low_freq / nyquist
        high_normalized = self._high_freq / nyquist
        
        # Design filter using SOS for numerical stability
        self._sos = scipy_signal.butter(
            self._filter_order,
            [low_normalized, high_normalized],
            btype='bandpass',
            output='sos'
        )
        
        logger.debug(
            f"Designed Butterworth bandpass filter: "
            f"{self._low_freq}-{self._high_freq} Hz, order={self._filter_order}"
        )
    
    def _process_array(self, data: np.ndarray) -> np.ndarray:
        """
        Apply filter to numpy array.
        
        Handles both 2D (channels, samples) and 3D (trials, channels, samples)
        arrays.
        
        Args:
            data: Input array
        
        Returns:
            Filtered array with same shape
        """
        self.validate_input(data)
        
        if data.ndim == 2:
            # (channels, samples) - filter each channel
            return self._filter_2d(data)
        elif data.ndim == 3:
            # (trials, channels, samples) - filter each trial and channel
            return self._filter_3d(data)
        else:
            raise ValueError(f"Unexpected data dimensions: {data.ndim}")
    
    def _filter_2d(self, data: np.ndarray) -> np.ndarray:
        """
        Filter 2D array (channels, samples).
        
        Args:
            data: Shape (n_channels, n_samples)
        
        Returns:
            Filtered array with same shape
        """
        n_channels, n_samples = data.shape
        filtered = np.zeros_like(data)
        
        # Determine padding length for edge effects
        # Use 3 times the filter order as padding
        padlen = min(3 * self._filter_order * 2, n_samples - 1)
        
        for ch in range(n_channels):
            try:
                filtered[ch] = scipy_signal.sosfiltfilt(
                    self._sos,
                    data[ch],
                    padlen=padlen
                )
            except ValueError as e:
                # Handle short signals
                logger.warning(
                    f"Channel {ch}: Could not apply filter ({e}). "
                    "Using unfiltered data."
                )
                filtered[ch] = data[ch]
        
        return filtered
    
    def _filter_3d(self, data: np.ndarray) -> np.ndarray:
        """
        Filter 3D array (trials, channels, samples).
        
        Args:
            data: Shape (n_trials, n_channels, n_samples)
        
        Returns:
            Filtered array with same shape
        """
        n_trials, n_channels, n_samples = data.shape
        filtered = np.zeros_like(data)
        
        for trial in range(n_trials):
            filtered[trial] = self._filter_2d(data[trial])
        
        return filtered
    
    def _process_eegdata(self, eeg_data: EEGData) -> EEGData:
        """
        Apply filter to EEGData object.
        
        Creates a new EEGData with filtered signals while preserving
        all metadata.
        
        Args:
            eeg_data: EEGData object to filter
        
        Returns:
            New EEGData with filtered signals
        """
        # Filter signals
        filtered_signals = self._filter_2d(eeg_data.signals)
        
        # Create new EEGData with filtered signals
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
                'bandpass_filtered': True,
                'bandpass_params': self.get_params()
            }
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_frequency_response(
        self,
        n_points: int = 512
    ) -> tuple:
        """
        Compute the filter's frequency response.
        
        Useful for visualizing the filter characteristics.
        
        Args:
            n_points: Number of frequency points to compute
        
        Returns:
            Tuple of (frequencies, magnitude_db):
                - frequencies: Array of frequencies in Hz
                - magnitude_db: Magnitude response in dB
        
        Example:
            >>> freqs, mag = filter.get_frequency_response()
            >>> plt.plot(freqs, mag)
            >>> plt.xlabel('Frequency (Hz)')
            >>> plt.ylabel('Magnitude (dB)')
        """
        if self._sos is None:
            raise RuntimeError("Filter not initialized")
        
        # Compute frequency response
        w, h = scipy_signal.sosfreqz(self._sos, worN=n_points)
        
        # Convert to Hz
        frequencies = w * self._sampling_rate / (2 * np.pi)
        
        # Convert to dB (avoid log of zero)
        magnitude = np.abs(h)
        magnitude[magnitude < 1e-10] = 1e-10
        magnitude_db = 20 * np.log10(magnitude)
        
        return frequencies, magnitude_db
    
    def get_passband(self) -> tuple:
        """
        Get the filter passband frequencies.
        
        Returns:
            Tuple of (low_freq, high_freq) in Hz
        """
        return (self._low_freq, self._high_freq)
    
    def __repr__(self) -> str:
        """String representation."""
        if self._is_initialized:
            return (
                f"BandpassFilter({self._low_freq}-{self._high_freq} Hz, "
                f"order={self._filter_order}, fs={self._sampling_rate})"
            )
        return "BandpassFilter(not initialized)"
