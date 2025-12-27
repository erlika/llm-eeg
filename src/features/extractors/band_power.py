"""
Band Power Feature Extractor
============================

This module implements frequency band power feature extraction for EEG signals.

Band power features measure the signal energy in specific frequency bands,
which is fundamental for EEG analysis as different cognitive states are
associated with different frequency patterns.

Standard EEG Frequency Bands:
----------------------------
| Band   | Frequency (Hz) | Associated States                     |
|--------|----------------|---------------------------------------|
| Delta  | 0.5 - 4        | Deep sleep                            |
| Theta  | 4 - 8          | Drowsiness, meditation                |
| Alpha  | 8 - 13         | Relaxed wakefulness, eyes closed      |
| Mu     | 8 - 12         | Motor cortex idle (motor imagery!)    |
| Beta   | 13 - 30        | Active thinking, motor activity       |
| Gamma  | 30 - 100       | High-level cognition                  |

Motor Imagery Specific:
----------------------
For motor imagery BCI, the most relevant bands are:
- Mu (8-12 Hz): Event-Related Desynchronization (ERD) during movement
- Beta (13-30 Hz): Post-movement synchronization, motor planning

Methods:
-------
1. **Welch's Method**: Averaged periodogram, most robust
2. **FFT**: Direct FFT power spectrum
3. **Multitaper**: Optimal for short segments

Usage Example:
    ```python
    from src.features.extractors.band_power import BandPowerExtractor
    
    # Create extractor with custom bands
    bp = BandPowerExtractor(
        bands={
            'mu': (8, 12),
            'beta': (13, 30)
        },
        method='welch'
    )
    bp.initialize({'sampling_rate': 250})
    
    # Extract features
    features = bp.extract(X)  # Shape: (n_trials, n_channels * n_bands)
    ```

References:
----------
1. Pfurtscheller & Lopes da Silva, "Event-related EEG/MEG synchronization 
   and desynchronization: basic principles", Clin Neurophysiol, 1999
2. Welch, "The use of fast Fourier transform for the estimation of power 
   spectra", IEEE Trans Audio Electroacoust, 1967

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from scipy import signal
from scipy.integrate import simpson
import logging

from src.features.base import BaseFeatureExtractor
from src.core.registry import registered

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT FREQUENCY BANDS
# =============================================================================

# Standard EEG bands
DEFAULT_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100),
}

# Motor imagery specific bands (recommended for BCI)
MOTOR_IMAGERY_BANDS = {
    'mu': (8, 12),      # Sensorimotor rhythm
    'beta_low': (12, 20),
    'beta_high': (20, 30),
}

# Simple bands for quick analysis
SIMPLE_BANDS = {
    'mu': (8, 12),
    'beta': (12, 30),
}


@registered('feature_extractor', 'band_power')
class BandPowerExtractor(BaseFeatureExtractor):
    """
    Frequency band power feature extractor.
    
    Extracts power in specified frequency bands for each channel.
    This is a stateless extractor (no fitting required).
    
    Attributes:
        bands (Dict[str, Tuple[float, float]]): Frequency bands to extract.
            Keys are band names, values are (low_freq, high_freq) tuples.
        method (str): PSD estimation method ('welch', 'fft', 'multitaper').
        relative (bool): If True, compute relative power (band / total).
        log (bool): If True, apply log transform to power values.
        average_channels (bool): If True, average across channels.
        selected_channels (List[int]): If set, only use these channels.
    
    Output Format:
        Default: (n_trials, n_channels * n_bands)
        With average_channels=True: (n_trials, n_bands)
        
        Feature order: [ch0_band0, ch0_band1, ..., ch1_band0, ch1_band1, ...]
    
    Example:
        >>> bp = BandPowerExtractor(bands={'mu': (8, 12), 'beta': (12, 30)})
        >>> bp.initialize({'sampling_rate': 250})
        >>> features = bp.extract(X)  # (n_trials, 22 * 2) = (n_trials, 44)
    """
    
    def __init__(self,
                 bands: Optional[Dict[str, Tuple[float, float]]] = None,
                 method: str = 'welch',
                 relative: bool = False,
                 log: bool = True,
                 average_channels: bool = False,
                 selected_channels: Optional[List[int]] = None,
                 nperseg: Optional[int] = None,
                 noverlap: Optional[int] = None):
        """
        Initialize band power extractor.
        
        Args:
            bands: Frequency bands as {name: (low, high)} dict.
                Default: mu (8-12 Hz) and beta (12-30 Hz).
            method: PSD method - 'welch', 'fft', or 'multitaper'.
            relative: Compute relative power (band power / total power).
            log: Apply log10 transform to power values.
            average_channels: Average power across all channels.
            selected_channels: Only use these channel indices.
            nperseg: Samples per segment for Welch (default: fs).
            noverlap: Overlap samples for Welch (default: nperseg/2).
        """
        super().__init__()
        
        # Band configuration
        self._bands = bands or SIMPLE_BANDS.copy()
        self._band_names = list(self._bands.keys())
        
        # Method parameters
        self._method = method.lower()
        self._relative = relative
        self._log = log
        self._average_channels = average_channels
        self._selected_channels = selected_channels
        
        # Welch parameters
        self._nperseg = nperseg
        self._noverlap = noverlap
        
        # Validate method
        valid_methods = ['welch', 'fft', 'multitaper']
        if self._method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
    
    # =========================================================================
    # PROPERTY IMPLEMENTATIONS
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Extractor name."""
        return "band_power"
    
    @property
    def is_trainable(self) -> bool:
        """Band power extraction is stateless."""
        return False
    
    @property
    def bands(self) -> Dict[str, Tuple[float, float]]:
        """Get frequency bands."""
        return self._bands.copy()
    
    @bands.setter
    def bands(self, value: Dict[str, Tuple[float, float]]) -> None:
        """Set frequency bands."""
        self._bands = value
        self._band_names = list(value.keys())
        self._update_feature_count()
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _initialize_implementation(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        # Override bands if provided
        if 'bands' in config:
            self._bands = config['bands']
            self._band_names = list(self._bands.keys())
        
        # Override other params
        if 'method' in config:
            self._method = config['method'].lower()
        if 'relative' in config:
            self._relative = config['relative']
        if 'log' in config:
            self._log = config['log']
        if 'average_channels' in config:
            self._average_channels = config['average_channels']
        if 'selected_channels' in config:
            self._selected_channels = config['selected_channels']
        if 'nperseg' in config:
            self._nperseg = config['nperseg']
        if 'noverlap' in config:
            self._noverlap = config['noverlap']
        
        # Validate bands against Nyquist
        nyquist = self._sampling_rate / 2
        for band_name, (low, high) in self._bands.items():
            if high > nyquist:
                logger.warning(
                    f"Band '{band_name}' high freq ({high} Hz) exceeds Nyquist "
                    f"({nyquist} Hz). Clamping to Nyquist."
                )
                self._bands[band_name] = (low, min(high, nyquist - 1))
        
        self._update_feature_count()
    
    def _update_feature_count(self) -> None:
        """Update feature count based on configuration."""
        n_bands = len(self._bands)
        
        if self._average_channels:
            self._n_features_value = n_bands
        elif self._selected_channels is not None:
            self._n_features_value = len(self._selected_channels) * n_bands
        elif self._n_channels is not None:
            self._n_features_value = self._n_channels * n_bands
        else:
            self._n_features_value = 0  # Unknown until data is seen
    
    # =========================================================================
    # FIT IMPLEMENTATION (Override for non-trainable)
    # =========================================================================
    
    def _fit_implementation(self,
                           X: np.ndarray,
                           y: Optional[np.ndarray],
                           **kwargs) -> None:
        """
        'Fit' for band power just stores data dimensions.
        
        No actual training is needed since this is stateless.
        """
        # Just update dimensions
        self._update_feature_count()
    
    # =========================================================================
    # EXTRACTION IMPLEMENTATION
    # =========================================================================
    
    def _extract_implementation(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Extract band power features.
        
        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
        
        Returns:
            Features, shape (n_trials, n_features)
        """
        n_trials, n_channels, n_samples = X.shape
        
        # Determine channels to use
        if self._selected_channels is not None:
            channel_indices = self._selected_channels
        else:
            channel_indices = list(range(n_channels))
        
        n_selected_channels = len(channel_indices)
        n_bands = len(self._bands)
        
        # Output shape
        if self._average_channels:
            n_features = n_bands
        else:
            n_features = n_selected_channels * n_bands
        
        features = np.zeros((n_trials, n_features))
        
        for trial_idx in range(n_trials):
            trial_features = self._extract_trial(
                X[trial_idx], channel_indices
            )
            features[trial_idx] = trial_features
        
        # Update feature count if not set
        if self._n_features_value == 0:
            self._n_features_value = n_features
            self._feature_names_list = self._generate_feature_names()
        
        return features
    
    def _extract_trial(self,
                       trial: np.ndarray,
                       channel_indices: List[int]) -> np.ndarray:
        """
        Extract band power from a single trial.
        
        Args:
            trial: Trial data, shape (n_channels, n_samples)
            channel_indices: Channels to extract from
        
        Returns:
            Features for this trial
        """
        n_channels = len(channel_indices)
        n_bands = len(self._bands)
        
        # Compute PSD for each channel
        channel_powers = []
        
        for ch_idx in channel_indices:
            channel_data = trial[ch_idx]
            
            # Compute PSD
            freqs, psd = self._compute_psd(channel_data)
            
            # Extract band powers
            band_powers = []
            total_power = simpson(psd, x=freqs) if self._relative else 1.0
            
            for band_name, (low, high) in self._bands.items():
                # Find frequency indices
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                
                # Integrate power in band
                if np.any(idx_band):
                    band_power = simpson(psd[idx_band], x=freqs[idx_band])
                else:
                    band_power = 0.0
                
                # Relative power
                if self._relative and total_power > 0:
                    band_power = band_power / total_power
                
                # Log transform
                if self._log:
                    band_power = np.log10(band_power + 1e-10)
                
                band_powers.append(band_power)
            
            channel_powers.append(band_powers)
        
        channel_powers = np.array(channel_powers)  # (n_channels, n_bands)
        
        # Average across channels if requested
        if self._average_channels:
            features = np.mean(channel_powers, axis=0)  # (n_bands,)
        else:
            features = channel_powers.flatten()  # (n_channels * n_bands,)
        
        return features
    
    def _compute_psd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density.
        
        Args:
            data: 1D signal, shape (n_samples,)
        
        Returns:
            Tuple of (frequencies, psd)
        """
        fs = self._sampling_rate
        
        if self._method == 'welch':
            # Welch's method
            nperseg = self._nperseg or min(int(fs), len(data))
            noverlap = self._noverlap or nperseg // 2
            
            freqs, psd = signal.welch(
                data,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann',
                scaling='density'
            )
            
        elif self._method == 'fft':
            # Simple FFT
            n = len(data)
            fft_vals = np.fft.rfft(data)
            psd = np.abs(fft_vals) ** 2 / n
            freqs = np.fft.rfftfreq(n, d=1/fs)
            
        elif self._method == 'multitaper':
            # Multitaper method (using scipy's periodogram as approximation)
            # Note: For true multitaper, consider using mne.time_frequency
            freqs, psd = signal.periodogram(
                data,
                fs=fs,
                window='hann',
                scaling='density'
            )
        
        else:
            raise ValueError(f"Unknown method: {self._method}")
        
        return freqs, psd
    
    # =========================================================================
    # FEATURE NAMES
    # =========================================================================
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive feature names."""
        names = []
        
        if self._average_channels:
            # Just band names
            for band_name in self._band_names:
                prefix = "rel_" if self._relative else ""
                suffix = "_log" if self._log else ""
                names.append(f"{prefix}{band_name}{suffix}")
        else:
            # Channel x band names
            n_channels = len(self._selected_channels) if self._selected_channels else self._n_channels
            
            for ch_idx in range(n_channels or 0):
                ch_name = self._channel_names[ch_idx] if ch_idx < len(self._channel_names) else f"ch{ch_idx}"
                
                for band_name in self._band_names:
                    prefix = "rel_" if self._relative else ""
                    suffix = "_log" if self._log else ""
                    names.append(f"{ch_name}_{prefix}{band_name}{suffix}")
        
        return names
    
    # =========================================================================
    # PARAMETER HANDLING
    # =========================================================================
    
    def _get_params_implementation(self) -> Dict[str, Any]:
        """Get band power specific parameters."""
        return {
            'bands': self._bands.copy(),
            'method': self._method,
            'relative': self._relative,
            'log': self._log,
            'average_channels': self._average_channels,
            'selected_channels': self._selected_channels,
            'nperseg': self._nperseg,
            'noverlap': self._noverlap,
        }
    
    def _set_params_implementation(self, **params) -> None:
        """Set band power specific parameters."""
        if 'bands' in params:
            self._bands = params['bands']
            self._band_names = list(params['bands'].keys())
        if 'method' in params:
            self._method = params['method']
        if 'relative' in params:
            self._relative = params['relative']
        if 'log' in params:
            self._log = params['log']
        if 'average_channels' in params:
            self._average_channels = params['average_channels']
        if 'selected_channels' in params:
            self._selected_channels = params['selected_channels']
        
        self._update_feature_count()
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_band_frequencies(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the frequency ranges for all bands.
        
        Returns:
            Dict mapping band names to (low, high) frequency tuples
        """
        return self._bands.copy()
    
    def add_band(self, name: str, low: float, high: float) -> 'BandPowerExtractor':
        """
        Add a frequency band.
        
        Args:
            name: Band name
            low: Low frequency (Hz)
            high: High frequency (Hz)
        
        Returns:
            Self for method chaining
        """
        self._bands[name] = (low, high)
        self._band_names = list(self._bands.keys())
        self._update_feature_count()
        return self
    
    def remove_band(self, name: str) -> 'BandPowerExtractor':
        """
        Remove a frequency band.
        
        Args:
            name: Band name to remove
        
        Returns:
            Self for method chaining
        """
        if name in self._bands:
            del self._bands[name]
            self._band_names = list(self._bands.keys())
            self._update_feature_count()
        return self
    
    def compute_psd_detailed(self,
                            trial: np.ndarray,
                            channel: int = 0
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute and return full PSD for visualization.
        
        Args:
            trial: Trial data, shape (n_channels, n_samples)
            channel: Channel index
        
        Returns:
            Tuple of (frequencies, psd)
        """
        return self._compute_psd(trial[channel])


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_band_power_extractor(
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = 'welch',
    relative: bool = False,
    log: bool = True,
    sampling_rate: float = 250.0,
    **kwargs
) -> BandPowerExtractor:
    """
    Factory function to create band power extractor.
    
    Args:
        bands: Frequency bands (default: mu and beta)
        method: PSD method ('welch', 'fft', 'multitaper')
        relative: Compute relative power
        log: Apply log transform
        sampling_rate: Sampling rate in Hz
        **kwargs: Additional configuration
    
    Returns:
        Configured BandPowerExtractor instance
    """
    extractor = BandPowerExtractor(
        bands=bands,
        method=method,
        relative=relative,
        log=log,
        **kwargs
    )
    
    extractor.initialize({
        'sampling_rate': sampling_rate,
        **kwargs
    })
    
    return extractor


def create_motor_imagery_band_power(
    sampling_rate: float = 250.0,
    **kwargs
) -> BandPowerExtractor:
    """
    Create band power extractor optimized for motor imagery.
    
    Uses mu (8-12 Hz) and beta (12-30 Hz) bands which are
    most relevant for motor imagery classification.
    
    Args:
        sampling_rate: Sampling rate in Hz
        **kwargs: Additional configuration
    
    Returns:
        Configured BandPowerExtractor for motor imagery
    """
    return create_band_power_extractor(
        bands=MOTOR_IMAGERY_BANDS,
        method='welch',
        relative=False,
        log=True,
        sampling_rate=sampling_rate,
        **kwargs
    )
