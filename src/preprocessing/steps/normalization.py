"""
Signal Normalization Preprocessor
=================================

This module implements various normalization methods for EEG signals.

Why Normalize EEG Signals?
-------------------------
Normalization is important for machine learning because:
1. It brings all channels to a common scale
2. It reduces the impact of inter-subject variability
3. It improves training stability for neural networks
4. It can reduce the effect of artifacts

Normalization Methods:
---------------------
1. Z-score (standardization): Centers to mean=0, std=1
   - Best for: Gaussian-distributed data, neural networks
   - Formula: (x - mean) / std

2. Min-Max scaling: Scales to [0, 1] or [-1, 1] range
   - Best for: When you need bounded values
   - Formula: (x - min) / (max - min)

3. Robust scaling: Uses median and IQR, less sensitive to outliers
   - Best for: Data with outliers/artifacts
   - Formula: (x - median) / IQR

4. Channel-wise normalization: Normalizes each channel independently
   - Best for: Handling different channel amplitude scales

5. Trial-wise normalization: Normalizes each trial independently
   - Best for: Reducing inter-trial variability

Usage Example:
    ```python
    from src.preprocessing.steps import Normalization
    
    # Create z-score normalizer
    normalizer = Normalization()
    normalizer.initialize({
        'method': 'zscore',
        'axis': 'channel',  # Normalize each channel independently
    })
    
    # Apply to EEG data
    normalized_data = normalizer.process(raw_eeg)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, Any, Union, Optional, Literal
import numpy as np
import logging

from src.core.interfaces.i_preprocessor import IPreprocessor
from src.core.types.eeg_data import EEGData


# Configure module logger
logger = logging.getLogger(__name__)


# Type alias for normalization methods
NormalizationMethod = Literal['zscore', 'minmax', 'robust', 'l2']

# Type alias for normalization axis
NormalizationAxis = Literal['channel', 'trial', 'global', 'sample']


class Normalization(IPreprocessor):
    """
    Signal normalization preprocessor.
    
    This preprocessor applies various normalization methods to EEG signals
    to standardize the data distribution.
    
    Attributes:
        _method (str): Normalization method ('zscore', 'minmax', 'robust', 'l2')
        _axis (str): Axis for normalization ('channel', 'trial', 'global', 'sample')
        _feature_range (tuple): Range for minmax scaling
        _epsilon (float): Small value to prevent division by zero
    
    Methods Overview:
        - zscore: Standardize to mean=0, std=1
        - minmax: Scale to specified range
        - robust: Use median and IQR (robust to outliers)
        - l2: Normalize to unit L2 norm
    
    Axis Options:
        - channel: Normalize each channel independently
        - trial: Normalize each trial independently
        - global: Normalize all data together
        - sample: Normalize each time sample across channels
    
    Example:
        >>> norm = Normalization()
        >>> norm.initialize({'method': 'zscore', 'axis': 'channel'})
        >>> normalized = norm.process(raw_signals)
    """
    
    def __init__(self):
        """Initialize the normalization preprocessor."""
        # Normalization parameters
        self._method: NormalizationMethod = 'zscore'
        self._axis: NormalizationAxis = 'channel'
        self._feature_range: tuple = (0, 1)
        self._epsilon: float = 1e-8
        
        # State tracking
        self._is_initialized: bool = False
        
        logger.debug("Normalization preprocessor instantiated")
    
    # =========================================================================
    # ABSTRACT PROPERTY IMPLEMENTATIONS
    # =========================================================================
    
    @property
    def name(self) -> str:
        """
        Unique identifier for this preprocessor.
        
        Returns:
            str: 'normalization'
        """
        return "normalization"
    
    @property
    def is_trainable(self) -> bool:
        """
        Indicates this preprocessor doesn't require training.
        
        Note: For fitted normalization (using training statistics on test data),
        set is_trainable=True and implement fit() method.
        
        Returns:
            bool: False (current implementation is stateless)
        """
        return False
    
    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the normalization preprocessor.
        
        Args:
            config: Configuration dictionary with keys:
                - 'method' (str, optional): Normalization method
                    Options: 'zscore', 'minmax', 'robust', 'l2'
                    Default: 'zscore'
                - 'axis' (str, optional): Normalization axis
                    Options: 'channel', 'trial', 'global', 'sample'
                    Default: 'channel'
                - 'feature_range' (tuple, optional): Range for minmax
                    Default: (0, 1)
                - 'epsilon' (float, optional): Small value for numerical stability
                    Default: 1e-8
        
        Raises:
            ValueError: If method or axis is invalid
        
        Example:
            >>> norm.initialize({
            ...     'method': 'zscore',
            ...     'axis': 'channel'
            ... })
        """
        logger.info("Initializing Normalization preprocessor")
        
        # Extract parameters
        self._method = config.get('method', 'zscore')
        self._axis = config.get('axis', 'channel')
        self._feature_range = tuple(config.get('feature_range', (0, 1)))
        self._epsilon = float(config.get('epsilon', 1e-8))
        
        # Validate parameters
        self._validate_parameters()
        
        self._is_initialized = True
        logger.info(
            f"Normalization initialized: method={self._method}, axis={self._axis}"
        )
    
    def process(
        self,
        data: Union[np.ndarray, EEGData],
        **kwargs
    ) -> Union[np.ndarray, EEGData]:
        """
        Apply normalization to the input data.
        
        Args:
            data: Input data to normalize
                - numpy array: Shape (channels, samples) or (trials, channels, samples)
                - EEGData: EEG data container
            **kwargs: Additional options (currently unused)
        
        Returns:
            Normalized data in the same format as input
        
        Raises:
            RuntimeError: If preprocessor not initialized
            ValueError: If data format is invalid
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Normalization not initialized. Call initialize() first."
            )
        
        # Handle EEGData input
        if isinstance(data, EEGData):
            return self._process_eegdata(data)
        
        # Handle numpy array input
        return self._process_array(data)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current normalization parameters.
        
        Returns:
            Dict containing all parameters
        """
        return {
            'method': self._method,
            'axis': self._axis,
            'feature_range': self._feature_range,
            'epsilon': self._epsilon
        }
    
    def set_params(self, **params) -> 'Normalization':
        """
        Set normalization parameters.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        """
        if 'method' in params:
            self._method = params['method']
        if 'axis' in params:
            self._axis = params['axis']
        if 'feature_range' in params:
            self._feature_range = tuple(params['feature_range'])
        if 'epsilon' in params:
            self._epsilon = float(params['epsilon'])
        
        if self._is_initialized:
            self._validate_parameters()
        
        return self
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _validate_parameters(self) -> None:
        """
        Validate normalization parameters.
        
        Raises:
            ValueError: If parameters are invalid
        """
        valid_methods = ['zscore', 'minmax', 'robust', 'l2']
        if self._method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self._method}'. "
                f"Valid options: {valid_methods}"
            )
        
        valid_axes = ['channel', 'trial', 'global', 'sample']
        if self._axis not in valid_axes:
            raise ValueError(
                f"Invalid axis '{self._axis}'. "
                f"Valid options: {valid_axes}"
            )
        
        if len(self._feature_range) != 2:
            raise ValueError("feature_range must be a tuple of length 2")
        
        if self._feature_range[0] >= self._feature_range[1]:
            raise ValueError(
                f"feature_range[0] ({self._feature_range[0]}) must be less than "
                f"feature_range[1] ({self._feature_range[1]})"
            )
    
    def _process_array(self, data: np.ndarray) -> np.ndarray:
        """
        Apply normalization to numpy array.
        
        Args:
            data: Input array
        
        Returns:
            Normalized array
        """
        self.validate_input(data)
        
        if data.ndim == 2:
            return self._normalize_2d(data)
        elif data.ndim == 3:
            return self._normalize_3d(data)
        else:
            raise ValueError(f"Unexpected data dimensions: {data.ndim}")
    
    def _normalize_2d(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize 2D array (channels, samples).
        
        Args:
            data: Shape (n_channels, n_samples)
        
        Returns:
            Normalized array
        """
        if self._axis == 'channel':
            # Normalize each channel independently
            return self._normalize_along_axis(data, axis=1)
        elif self._axis == 'sample':
            # Normalize each time point across channels
            return self._normalize_along_axis(data, axis=0)
        elif self._axis == 'global':
            # Normalize all data together
            return self._normalize_global(data)
        else:
            # For 2D data, 'trial' axis treats entire array as one trial
            return self._normalize_global(data)
    
    def _normalize_3d(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize 3D array (trials, channels, samples).
        
        Args:
            data: Shape (n_trials, n_channels, n_samples)
        
        Returns:
            Normalized array
        """
        n_trials = data.shape[0]
        normalized = np.zeros_like(data)
        
        if self._axis == 'trial':
            # Normalize each trial independently
            for t in range(n_trials):
                normalized[t] = self._normalize_global(data[t])
        
        elif self._axis == 'channel':
            # Normalize each channel across all trials
            for t in range(n_trials):
                normalized[t] = self._normalize_along_axis(data[t], axis=1)
        
        elif self._axis == 'sample':
            # Normalize each time point
            for t in range(n_trials):
                normalized[t] = self._normalize_along_axis(data[t], axis=0)
        
        elif self._axis == 'global':
            # Normalize all data together
            return self._normalize_global(data)
        
        return normalized
    
    def _normalize_along_axis(
        self,
        data: np.ndarray,
        axis: int
    ) -> np.ndarray:
        """
        Apply normalization along a specific axis.
        
        Args:
            data: Input array
            axis: Axis along which to normalize
        
        Returns:
            Normalized array
        """
        if self._method == 'zscore':
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            return (data - mean) / (std + self._epsilon)
        
        elif self._method == 'minmax':
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
            range_val = max_val - min_val + self._epsilon
            scaled = (data - min_val) / range_val
            # Scale to feature_range
            low, high = self._feature_range
            return scaled * (high - low) + low
        
        elif self._method == 'robust':
            median = np.median(data, axis=axis, keepdims=True)
            q75 = np.percentile(data, 75, axis=axis, keepdims=True)
            q25 = np.percentile(data, 25, axis=axis, keepdims=True)
            iqr = q75 - q25 + self._epsilon
            return (data - median) / iqr
        
        elif self._method == 'l2':
            norm = np.linalg.norm(data, axis=axis, keepdims=True)
            return data / (norm + self._epsilon)
        
        else:
            raise ValueError(f"Unknown method: {self._method}")
    
    def _normalize_global(self, data: np.ndarray) -> np.ndarray:
        """
        Apply global normalization to entire array.
        
        Args:
            data: Input array
        
        Returns:
            Normalized array
        """
        if self._method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            return (data - mean) / (std + self._epsilon)
        
        elif self._method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            range_val = max_val - min_val + self._epsilon
            scaled = (data - min_val) / range_val
            low, high = self._feature_range
            return scaled * (high - low) + low
        
        elif self._method == 'robust':
            median = np.median(data)
            q75 = np.percentile(data, 75)
            q25 = np.percentile(data, 25)
            iqr = q75 - q25 + self._epsilon
            return (data - median) / iqr
        
        elif self._method == 'l2':
            norm = np.linalg.norm(data)
            return data / (norm + self._epsilon)
        
        else:
            raise ValueError(f"Unknown method: {self._method}")
    
    def _process_eegdata(self, eeg_data: EEGData) -> EEGData:
        """
        Apply normalization to EEGData object.
        
        Args:
            eeg_data: EEGData object
        
        Returns:
            New EEGData with normalized signals
        """
        normalized_signals = self._normalize_2d(eeg_data.signals)
        
        return EEGData(
            signals=normalized_signals,
            sampling_rate=eeg_data.sampling_rate,
            channel_names=eeg_data.channel_names.copy(),
            events=eeg_data.events.copy(),
            subject_id=eeg_data.subject_id,
            session_id=eeg_data.session_id,
            recording_date=eeg_data.recording_date,
            source_file=eeg_data.source_file,
            metadata={
                **eeg_data.metadata,
                'normalized': True,
                'normalization_params': self.get_params()
            }
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Compute statistics used in normalization.
        
        Useful for understanding data distribution and for debugging.
        
        Args:
            data: Input data
        
        Returns:
            Dict with statistics (method-dependent)
        """
        if self._method == 'zscore':
            return {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
        elif self._method == 'minmax':
            return {
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data)
            }
        elif self._method == 'robust':
            return {
                'median': np.median(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'iqr': np.percentile(data, 75) - np.percentile(data, 25)
            }
        elif self._method == 'l2':
            return {
                'l2_norm': np.linalg.norm(data),
                'mean': np.mean(data)
            }
        return {}
    
    def __repr__(self) -> str:
        """String representation."""
        if self._is_initialized:
            return f"Normalization(method={self._method}, axis={self._axis})"
        return "Normalization(not initialized)"
