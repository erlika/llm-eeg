"""
Time Domain Feature Extractor
=============================

This module implements time domain statistical feature extraction for EEG signals.

Time domain features capture the statistical properties of the signal in the time domain,
without requiring frequency transformation. These features are computationally efficient
and can capture important signal characteristics.

Available Features:
------------------
| Feature        | Description                           | Formula                |
|----------------|---------------------------------------|------------------------|
| mean           | Average amplitude                     | mean(x)                |
| variance       | Signal power/variability              | var(x)                 |
| std            | Standard deviation                    | std(x)                 |
| skewness       | Asymmetry of distribution             | E[(x-μ)³]/σ³          |
| kurtosis       | Tailedness of distribution            | E[(x-μ)⁴]/σ⁴ - 3      |
| rms            | Root mean square                      | sqrt(mean(x²))         |
| peak_to_peak   | Max - Min amplitude                   | max(x) - min(x)        |
| zero_crossings | Rate of sign changes                  | count(sign changes)/N  |
| line_length    | Total signal length                   | sum(|diff(x)|)         |
| hjorth_*       | Hjorth parameters (activity, mobility, complexity) |

Hjorth Parameters (commonly used in EEG):
----------------------------------------
- Activity: Variance of the signal (signal power)
- Mobility: sqrt(var(dx/dt) / var(x)) - Mean frequency
- Complexity: mobility(dx/dt) / mobility(x) - Bandwidth

Usage Example:
    ```python
    from src.features.extractors.time_domain import TimeDomainExtractor
    
    # Create extractor with selected features
    td = TimeDomainExtractor(
        features=['mean', 'variance', 'skewness', 'kurtosis', 'rms']
    )
    td.initialize({'sampling_rate': 250})
    
    # Extract features
    features = td.extract(X)  # Shape: (n_trials, n_channels * n_features)
    ```

References:
----------
1. Hjorth, "EEG analysis based on time domain properties", 
   Electroencephalography and Clinical Neurophysiology, 1970
2. Vidaurre et al., "A fully on-line adaptive BCI", IEEE TBME, 2006

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
from scipy import stats
import logging

from src.features.base import BaseFeatureExtractor
from src.core.registry import registered

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT FEATURE SETS
# =============================================================================

# All available features
ALL_FEATURES = [
    'mean', 'variance', 'std', 'skewness', 'kurtosis',
    'rms', 'peak_to_peak', 'min', 'max',
    'zero_crossings', 'line_length',
    'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
]

# Default feature set (commonly used)
DEFAULT_FEATURES = [
    'mean', 'variance', 'skewness', 'kurtosis', 'rms'
]

# Hjorth parameters only
HJORTH_FEATURES = [
    'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
]

# Simple statistics
SIMPLE_FEATURES = [
    'mean', 'variance', 'std'
]


@registered('feature_extractor', 'time_domain')
class TimeDomainExtractor(BaseFeatureExtractor):
    """
    Time domain statistical feature extractor.
    
    Extracts statistical features from EEG signals in the time domain.
    This is a stateless extractor (no fitting required).
    
    Attributes:
        features (List[str]): List of features to extract.
        normalize (bool): If True, normalize features to zero mean, unit variance.
        average_channels (bool): If True, average features across channels.
        selected_channels (List[int]): If set, only use these channels.
    
    Output Format:
        Default: (n_trials, n_channels * n_features)
        With average_channels=True: (n_trials, n_features)
        
        Feature order: [ch0_feat0, ch0_feat1, ..., ch1_feat0, ch1_feat1, ...]
    
    Example:
        >>> td = TimeDomainExtractor(features=['variance', 'hjorth_mobility'])
        >>> td.initialize({'sampling_rate': 250})
        >>> features = td.extract(X)  # (n_trials, 22 * 2) = (n_trials, 44)
    """
    
    def __init__(self,
                 features: Optional[List[str]] = None,
                 normalize: bool = False,
                 average_channels: bool = False,
                 selected_channels: Optional[List[int]] = None):
        """
        Initialize time domain extractor.
        
        Args:
            features: List of feature names to extract.
                Default: ['mean', 'variance', 'skewness', 'kurtosis', 'rms']
            normalize: Normalize features to zero mean, unit variance.
            average_channels: Average features across all channels.
            selected_channels: Only use these channel indices.
        """
        super().__init__()
        
        # Feature configuration
        self._features = features or DEFAULT_FEATURES.copy()
        self._normalize = normalize
        self._average_channels = average_channels
        self._selected_channels = selected_channels
        
        # Validate features
        for feat in self._features:
            if feat not in ALL_FEATURES:
                raise ValueError(
                    f"Unknown feature '{feat}'. "
                    f"Available: {ALL_FEATURES}"
                )
        
        # Feature computation functions
        self._feature_funcs: Dict[str, Callable] = {
            'mean': self._compute_mean,
            'variance': self._compute_variance,
            'std': self._compute_std,
            'skewness': self._compute_skewness,
            'kurtosis': self._compute_kurtosis,
            'rms': self._compute_rms,
            'peak_to_peak': self._compute_peak_to_peak,
            'min': self._compute_min,
            'max': self._compute_max,
            'zero_crossings': self._compute_zero_crossings,
            'line_length': self._compute_line_length,
            'hjorth_activity': self._compute_hjorth_activity,
            'hjorth_mobility': self._compute_hjorth_mobility,
            'hjorth_complexity': self._compute_hjorth_complexity,
        }
        
        # Normalization statistics (computed during fit if normalize=True)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
    
    # =========================================================================
    # PROPERTY IMPLEMENTATIONS
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Extractor name."""
        return "time_domain"
    
    @property
    def is_trainable(self) -> bool:
        """Time domain features are stateless (unless normalization is used)."""
        return self._normalize
    
    @property
    def features(self) -> List[str]:
        """Get list of features."""
        return self._features.copy()
    
    @features.setter
    def features(self, value: List[str]) -> None:
        """Set feature list."""
        for feat in value:
            if feat not in ALL_FEATURES:
                raise ValueError(f"Unknown feature '{feat}'")
        self._features = value
        self._update_feature_count()
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _initialize_implementation(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        if 'features' in config:
            self._features = config['features']
        if 'normalize' in config:
            self._normalize = config['normalize']
        if 'average_channels' in config:
            self._average_channels = config['average_channels']
        if 'selected_channels' in config:
            self._selected_channels = config['selected_channels']
        
        self._update_feature_count()
    
    def _update_feature_count(self) -> None:
        """Update feature count based on configuration."""
        n_features = len(self._features)
        
        if self._average_channels:
            self._n_features_value = n_features
        elif self._selected_channels is not None:
            self._n_features_value = len(self._selected_channels) * n_features
        elif self._n_channels is not None:
            self._n_features_value = self._n_channels * n_features
        else:
            self._n_features_value = 0
    
    # =========================================================================
    # FIT IMPLEMENTATION
    # =========================================================================
    
    def _fit_implementation(self,
                           X: np.ndarray,
                           y: Optional[np.ndarray],
                           **kwargs) -> None:
        """
        Fit for computing normalization statistics if needed.
        """
        self._update_feature_count()
        
        if self._normalize:
            # Extract features on training data
            features = self._extract_raw(X)
            
            # Compute normalization statistics
            self._feature_means = np.mean(features, axis=0)
            self._feature_stds = np.std(features, axis=0)
            
            # Avoid division by zero
            self._feature_stds[self._feature_stds < 1e-10] = 1.0
    
    # =========================================================================
    # EXTRACTION IMPLEMENTATION
    # =========================================================================
    
    def _extract_implementation(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Extract time domain features.
        
        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
        
        Returns:
            Features, shape (n_trials, n_features)
        """
        features = self._extract_raw(X)
        
        # Apply normalization if fitted
        if self._normalize and self._feature_means is not None:
            features = (features - self._feature_means) / self._feature_stds
        
        return features
    
    def _extract_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Extract raw (unnormalized) features.
        
        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
        
        Returns:
            Raw features
        """
        n_trials, n_channels, n_samples = X.shape
        
        # Determine channels to use
        if self._selected_channels is not None:
            channel_indices = self._selected_channels
        else:
            channel_indices = list(range(n_channels))
        
        n_selected_channels = len(channel_indices)
        n_feat = len(self._features)
        
        # Output shape
        if self._average_channels:
            n_output_features = n_feat
        else:
            n_output_features = n_selected_channels * n_feat
        
        features = np.zeros((n_trials, n_output_features))
        
        for trial_idx in range(n_trials):
            trial_features = self._extract_trial(
                X[trial_idx], channel_indices
            )
            features[trial_idx] = trial_features
        
        # Update feature count if not set
        if self._n_features_value == 0:
            self._n_features_value = n_output_features
            self._feature_names_list = self._generate_feature_names()
        
        return features
    
    def _extract_trial(self,
                       trial: np.ndarray,
                       channel_indices: List[int]) -> np.ndarray:
        """
        Extract features from a single trial.
        
        Args:
            trial: Trial data, shape (n_channels, n_samples)
            channel_indices: Channels to extract from
        
        Returns:
            Features for this trial
        """
        channel_features = []
        
        for ch_idx in channel_indices:
            channel_data = trial[ch_idx]
            
            # Compute each feature
            feat_values = []
            for feat_name in self._features:
                func = self._feature_funcs[feat_name]
                value = func(channel_data)
                feat_values.append(value)
            
            channel_features.append(feat_values)
        
        channel_features = np.array(channel_features)  # (n_channels, n_features)
        
        # Average across channels if requested
        if self._average_channels:
            return np.mean(channel_features, axis=0)
        else:
            return channel_features.flatten()
    
    # =========================================================================
    # FEATURE COMPUTATION FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def _compute_mean(x: np.ndarray) -> float:
        """Compute signal mean."""
        return np.mean(x)
    
    @staticmethod
    def _compute_variance(x: np.ndarray) -> float:
        """Compute signal variance."""
        return np.var(x)
    
    @staticmethod
    def _compute_std(x: np.ndarray) -> float:
        """Compute standard deviation."""
        return np.std(x)
    
    @staticmethod
    def _compute_skewness(x: np.ndarray) -> float:
        """Compute skewness (asymmetry)."""
        return float(stats.skew(x))
    
    @staticmethod
    def _compute_kurtosis(x: np.ndarray) -> float:
        """Compute kurtosis (tailedness)."""
        return float(stats.kurtosis(x))
    
    @staticmethod
    def _compute_rms(x: np.ndarray) -> float:
        """Compute root mean square."""
        return np.sqrt(np.mean(x ** 2))
    
    @staticmethod
    def _compute_peak_to_peak(x: np.ndarray) -> float:
        """Compute peak-to-peak amplitude."""
        return np.max(x) - np.min(x)
    
    @staticmethod
    def _compute_min(x: np.ndarray) -> float:
        """Compute minimum value."""
        return np.min(x)
    
    @staticmethod
    def _compute_max(x: np.ndarray) -> float:
        """Compute maximum value."""
        return np.max(x)
    
    @staticmethod
    def _compute_zero_crossings(x: np.ndarray) -> float:
        """Compute zero-crossing rate."""
        # Remove mean to get zero crossings around mean
        x_centered = x - np.mean(x)
        
        # Count sign changes
        signs = np.sign(x_centered)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        
        # Return rate (crossings per sample)
        return crossings / len(x)
    
    @staticmethod
    def _compute_line_length(x: np.ndarray) -> float:
        """Compute line length (total absolute derivative)."""
        return np.sum(np.abs(np.diff(x)))
    
    @staticmethod
    def _compute_hjorth_activity(x: np.ndarray) -> float:
        """Compute Hjorth Activity (variance)."""
        return np.var(x)
    
    @staticmethod
    def _compute_hjorth_mobility(x: np.ndarray) -> float:
        """
        Compute Hjorth Mobility.
        
        Mobility = sqrt(var(dx/dt) / var(x))
        Represents the mean frequency of the signal.
        """
        dx = np.diff(x)
        var_x = np.var(x)
        var_dx = np.var(dx)
        
        if var_x < 1e-10:
            return 0.0
        
        return np.sqrt(var_dx / var_x)
    
    @staticmethod
    def _compute_hjorth_complexity(x: np.ndarray) -> float:
        """
        Compute Hjorth Complexity.
        
        Complexity = mobility(dx/dt) / mobility(x)
        Represents the bandwidth of the signal.
        """
        dx = np.diff(x)
        ddx = np.diff(dx)
        
        var_x = np.var(x)
        var_dx = np.var(dx)
        var_ddx = np.var(ddx)
        
        if var_x < 1e-10 or var_dx < 1e-10:
            return 0.0
        
        mobility_x = np.sqrt(var_dx / var_x)
        mobility_dx = np.sqrt(var_ddx / var_dx)
        
        if mobility_x < 1e-10:
            return 0.0
        
        return mobility_dx / mobility_x
    
    # =========================================================================
    # FEATURE NAMES
    # =========================================================================
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive feature names."""
        names = []
        
        if self._average_channels:
            # Just feature names
            names = [f"td_{feat}" for feat in self._features]
        else:
            # Channel x feature names
            n_channels = len(self._selected_channels) if self._selected_channels else self._n_channels
            
            for ch_idx in range(n_channels or 0):
                ch_name = self._channel_names[ch_idx] if ch_idx < len(self._channel_names) else f"ch{ch_idx}"
                
                for feat in self._features:
                    names.append(f"{ch_name}_{feat}")
        
        return names
    
    # =========================================================================
    # PARAMETER HANDLING
    # =========================================================================
    
    def _get_params_implementation(self) -> Dict[str, Any]:
        """Get time domain specific parameters."""
        return {
            'features': self._features.copy(),
            'normalize': self._normalize,
            'average_channels': self._average_channels,
            'selected_channels': self._selected_channels,
        }
    
    def _set_params_implementation(self, **params) -> None:
        """Set time domain specific parameters."""
        if 'features' in params:
            self._features = params['features']
        if 'normalize' in params:
            self._normalize = params['normalize']
        if 'average_channels' in params:
            self._average_channels = params['average_channels']
        if 'selected_channels' in params:
            self._selected_channels = params['selected_channels']
        
        self._update_feature_count()
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def _get_fitted_state(self) -> Dict[str, Any]:
        """Get fitted state for serialization."""
        state = {}
        if self._feature_means is not None:
            state['feature_means'] = self._feature_means.tolist()
        if self._feature_stds is not None:
            state['feature_stds'] = self._feature_stds.tolist()
        return state
    
    def _load_fitted_state(self, state: Dict[str, Any]) -> None:
        """Load fitted state from serialization."""
        if 'feature_means' in state:
            self._feature_means = np.array(state['feature_means'])
        if 'feature_stds' in state:
            self._feature_stds = np.array(state['feature_stds'])
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def add_feature(self, feature: str) -> 'TimeDomainExtractor':
        """
        Add a feature to extract.
        
        Args:
            feature: Feature name
        
        Returns:
            Self for method chaining
        """
        if feature not in ALL_FEATURES:
            raise ValueError(f"Unknown feature '{feature}'")
        if feature not in self._features:
            self._features.append(feature)
            self._update_feature_count()
        return self
    
    def remove_feature(self, feature: str) -> 'TimeDomainExtractor':
        """
        Remove a feature.
        
        Args:
            feature: Feature name to remove
        
        Returns:
            Self for method chaining
        """
        if feature in self._features:
            self._features.remove(feature)
            self._update_feature_count()
        return self
    
    @staticmethod
    def get_available_features() -> List[str]:
        """
        Get list of all available features.
        
        Returns:
            List of feature names
        """
        return ALL_FEATURES.copy()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_time_domain_extractor(
    features: Optional[List[str]] = None,
    normalize: bool = False,
    sampling_rate: float = 250.0,
    **kwargs
) -> TimeDomainExtractor:
    """
    Factory function to create time domain extractor.
    
    Args:
        features: List of features to extract
        normalize: Normalize features
        sampling_rate: Sampling rate in Hz
        **kwargs: Additional configuration
    
    Returns:
        Configured TimeDomainExtractor instance
    """
    extractor = TimeDomainExtractor(
        features=features,
        normalize=normalize,
        **kwargs
    )
    
    extractor.initialize({
        'sampling_rate': sampling_rate,
        **kwargs
    })
    
    return extractor


def create_hjorth_extractor(
    sampling_rate: float = 250.0,
    **kwargs
) -> TimeDomainExtractor:
    """
    Create extractor for Hjorth parameters only.
    
    Hjorth parameters (Activity, Mobility, Complexity) are commonly
    used in EEG analysis for characterizing signal properties.
    
    Args:
        sampling_rate: Sampling rate in Hz
        **kwargs: Additional configuration
    
    Returns:
        Configured TimeDomainExtractor for Hjorth parameters
    """
    return create_time_domain_extractor(
        features=HJORTH_FEATURES,
        sampling_rate=sampling_rate,
        **kwargs
    )
