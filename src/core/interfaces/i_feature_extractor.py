"""
IFeatureExtractor Interface
===========================

This module defines the abstract interface for all feature extractors in the EEG-BCI framework.

Feature extractors are responsible for:
- Extracting meaningful features from preprocessed EEG signals
- Transforming raw time-series data into feature vectors
- Handling different feature domains (time, frequency, time-frequency, spatial)

Feature Types Supported:
- Time Domain: Statistical features (mean, variance, skewness, kurtosis, etc.)
- Frequency Domain: Power Spectral Density, band powers (delta, theta, alpha, mu, beta, gamma)
- Time-Frequency: Wavelet coefficients, STFT features
- Spatial: Common Spatial Patterns (CSP), source localization
- Connectivity: Coherence, phase-locking value, mutual information
- Deep Features: CNN-extracted features, embeddings

Design Principles:
- Each feature type has its own extractor class
- Extractors can be combined in pipelines
- Some extractors are trainable (CSP), others are not (band power)
- All extractors output standardized FeatureVector/FeatureSet objects

Example Usage:
    ```python
    # Single extractor
    csp = CSPExtractor(n_components=6)
    csp.fit(train_data, train_labels)
    features = csp.extract(test_data)
    
    # Feature pipeline (multiple extractors)
    pipeline = FeatureExtractionPipeline([
        CSPExtractor(n_components=6),
        BandPowerExtractor(bands={'mu': (8, 12), 'beta': (12, 30)}),
        WaveletExtractor(wavelet='db4', level=4)
    ])
    combined_features = pipeline.extract(data)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

# Forward references
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.types.eeg_data import EEGData
    from src.core.types.features import FeatureVector, FeatureSet


class IFeatureExtractor(ABC):
    """
    Abstract interface for EEG feature extractors.
    
    All feature extractor implementations must inherit from this class.
    This ensures consistent behavior and enables pipeline composition.
    
    Attributes:
        name (str): Unique identifier for this extractor
        feature_names (List[str]): Names of features produced by this extractor
        n_features (int): Number of features produced
        is_trainable (bool): Whether extractor requires fitting
    
    Feature Output Format:
        - Single trial: 1D numpy array of shape (n_features,)
        - Multiple trials: 2D numpy array of shape (n_trials, n_features)
        - FeatureVector/FeatureSet objects for rich metadata
    """
    
    # =========================================================================
    # ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this feature extractor.
        
        Returns:
            str: Extractor name (e.g., "csp", "band_power", "wavelet")
        
        Example:
            >>> extractor.name
            'csp'
        """
        pass
    
    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        """
        Indicates whether this extractor requires fitting/training.
        
        Returns:
            bool: True if fit() must be called before extract()
                  False if extractor is stateless
        
        Examples:
            - BandPowerExtractor: is_trainable = False
            - CSPExtractor: is_trainable = True (must fit spatial filters)
        """
        pass
    
    @property
    @abstractmethod
    def n_features(self) -> int:
        """
        Number of features produced by this extractor.
        
        Returns:
            int: Feature dimensionality
        
        Note:
            For trainable extractors, this may only be valid after fitting.
            
        Example:
            >>> csp = CSPExtractor(n_components=6)
            >>> csp.fit(data, labels)
            >>> csp.n_features
            6
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the feature extractor with configuration.
        
        Args:
            config: Dictionary containing extractor-specific settings
                Common keys:
                - 'sampling_rate': Required for frequency-based extractors
                - 'n_channels': Number of EEG channels
                - Additional keys depend on specific extractor
        
        Raises:
            ValueError: If required configuration is missing or invalid
            
        Example:
            >>> band_power = BandPowerExtractor()
            >>> band_power.initialize({
            ...     'sampling_rate': 250,
            ...     'bands': {'mu': (8, 12), 'beta': (12, 30)},
            ...     'window_size': 1.0  # seconds
            ... })
        """
        pass
    
    @abstractmethod
    def extract(self,
                data: Union[np.ndarray, 'EEGData'],
                **kwargs) -> np.ndarray:
        """
        Extract features from input data.
        
        This is the main feature extraction method.
        
        Args:
            data: Input EEG data
                - numpy array: Shape (channels, samples) for single trial
                              or (trials, channels, samples) for multiple trials
                - EEGData object
            **kwargs: Additional extraction options (extractor-specific)
        
        Returns:
            np.ndarray: Extracted features
                - Shape (n_features,) for single trial
                - Shape (n_trials, n_features) for multiple trials
        
        Raises:
            ValueError: If data format is invalid
            RuntimeError: If trainable extractor hasn't been fitted
            
        Example:
            >>> # Single trial
            >>> trial_data = np.random.randn(22, 1000)  # 22 channels, 1000 samples
            >>> features = extractor.extract(trial_data)
            >>> features.shape
            (12,)  # e.g., 12 features
            
            >>> # Multiple trials
            >>> batch_data = np.random.randn(100, 22, 1000)  # 100 trials
            >>> features = extractor.extract(batch_data)
            >>> features.shape
            (100, 12)
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get names/descriptions of extracted features.
        
        Returns list of human-readable feature names for interpretability.
        
        Returns:
            List[str]: Feature names in order of feature indices
            
        Example:
            >>> band_power.get_feature_names()
            ['C3_mu_power', 'C3_beta_power', 'C4_mu_power', 'C4_beta_power', ...]
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get current extractor parameters.
        
        Returns:
            Dict containing all extractor parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'IFeatureExtractor':
        """
        Set extractor parameters.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS - Override for trainable extractors
    # =========================================================================
    
    def fit(self,
            data: Union[np.ndarray, 'EEGData'],
            labels: Optional[np.ndarray] = None,
            **kwargs) -> 'IFeatureExtractor':
        """
        Fit the feature extractor on training data.
        
        Required for trainable extractors (e.g., CSP needs class labels).
        
        Args:
            data: Training data
                - numpy array: Shape (trials, channels, samples)
                - EEGData object with multiple trials
            labels: Class labels, shape (n_trials,)
                Required for supervised methods like CSP
            **kwargs: Additional fitting options
        
        Returns:
            Self for method chaining
        
        Raises:
            NotImplementedError: If called on stateless extractor that doesn't need fitting
            ValueError: If data or labels format is invalid
            
        Example:
            >>> csp.fit(train_data, train_labels)
            >>> features = csp.extract(test_data)
        """
        if self.is_trainable:
            raise NotImplementedError(
                f"{self.__class__.__name__} is trainable but fit() is not implemented"
            )
        return self
    
    def fit_extract(self,
                    data: Union[np.ndarray, 'EEGData'],
                    labels: Optional[np.ndarray] = None,
                    **kwargs) -> np.ndarray:
        """
        Fit the extractor and extract features in one step.
        
        Convenience method combining fit() and extract().
        
        Args:
            data: Training data to fit on and extract from
            labels: Optional class labels
            **kwargs: Additional options
        
        Returns:
            np.ndarray: Extracted features
        """
        self.fit(data, labels, **kwargs)
        return self.extract(data, **kwargs)
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get complete extractor state for serialization.
        
        Returns:
            Dict containing:
                - 'name': Extractor name
                - 'params': Configuration parameters
                - 'fitted_state': Learned parameters (if trainable)
                - 'feature_names': List of feature names
        """
        return {
            'name': self.name,
            'params': self.get_params(),
            'is_trainable': self.is_trainable,
            'n_features': self.n_features,
            'feature_names': self.get_feature_names(),
            'is_fitted': getattr(self, '_is_fitted', not self.is_trainable)
        }
    
    def load_state(self, state: Dict[str, Any]) -> 'IFeatureExtractor':
        """
        Load extractor state from dictionary.
        
        Args:
            state: State dictionary from get_state()
        
        Returns:
            Self for method chaining
        """
        if 'params' in state:
            self.set_params(**state['params'])
        return self
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def validate_input(self, data: Union[np.ndarray, 'EEGData']) -> Tuple[np.ndarray, bool]:
        """
        Validate and standardize input data.
        
        Args:
            data: Input data to validate
        
        Returns:
            Tuple of:
                - Standardized numpy array
                - Boolean indicating if input was single trial (2D) or batch (3D)
        
        Raises:
            ValueError: If data format is invalid
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # Single trial: (channels, samples)
                return data, True
            elif data.ndim == 3:
                # Multiple trials: (trials, channels, samples)
                return data, False
            else:
                raise ValueError(
                    f"Expected 2D or 3D array, got {data.ndim}D with shape {data.shape}"
                )
        else:
            # Assume EEGData, extract numpy array
            # This will be implemented when EEGData is defined
            raise NotImplementedError("EEGData input handling to be implemented")
    
    def __repr__(self) -> str:
        """String representation of the extractor."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"n_features={self.n_features if hasattr(self, '_is_fitted') and self._is_fitted else '?'}, "
            f"trainable={self.is_trainable})"
        )
    
    def __call__(self,
                 data: Union[np.ndarray, 'EEGData'],
                 **kwargs) -> np.ndarray:
        """
        Allow using extractor as callable.
        
        Equivalent to extract().
        
        Example:
            >>> features = extractor(data)  # Same as extractor.extract(data)
        """
        return self.extract(data, **kwargs)
