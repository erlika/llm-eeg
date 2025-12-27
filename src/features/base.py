"""
Base Feature Extractor
======================

This module provides the base implementation for all feature extractors in the EEG-BCI framework.

The BaseFeatureExtractor provides:
- Common functionality shared across all feature extractors
- Default implementations for IFeatureExtractor interface methods
- Utility methods for input validation, state management, and serialization
- EEGData integration for seamless data handling

Design Philosophy:
-----------------
1. **Template Method Pattern**: Base class defines the algorithm skeleton,
   subclasses implement specific extraction logic.
2. **Consistent Interface**: All extractors follow the same fit/extract pattern.
3. **State Management**: Unified approach to saving/loading fitted extractors.
4. **Extensibility**: Easy to add new extractors by extending this base class.

Inheritance Hierarchy:
---------------------
    IFeatureExtractor (interface)
           |
    BaseFeatureExtractor (this class)
           |
    +------+------+------+
    |      |      |      |
   CSP  BandPower TimeDomain  ... (concrete implementations)

Example:
    ```python
    from src.features.base import BaseFeatureExtractor
    
    class MyCustomExtractor(BaseFeatureExtractor):
        
        @property
        def name(self) -> str:
            return "my_custom"
        
        @property
        def is_trainable(self) -> bool:
            return False  # Stateless extractor
        
        def _extract_implementation(self, data: np.ndarray) -> np.ndarray:
            # Custom extraction logic
            return my_custom_features(data)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pickle
from pathlib import Path
import logging

from src.core.interfaces.i_feature_extractor import IFeatureExtractor
from src.core.types.eeg_data import EEGData, TrialData

# Configure logging
logger = logging.getLogger(__name__)


class BaseFeatureExtractor(IFeatureExtractor):
    """
    Base class for all feature extractors.
    
    Provides common functionality and default implementations for the
    IFeatureExtractor interface. Concrete extractors should extend this
    class and implement the abstract methods.
    
    Attributes:
        _is_fitted (bool): Whether the extractor has been fitted
        _is_initialized (bool): Whether the extractor has been initialized
        _config (Dict): Configuration dictionary
        _n_channels (int): Number of input channels (set during fit/initialize)
        _n_samples (int): Number of samples per trial (set during fit/initialize)
        _sampling_rate (float): Sampling rate in Hz
        _feature_names_list (List[str]): List of feature names (set after fit)
    
    Class Attributes:
        _registry_name (str): Name used for registry registration
        _registry_category (str): Category in component registry
    """
    
    # Registry attributes for auto-registration
    _registry_name: str = "base"
    _registry_category: str = "feature_extractor"
    
    def __init__(self):
        """Initialize the base feature extractor."""
        self._is_fitted: bool = False
        self._is_initialized: bool = False
        self._config: Dict[str, Any] = {}
        
        # Data shape info (set during fit or initialize)
        self._n_channels: Optional[int] = None
        self._n_samples: Optional[int] = None
        self._sampling_rate: float = 250.0
        
        # Feature info (set during/after fit)
        self._feature_names_list: List[str] = []
        self._n_features_value: int = 0
        
        # Channel names for feature naming
        self._channel_names: List[str] = []
        
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    # =========================================================================
    # ABSTRACT PROPERTIES (must be implemented by subclasses)
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this feature extractor.
        
        Returns:
            str: Extractor name (e.g., "csp", "band_power", "time_domain")
        """
        pass
    
    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        """
        Indicates whether this extractor requires fitting/training.
        
        Returns:
            bool: True if fit() must be called before extract()
        """
        pass
    
    # =========================================================================
    # PROPERTY IMPLEMENTATIONS
    # =========================================================================
    
    @property
    def n_features(self) -> int:
        """
        Number of features produced by this extractor.
        
        Returns:
            int: Feature dimensionality
            
        Note:
            For trainable extractors, this is only valid after fitting.
            For non-trainable extractors, this may require initialization
            with data shape information.
        """
        return self._n_features_value
    
    @property
    def is_fitted(self) -> bool:
        """Check if the extractor has been fitted."""
        return self._is_fitted
    
    @property
    def is_initialized(self) -> bool:
        """Check if the extractor has been initialized."""
        return self._is_initialized
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()
    
    # =========================================================================
    # INTERFACE METHOD IMPLEMENTATIONS
    # =========================================================================
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the feature extractor with configuration.
        
        Args:
            config: Dictionary containing extractor-specific settings
                Common keys:
                - 'sampling_rate': Sampling frequency in Hz (default: 250)
                - 'n_channels': Number of EEG channels
                - 'n_samples': Number of samples per trial
                - 'channel_names': List of channel names
                
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        # Store config
        self._config.update(config)
        
        # Extract common parameters
        self._sampling_rate = config.get('sampling_rate', 250.0)
        self._n_channels = config.get('n_channels', None)
        self._n_samples = config.get('n_samples', None)
        self._channel_names = config.get('channel_names', [])
        
        # Call subclass-specific initialization
        self._initialize_implementation(config)
        
        self._is_initialized = True
        logger.debug(f"{self.name} initialized with config: {list(config.keys())}")
    
    def _initialize_implementation(self, config: Dict[str, Any]) -> None:
        """
        Subclass-specific initialization.
        
        Override this method to implement custom initialization logic.
        
        Args:
            config: Configuration dictionary
        """
        pass  # Default: no additional initialization needed
    
    def fit(self,
            data: Union[np.ndarray, EEGData, List[TrialData]],
            labels: Optional[np.ndarray] = None,
            **kwargs) -> 'BaseFeatureExtractor':
        """
        Fit the feature extractor on training data.
        
        Args:
            data: Training data
                - numpy array: Shape (trials, channels, samples)
                - EEGData object: Will extract trials
                - List[TrialData]: List of trial objects
            labels: Class labels, shape (n_trials,)
                Required for supervised methods like CSP
            **kwargs: Additional fitting options
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If data or labels format is invalid
        """
        # Convert input to standard format
        X, y = self._prepare_data(data, labels)
        
        # Validate input
        self._validate_fit_input(X, y)
        
        # Store data shape info
        self._n_channels = X.shape[1]
        self._n_samples = X.shape[2]
        
        # Generate default channel names if not provided
        if not self._channel_names:
            self._channel_names = [f"Ch{i+1}" for i in range(self._n_channels)]
        
        # Call subclass implementation
        if self.is_trainable:
            self._fit_implementation(X, y, **kwargs)
        
        # Update feature names
        self._feature_names_list = self._generate_feature_names()
        self._n_features_value = len(self._feature_names_list)
        
        self._is_fitted = True
        logger.info(f"{self.name} fitted on data shape {X.shape}, "
                   f"producing {self.n_features} features")
        
        return self
    
    def _fit_implementation(self,
                           X: np.ndarray,
                           y: Optional[np.ndarray],
                           **kwargs) -> None:
        """
        Subclass-specific fitting implementation.
        
        Override this method to implement custom fitting logic.
        
        Args:
            X: Training data, shape (n_trials, n_channels, n_samples)
            y: Labels, shape (n_trials,)
            **kwargs: Additional options
        """
        # Default: no fitting needed for non-trainable extractors
        pass
    
    def extract(self,
                data: Union[np.ndarray, EEGData, List[TrialData]],
                **kwargs) -> np.ndarray:
        """
        Extract features from input data.
        
        Args:
            data: Input EEG data
                - numpy array: Shape (channels, samples) for single trial
                              or (trials, channels, samples) for multiple trials
                - EEGData object
                - List[TrialData]
            **kwargs: Additional extraction options
        
        Returns:
            np.ndarray: Extracted features
                - Shape (n_features,) for single trial
                - Shape (n_trials, n_features) for multiple trials
        
        Raises:
            RuntimeError: If trainable extractor hasn't been fitted
            ValueError: If data format is invalid
        """
        # Check if fitting is required
        if self.is_trainable and not self._is_fitted:
            raise RuntimeError(
                f"{self.name} is trainable and must be fitted before extraction. "
                f"Call fit() first."
            )
        
        # Convert input to standard format
        X, is_single_trial = self._prepare_extract_data(data)
        
        # Validate input
        self._validate_extract_input(X)
        
        # Call subclass implementation
        features = self._extract_implementation(X, **kwargs)
        
        # Handle single trial output
        if is_single_trial and features.ndim == 2:
            features = features.squeeze(0)
        
        return features
    
    @abstractmethod
    def _extract_implementation(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Subclass-specific feature extraction implementation.
        
        MUST be implemented by all subclasses.
        
        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
            **kwargs: Additional options
        
        Returns:
            np.ndarray: Features, shape (n_trials, n_features)
        """
        pass
    
    def get_feature_names(self) -> List[str]:
        """
        Get names/descriptions of extracted features.
        
        Returns:
            List[str]: Feature names in order of feature indices
        """
        if self._feature_names_list:
            return self._feature_names_list.copy()
        return self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """
        Generate feature names.
        
        Override this method to provide meaningful feature names.
        
        Returns:
            List[str]: Feature names
        """
        # Default: generic feature names
        if self._n_features_value > 0:
            return [f"{self.name}_feat_{i}" for i in range(self._n_features_value)]
        return []
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current extractor parameters.
        
        Returns:
            Dict containing all extractor parameters
        """
        params = {
            'name': self.name,
            'is_trainable': self.is_trainable,
            'sampling_rate': self._sampling_rate,
            'n_channels': self._n_channels,
            'n_samples': self._n_samples,
        }
        # Add subclass-specific params
        params.update(self._get_params_implementation())
        return params
    
    def _get_params_implementation(self) -> Dict[str, Any]:
        """
        Get subclass-specific parameters.
        
        Override to add custom parameters.
        
        Returns:
            Dict of parameters
        """
        return {}
    
    def set_params(self, **params) -> 'BaseFeatureExtractor':
        """
        Set extractor parameters.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        """
        # Update common params
        if 'sampling_rate' in params:
            self._sampling_rate = params['sampling_rate']
        if 'n_channels' in params:
            self._n_channels = params['n_channels']
        if 'n_samples' in params:
            self._n_samples = params['n_samples']
        if 'channel_names' in params:
            self._channel_names = params['channel_names']
        
        # Call subclass implementation
        self._set_params_implementation(**params)
        
        return self
    
    def _set_params_implementation(self, **params) -> None:
        """
        Set subclass-specific parameters.
        
        Override to handle custom parameters.
        
        Args:
            **params: Parameters to set
        """
        pass
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get complete extractor state for serialization.
        
        Returns:
            Dict containing all state information
        """
        state = {
            'name': self.name,
            'class': self.__class__.__name__,
            'is_trainable': self.is_trainable,
            'is_fitted': self._is_fitted,
            'is_initialized': self._is_initialized,
            'config': self._config,
            'params': self.get_params(),
            'n_features': self._n_features_value,
            'feature_names': self._feature_names_list,
            'n_channels': self._n_channels,
            'n_samples': self._n_samples,
            'sampling_rate': self._sampling_rate,
            'channel_names': self._channel_names,
        }
        
        # Add subclass-specific state
        state['fitted_state'] = self._get_fitted_state()
        
        return state
    
    def _get_fitted_state(self) -> Dict[str, Any]:
        """
        Get subclass-specific fitted state.
        
        Override to save learned parameters (e.g., CSP filters).
        
        Returns:
            Dict of fitted state
        """
        return {}
    
    def load_state(self, state: Dict[str, Any]) -> 'BaseFeatureExtractor':
        """
        Load extractor state from dictionary.
        
        Args:
            state: State dictionary from get_state()
        
        Returns:
            Self for method chaining
        """
        # Restore common state
        self._is_fitted = state.get('is_fitted', False)
        self._is_initialized = state.get('is_initialized', False)
        self._config = state.get('config', {})
        self._n_features_value = state.get('n_features', 0)
        self._feature_names_list = state.get('feature_names', [])
        self._n_channels = state.get('n_channels')
        self._n_samples = state.get('n_samples')
        self._sampling_rate = state.get('sampling_rate', 250.0)
        self._channel_names = state.get('channel_names', [])
        
        # Restore subclass-specific state
        if 'fitted_state' in state:
            self._load_fitted_state(state['fitted_state'])
        
        logger.debug(f"{self.name} state loaded")
        return self
    
    def _load_fitted_state(self, fitted_state: Dict[str, Any]) -> None:
        """
        Load subclass-specific fitted state.
        
        Override to restore learned parameters.
        
        Args:
            fitted_state: Dictionary of fitted state
        """
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the extractor to disk.
        
        Args:
            path: File path to save the extractor
        
        Raises:
            IOError: If file cannot be written
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = self.get_state()
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"{self.name} saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseFeatureExtractor':
        """
        Load an extractor from disk.
        
        Args:
            path: File path to load from
        
        Returns:
            Loaded extractor instance
        
        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Extractor file not found: {path}")
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance and load state
        instance = cls()
        instance.load_state(state)
        
        logger.info(f"Loaded {instance.name} from {path}")
        return instance
    
    # =========================================================================
    # DATA PREPARATION UTILITIES
    # =========================================================================
    
    def _prepare_data(self,
                      data: Union[np.ndarray, EEGData, List[TrialData]],
                      labels: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare input data for fitting.
        
        Converts various input formats to standard numpy arrays.
        
        Args:
            data: Input data in various formats
            labels: Optional labels
        
        Returns:
            Tuple of (X, y) where X is shape (n_trials, n_channels, n_samples)
        """
        if isinstance(data, np.ndarray):
            X = data
            y = labels
            
        elif isinstance(data, EEGData):
            X, y = data.get_trials_array()
            if labels is not None:
                y = labels  # Override with provided labels
                
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], TrialData):
            X = np.array([t.signals for t in data])
            y = np.array([t.label for t in data]) if labels is None else labels
            
            # Store channel names from first trial
            if data[0].channel_names:
                self._channel_names = data[0].channel_names
                
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                f"Expected np.ndarray, EEGData, or List[TrialData]"
            )
        
        # Ensure 3D shape
        if X.ndim == 2:
            X = X[np.newaxis, ...]  # Add trial dimension
        
        return X, y
    
    def _prepare_extract_data(self,
                              data: Union[np.ndarray, EEGData, List[TrialData]]
                              ) -> Tuple[np.ndarray, bool]:
        """
        Prepare input data for extraction.
        
        Args:
            data: Input data in various formats
        
        Returns:
            Tuple of (X, is_single_trial)
        """
        is_single_trial = False
        
        if isinstance(data, np.ndarray):
            X = data
            if X.ndim == 2:
                X = X[np.newaxis, ...]
                is_single_trial = True
                
        elif isinstance(data, EEGData):
            X, _ = data.get_trials_array()
            
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], TrialData):
            X = np.array([t.signals for t in data])
            is_single_trial = len(data) == 1
            
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                f"Expected np.ndarray, EEGData, or List[TrialData]"
            )
        
        return X, is_single_trial
    
    # =========================================================================
    # VALIDATION UTILITIES
    # =========================================================================
    
    def _validate_fit_input(self,
                           X: np.ndarray,
                           y: Optional[np.ndarray]) -> None:
        """
        Validate input data for fitting.
        
        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
            y: Labels, shape (n_trials,)
        
        Raises:
            ValueError: If validation fails
        """
        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input (trials, channels, samples), "
                f"got {X.ndim}D with shape {X.shape}"
            )
        
        n_trials = X.shape[0]
        
        if n_trials < 2:
            raise ValueError(
                f"Need at least 2 trials for fitting, got {n_trials}"
            )
        
        if self.is_trainable and y is None:
            raise ValueError(
                f"{self.name} is trainable and requires labels for fitting"
            )
        
        if y is not None and len(y) != n_trials:
            raise ValueError(
                f"Number of labels ({len(y)}) doesn't match "
                f"number of trials ({n_trials})"
            )
        
        if np.isnan(X).any():
            raise ValueError("Input data contains NaN values")
        
        if np.isinf(X).any():
            raise ValueError("Input data contains infinite values")
    
    def _validate_extract_input(self, X: np.ndarray) -> None:
        """
        Validate input data for extraction.
        
        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
        
        Raises:
            ValueError: If validation fails
        """
        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input (trials, channels, samples), "
                f"got {X.ndim}D with shape {X.shape}"
            )
        
        # Check channel count matches (if set)
        if self._n_channels is not None and X.shape[1] != self._n_channels:
            raise ValueError(
                f"Channel count mismatch: extractor fitted with "
                f"{self._n_channels} channels, but input has {X.shape[1]}"
            )
        
        if np.isnan(X).any():
            raise ValueError("Input data contains NaN values")
        
        if np.isinf(X).any():
            raise ValueError("Input data contains infinite values")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def clone(self) -> 'BaseFeatureExtractor':
        """
        Create a clone of this extractor with same parameters.
        
        Returns:
            New instance with same parameters (but not fitted)
        """
        clone = self.__class__()
        clone.set_params(**self.get_params())
        if self._is_initialized:
            clone.initialize(self._config)
        return clone
    
    def __repr__(self) -> str:
        """String representation of the extractor."""
        status = "fitted" if self._is_fitted else "not fitted"
        n_feat = self._n_features_value if self._is_fitted else "?"
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"trainable={self.is_trainable}, "
            f"n_features={n_feat}, "
            f"{status})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return repr(self)
