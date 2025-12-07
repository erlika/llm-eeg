"""
IPreprocessor Interface
=======================

This module defines the abstract interface for all preprocessing steps in the EEG-BCI framework.

Preprocessors are responsible for:
- Filtering signals (bandpass, notch, highpass, lowpass)
- Removing artifacts (EOG, EMG, motion)
- Normalizing data
- Resampling signals
- Any other signal conditioning operations

Design Principles:
- Each preprocessing step is a separate class implementing IPreprocessor
- Steps can be composed into pipelines (order matters)
- All steps are stateless by default (same input → same output)
- Some steps may have learnable parameters (e.g., ICA)
- All steps must support both single trial and batch processing

Pipeline Composition:
    ```python
    # Preprocessing steps are composable
    pipeline = PreprocessingPipeline([
        BandpassFilter(low=8, high=30),
        NotchFilter(freq=50),
        ArtifactRemoval(method='regression'),
        Normalization(method='zscore')
    ])
    
    # Apply to data
    processed_data = pipeline.process(raw_data)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

# Forward reference for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.types.eeg_data import EEGData


class IPreprocessor(ABC):
    """
    Abstract interface for EEG preprocessing steps.
    
    All preprocessing implementations must inherit from this class.
    This ensures consistent behavior and enables pipeline composition.
    
    Attributes:
        name (str): Unique identifier for this preprocessor
        is_fitted (bool): Whether the preprocessor has been fitted (if applicable)
    
    Processing Modes:
        - Stateless: Output depends only on input (e.g., filtering)
        - Stateful: Requires fitting on training data first (e.g., ICA, CSP)
    
    Data Formats Supported:
        - Raw numpy arrays: (channels, samples) or (trials, channels, samples)
        - EEGData objects: Standardized data container
    """
    
    # =========================================================================
    # ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this preprocessing step.
        
        Returns:
            str: Preprocessor name (e.g., "bandpass_filter", "notch_filter", "ica")
        
        Example:
            >>> preprocessor.name
            'bandpass_filter'
        """
        pass
    
    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        """
        Indicates whether this preprocessor requires fitting/training.
        
        Returns:
            bool: True if fit() must be called before process()
                  False if preprocessor is stateless
        
        Examples:
            - BandpassFilter: is_trainable = False (stateless)
            - ICA: is_trainable = True (must fit components first)
            - CSP: is_trainable = True (must fit spatial filters)
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the preprocessor with configuration parameters.
        
        Called after instantiation to set up the preprocessor.
        Configuration varies by preprocessor type.
        
        Args:
            config: Dictionary containing preprocessor-specific settings
                Common keys:
                - 'sampling_rate': Required for most filters
                - Additional keys depend on specific preprocessor
        
        Raises:
            ValueError: If required configuration is missing or invalid
            
        Example:
            >>> bandpass = BandpassFilter()
            >>> bandpass.initialize({
            ...     'sampling_rate': 250,
            ...     'low_freq': 8,
            ...     'high_freq': 30,
            ...     'filter_order': 5
            ... })
        """
        pass
    
    @abstractmethod
    def process(self, 
                data: Union[np.ndarray, 'EEGData'],
                **kwargs) -> Union[np.ndarray, 'EEGData']:
        """
        Apply preprocessing to the input data.
        
        This is the main processing method. It applies the preprocessing
        transformation to the input data and returns the processed result.
        
        Args:
            data: Input data to process
                - numpy array: Shape (channels, samples) or (trials, channels, samples)
                - EEGData object: Standardized data container
            **kwargs: Additional processing options (preprocessor-specific)
        
        Returns:
            Processed data in same format as input:
                - numpy array if input was numpy array
                - EEGData if input was EEGData
        
        Raises:
            ValueError: If data format is invalid
            RuntimeError: If trainable preprocessor hasn't been fitted
            
        Example:
            >>> raw_signal = np.random.randn(22, 1000)  # 22 channels, 1000 samples
            >>> filtered = bandpass.process(raw_signal)
            >>> filtered.shape
            (22, 1000)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get current preprocessor parameters.
        
        Returns all configuration parameters, useful for:
        - Logging and reproducibility
        - Saving/loading preprocessor state
        - Debugging
        
        Returns:
            Dict containing all preprocessor parameters
            
        Example:
            >>> params = bandpass.get_params()
            >>> print(params)
            {'low_freq': 8, 'high_freq': 30, 'filter_order': 5, 'sampling_rate': 250}
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'IPreprocessor':
        """
        Set preprocessor parameters.
        
        Allows modifying parameters after initialization.
        Returns self for method chaining.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If parameter name is invalid
            
        Example:
            >>> bandpass.set_params(low_freq=4, high_freq=40)
            >>> bandpass.get_params()['low_freq']
            4
        """
        pass
    
    # =========================================================================
    # OPTIONAL ABSTRACT METHODS - Override if is_trainable=True
    # =========================================================================
    
    def fit(self, 
            data: Union[np.ndarray, 'EEGData'],
            labels: Optional[np.ndarray] = None,
            **kwargs) -> 'IPreprocessor':
        """
        Fit the preprocessor on training data (for trainable preprocessors).
        
        Must be called before process() for trainable preprocessors.
        For stateless preprocessors, this method does nothing.
        
        Args:
            data: Training data to fit on
                - numpy array: Shape (trials, channels, samples)
                - EEGData object with multiple trials
            labels: Optional class labels (required for some methods like CSP)
            **kwargs: Additional fitting options
        
        Returns:
            Self for method chaining
        
        Raises:
            NotImplementedError: If called on stateless preprocessor
            ValueError: If data format is invalid
            
        Example:
            >>> # For ICA
            >>> ica.fit(training_data)
            >>> processed = ica.process(test_data)
            
            >>> # For CSP (needs labels)
            >>> csp.fit(training_data, labels=training_labels)
        """
        if self.is_trainable:
            raise NotImplementedError(
                f"{self.__class__.__name__} is trainable but fit() is not implemented"
            )
        # For stateless preprocessors, do nothing
        return self
    
    def fit_transform(self,
                      data: Union[np.ndarray, 'EEGData'],
                      labels: Optional[np.ndarray] = None,
                      **kwargs) -> Union[np.ndarray, 'EEGData']:
        """
        Fit the preprocessor and transform data in one step.
        
        Convenience method that combines fit() and process().
        
        Args:
            data: Data to fit on and transform
            labels: Optional class labels
            **kwargs: Additional options
        
        Returns:
            Transformed data
            
        Example:
            >>> processed = ica.fit_transform(data)
        """
        self.fit(data, labels, **kwargs)
        return self.process(data, **kwargs)
    
    # =========================================================================
    # STATE MANAGEMENT METHODS
    # =========================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the complete state of the preprocessor for serialization.
        
        Includes both parameters and any learned state (for trainable preprocessors).
        Used for checkpointing and reproducibility.
        
        Returns:
            Dict containing:
                - 'name': Preprocessor name
                - 'params': Configuration parameters
                - 'fitted_state': Learned parameters (if trainable)
                - 'is_fitted': Whether fit() has been called
        """
        return {
            'name': self.name,
            'params': self.get_params(),
            'is_trainable': self.is_trainable,
            'is_fitted': getattr(self, '_is_fitted', False)
        }
    
    def load_state(self, state: Dict[str, Any]) -> 'IPreprocessor':
        """
        Load preprocessor state from a dictionary.
        
        Restores the preprocessor to a previous state.
        
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
    
    def validate_input(self, data: Union[np.ndarray, 'EEGData']) -> None:
        """
        Validate input data format.
        
        Args:
            data: Input data to validate
        
        Raises:
            ValueError: If data format is invalid
        """
        if isinstance(data, np.ndarray):
            if data.ndim not in [2, 3]:
                raise ValueError(
                    f"Expected 2D (channels, samples) or 3D (trials, channels, samples) array, "
                    f"got {data.ndim}D array with shape {data.shape}"
                )
        # EEGData validation will be handled by EEGData class
    
    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        params = self.get_params()
        param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
        return f"{self.__class__.__name__}({param_str})"
    
    def __call__(self, 
                 data: Union[np.ndarray, 'EEGData'],
                 **kwargs) -> Union[np.ndarray, 'EEGData']:
        """
        Allow using preprocessor as a callable.
        
        Equivalent to calling process().
        
        Example:
            >>> filtered = bandpass(raw_data)  # Same as bandpass.process(raw_data)
        """
        return self.process(data, **kwargs)
