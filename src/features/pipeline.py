"""
Feature Extraction Pipeline
===========================

This module provides a composable pipeline for combining multiple feature extractors.

The pipeline enables:
- Combining multiple feature extractors
- Concatenating features from different extractors
- Unified fit/extract interface
- Configuration-driven pipeline creation
- State management (save/load entire pipeline)

Pipeline Modes:
--------------
1. **concatenate** (default): Features from all extractors are concatenated
   - Output: (n_trials, sum(n_features_per_extractor))

2. **sequential**: Extractors are applied in sequence (output of one feeds into next)
   - Use case: e.g., bandpass filter -> CSP
   - Output: depends on final extractor

3. **parallel**: Extractors run independently, results combined
   - Same as concatenate but makes the independence explicit

Usage Example:
    ```python
    from src.features.pipeline import FeatureExtractionPipeline
    from src.features.extractors.csp import CSPExtractor
    from src.features.extractors.band_power import BandPowerExtractor
    
    # Create pipeline
    pipeline = FeatureExtractionPipeline()
    pipeline.add_extractor(CSPExtractor(n_components=6), name='csp')
    pipeline.add_extractor(BandPowerExtractor(bands={'mu': (8, 12)}), name='band_power')
    
    # Fit and extract
    pipeline.fit(X_train, y_train)
    features = pipeline.extract(X_test)  # Shape: (n_test, 6 + 22) = (n_test, 28)
    
    # Get feature names (for interpretability)
    print(pipeline.get_feature_names())
    # ['csp_class0_comp0', ..., 'C3_mu_log', ...]
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pickle
from pathlib import Path
import logging

from src.features.base import BaseFeatureExtractor
from src.features.factory import FeatureExtractorFactory

# Configure logging
logger = logging.getLogger(__name__)


class FeatureExtractionPipeline:
    """
    Composable pipeline for feature extraction.
    
    Combines multiple feature extractors and provides a unified interface
    for fitting and extracting features.
    
    Attributes:
        mode (str): Pipeline mode ('concatenate', 'sequential', 'parallel')
        extractors (Dict[str, BaseFeatureExtractor]): Named extractors
        is_fitted (bool): Whether pipeline has been fitted
    
    Example:
        >>> pipeline = FeatureExtractionPipeline()
        >>> pipeline.add_extractor(CSPExtractor(n_components=6))
        >>> pipeline.add_extractor(BandPowerExtractor())
        >>> pipeline.fit(X_train, y_train)
        >>> features = pipeline.extract(X_test)
    """
    
    def __init__(self,
                 mode: str = 'concatenate',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature extraction pipeline.
        
        Args:
            mode: Pipeline mode
                - 'concatenate': Combine features from all extractors
                - 'sequential': Apply extractors in sequence
                - 'parallel': Same as concatenate (explicit)
            config: Optional configuration dictionary
        """
        valid_modes = ['concatenate', 'sequential', 'parallel']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
        
        self._mode = mode
        self._config = config or {}
        
        # Ordered dict of extractors
        self._extractors: Dict[str, BaseFeatureExtractor] = {}
        self._extractor_order: List[str] = []
        
        # State
        self._is_fitted = False
        self._n_features = 0
        self._feature_names: List[str] = []
        
        # Data info (set during fit)
        self._n_channels: Optional[int] = None
        self._n_samples: Optional[int] = None
        self._sampling_rate: float = 250.0
        
        logger.debug(f"FeatureExtractionPipeline created with mode={mode}")
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def mode(self) -> str:
        """Pipeline mode."""
        return self._mode
    
    @property
    def is_fitted(self) -> bool:
        """Whether pipeline has been fitted."""
        return self._is_fitted
    
    @property
    def n_features(self) -> int:
        """Total number of features produced by pipeline."""
        return self._n_features
    
    @property
    def n_extractors(self) -> int:
        """Number of extractors in pipeline."""
        return len(self._extractors)
    
    @property
    def extractor_names(self) -> List[str]:
        """Names of extractors in order."""
        return self._extractor_order.copy()
    
    # =========================================================================
    # EXTRACTOR MANAGEMENT
    # =========================================================================
    
    def add_extractor(self,
                      extractor: Union[BaseFeatureExtractor, str],
                      name: Optional[str] = None,
                      config: Optional[Dict[str, Any]] = None,
                      **kwargs) -> 'FeatureExtractionPipeline':
        """
        Add a feature extractor to the pipeline.
        
        Args:
            extractor: Feature extractor instance or name (str)
            name: Optional unique name for this extractor.
                  Defaults to extractor.name or auto-generated.
            config: Configuration for extractor initialization
            **kwargs: Additional arguments if extractor is string
        
        Returns:
            Self for method chaining
        
        Example:
            >>> pipeline.add_extractor(CSPExtractor(n_components=6), name='csp')
            >>> pipeline.add_extractor('band_power', bands={'mu': (8, 12)})
        """
        # Create extractor if string name provided
        if isinstance(extractor, str):
            extractor = FeatureExtractorFactory.create(
                extractor, config=config, **kwargs
            )
        elif config:
            extractor.initialize(config)
        
        # Generate name if not provided
        if name is None:
            base_name = extractor.name
            # Make unique
            counter = 1
            name = base_name
            while name in self._extractors:
                name = f"{base_name}_{counter}"
                counter += 1
        
        # Check for duplicate
        if name in self._extractors:
            raise ValueError(f"Extractor with name '{name}' already exists")
        
        # Add to pipeline
        self._extractors[name] = extractor
        self._extractor_order.append(name)
        
        # Reset fitted state
        self._is_fitted = False
        
        logger.debug(f"Added extractor '{name}' to pipeline")
        return self
    
    def remove_extractor(self, name: str) -> 'FeatureExtractionPipeline':
        """
        Remove an extractor from the pipeline.
        
        Args:
            name: Name of extractor to remove
        
        Returns:
            Self for method chaining
        """
        if name in self._extractors:
            del self._extractors[name]
            self._extractor_order.remove(name)
            self._is_fitted = False
            logger.debug(f"Removed extractor '{name}' from pipeline")
        
        return self
    
    def get_extractor(self, name: str) -> Optional[BaseFeatureExtractor]:
        """
        Get an extractor by name.
        
        Args:
            name: Extractor name
        
        Returns:
            Extractor instance or None if not found
        """
        return self._extractors.get(name)
    
    def clear(self) -> 'FeatureExtractionPipeline':
        """
        Remove all extractors from pipeline.
        
        Returns:
            Self for method chaining
        """
        self._extractors.clear()
        self._extractor_order.clear()
        self._is_fitted = False
        self._n_features = 0
        self._feature_names.clear()
        return self
    
    # =========================================================================
    # FIT / EXTRACT
    # =========================================================================
    
    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            **kwargs) -> 'FeatureExtractionPipeline':
        """
        Fit all extractors in the pipeline.
        
        Args:
            X: Training data, shape (n_trials, n_channels, n_samples)
            y: Class labels, shape (n_trials,)
            **kwargs: Additional arguments passed to extractors
        
        Returns:
            Self for method chaining
        """
        if len(self._extractors) == 0:
            raise ValueError("Pipeline is empty. Add extractors first.")
        
        # Validate input
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, got {X.ndim}D")
        
        # Store data info
        self._n_channels = X.shape[1]
        self._n_samples = X.shape[2]
        
        logger.info(f"Fitting pipeline with {len(self._extractors)} extractors "
                   f"on data shape {X.shape}")
        
        if self._mode in ['concatenate', 'parallel']:
            self._fit_parallel(X, y, **kwargs)
        else:  # sequential
            self._fit_sequential(X, y, **kwargs)
        
        # Compute total features and names
        self._update_feature_info()
        
        self._is_fitted = True
        logger.info(f"Pipeline fitted. Total features: {self._n_features}")
        
        return self
    
    def _fit_parallel(self,
                      X: np.ndarray,
                      y: Optional[np.ndarray],
                      **kwargs) -> None:
        """Fit extractors in parallel (independently)."""
        for name in self._extractor_order:
            extractor = self._extractors[name]
            logger.debug(f"Fitting extractor '{name}'")
            extractor.fit(X, y, **kwargs)
    
    def _fit_sequential(self,
                        X: np.ndarray,
                        y: Optional[np.ndarray],
                        **kwargs) -> None:
        """Fit extractors sequentially (output feeds to next)."""
        current_data = X
        
        for name in self._extractor_order:
            extractor = self._extractors[name]
            logger.debug(f"Fitting extractor '{name}' in sequence")
            extractor.fit(current_data, y, **kwargs)
            
            # Get output for next extractor
            current_data = extractor.extract(current_data)
            
            # Reshape for next extractor if needed
            if current_data.ndim == 2:
                # Features output, reshape to (n_trials, n_features, 1) for next
                current_data = current_data[:, :, np.newaxis]
    
    def extract(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Extract features using all extractors.
        
        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
               or (n_channels, n_samples) for single trial
            **kwargs: Additional arguments passed to extractors
        
        Returns:
            np.ndarray: Extracted features
                - concatenate/parallel: (n_trials, total_features)
                - sequential: depends on final extractor
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        # Handle single trial
        is_single_trial = False
        if X.ndim == 2:
            X = X[np.newaxis, ...]
            is_single_trial = True
        
        if self._mode in ['concatenate', 'parallel']:
            features = self._extract_parallel(X, **kwargs)
        else:  # sequential
            features = self._extract_sequential(X, **kwargs)
        
        # Handle single trial output
        if is_single_trial and features.ndim == 2:
            features = features.squeeze(0)
        
        return features
    
    def _extract_parallel(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Extract features in parallel and concatenate."""
        all_features = []
        
        for name in self._extractor_order:
            extractor = self._extractors[name]
            features = extractor.extract(X, **kwargs)
            all_features.append(features)
        
        # Concatenate along feature dimension
        return np.hstack(all_features)
    
    def _extract_sequential(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Extract features sequentially."""
        current_data = X
        
        for name in self._extractor_order:
            extractor = self._extractors[name]
            current_data = extractor.extract(current_data, **kwargs)
            
            # Reshape if needed for next extractor
            if current_data.ndim == 2:
                current_data = current_data[:, :, np.newaxis]
        
        # Return final output
        if current_data.ndim == 3:
            current_data = current_data.squeeze(-1)
        
        return current_data
    
    def fit_extract(self,
                    X: np.ndarray,
                    y: Optional[np.ndarray] = None,
                    **kwargs) -> np.ndarray:
        """
        Fit and extract in one step.
        
        Args:
            X: Training data
            y: Labels
            **kwargs: Additional arguments
        
        Returns:
            Extracted features
        """
        self.fit(X, y, **kwargs)
        return self.extract(X, **kwargs)
    
    # =========================================================================
    # FEATURE INFO
    # =========================================================================
    
    def _update_feature_info(self) -> None:
        """Update total features and feature names."""
        if self._mode in ['concatenate', 'parallel']:
            # Sum of all extractor features
            self._n_features = 0
            self._feature_names = []
            
            for name in self._extractor_order:
                extractor = self._extractors[name]
                self._n_features += extractor.n_features
                
                # Prefix feature names with extractor name if multiple
                ext_names = extractor.get_feature_names()
                if len(self._extractors) > 1:
                    ext_names = [f"{name}_{fn}" for fn in ext_names]
                self._feature_names.extend(ext_names)
        else:
            # Sequential: use last extractor's features
            if self._extractor_order:
                last_name = self._extractor_order[-1]
                last_extractor = self._extractors[last_name]
                self._n_features = last_extractor.n_features
                self._feature_names = last_extractor.get_feature_names()
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features produced by pipeline.
        
        Returns:
            List of feature names in order
        """
        if self._feature_names:
            return self._feature_names.copy()
        
        # Generate if not available
        self._update_feature_info()
        return self._feature_names.copy()
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get detailed feature information.
        
        Returns:
            Dict with feature breakdown by extractor
        """
        info = {
            'total_features': self._n_features,
            'n_extractors': len(self._extractors),
            'mode': self._mode,
            'extractors': {}
        }
        
        for name in self._extractor_order:
            extractor = self._extractors[name]
            info['extractors'][name] = {
                'type': extractor.name,
                'n_features': extractor.n_features,
                'is_trainable': extractor.is_trainable,
                'is_fitted': extractor.is_fitted
            }
        
        return info
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get pipeline state for serialization.
        
        Returns:
            Dict containing complete pipeline state
        """
        state = {
            'mode': self._mode,
            'config': self._config,
            'is_fitted': self._is_fitted,
            'n_features': self._n_features,
            'feature_names': self._feature_names,
            'n_channels': self._n_channels,
            'n_samples': self._n_samples,
            'sampling_rate': self._sampling_rate,
            'extractor_order': self._extractor_order,
            'extractors': {}
        }
        
        # Store each extractor's state
        for name, extractor in self._extractors.items():
            state['extractors'][name] = {
                'class': extractor.__class__.__name__,
                'module': extractor.__class__.__module__,
                'state': extractor.get_state()
            }
        
        return state
    
    def load_state(self, state: Dict[str, Any]) -> 'FeatureExtractionPipeline':
        """
        Load pipeline state.
        
        Args:
            state: State dictionary from get_state()
        
        Returns:
            Self for method chaining
        """
        self._mode = state['mode']
        self._config = state.get('config', {})
        self._is_fitted = state.get('is_fitted', False)
        self._n_features = state.get('n_features', 0)
        self._feature_names = state.get('feature_names', [])
        self._n_channels = state.get('n_channels')
        self._n_samples = state.get('n_samples')
        self._sampling_rate = state.get('sampling_rate', 250.0)
        self._extractor_order = state.get('extractor_order', [])
        
        # Restore extractors
        self._extractors.clear()
        for name, ext_info in state.get('extractors', {}).items():
            # Create extractor from factory
            ext_state = ext_info['state']
            extractor_type = ext_state.get('name', 'csp')
            
            extractor = FeatureExtractorFactory.create(extractor_type)
            extractor.load_state(ext_state)
            
            self._extractors[name] = extractor
        
        return self
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save pipeline to disk.
        
        Args:
            path: File path to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = self.get_state()
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureExtractionPipeline':
        """
        Load pipeline from disk.
        
        Args:
            path: File path to load from
        
        Returns:
            Loaded pipeline instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        pipeline = cls()
        pipeline.load_state(state)
        
        logger.info(f"Pipeline loaded from {path}")
        return pipeline
    
    # =========================================================================
    # CONFIGURATION-BASED CREATION
    # =========================================================================
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FeatureExtractionPipeline':
        """
        Create pipeline from configuration dictionary.
        
        Args:
            config: Configuration with structure:
                {
                    'mode': 'concatenate',
                    'extractors': [
                        {'type': 'csp', 'params': {'n_components': 6}},
                        {'type': 'band_power', 'params': {'bands': {'mu': (8, 12)}}}
                    ],
                    'init_config': {'sampling_rate': 250}
                }
        
        Returns:
            Configured pipeline
        
        Example:
            >>> config = {
            ...     'extractors': [
            ...         {'type': 'csp', 'params': {'n_components': 6}}
            ...     ]
            ... }
            >>> pipeline = FeatureExtractionPipeline.from_config(config)
        """
        mode = config.get('mode', 'concatenate')
        pipeline = cls(mode=mode)
        
        init_config = config.get('init_config', {})
        
        for ext_config in config.get('extractors', []):
            ext_type = ext_config.get('type')
            ext_params = ext_config.get('params', {})
            ext_name = ext_config.get('name')
            
            # Merge init_config
            full_config = {**init_config, **ext_config.get('config', {})}
            
            pipeline.add_extractor(
                ext_type,
                name=ext_name,
                config=full_config if full_config else None,
                **ext_params
            )
        
        return pipeline
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def summary(self) -> str:
        """
        Get pipeline summary string.
        
        Returns:
            Formatted summary
        """
        lines = [
            "Feature Extraction Pipeline",
            "=" * 40,
            f"Mode: {self._mode}",
            f"Fitted: {self._is_fitted}",
            f"Total Features: {self._n_features}",
            f"Extractors ({len(self._extractors)}):"
        ]
        
        for i, name in enumerate(self._extractor_order, 1):
            extractor = self._extractors[name]
            lines.append(
                f"  {i}. {name} ({extractor.name}): "
                f"{extractor.n_features} features, "
                f"trainable={extractor.is_trainable}"
            )
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"FeatureExtractionPipeline("
            f"mode='{self._mode}', "
            f"extractors={len(self._extractors)}, "
            f"features={self._n_features}, "
            f"{status})"
        )
    
    def __len__(self) -> int:
        """Number of extractors."""
        return len(self._extractors)
    
    def __iter__(self):
        """Iterate over extractors."""
        for name in self._extractor_order:
            yield name, self._extractors[name]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(*extractors,
                    mode: str = 'concatenate',
                    **kwargs) -> FeatureExtractionPipeline:
    """
    Convenience function to create pipeline with extractors.
    
    Args:
        *extractors: Extractor instances or (name, extractor) tuples
        mode: Pipeline mode
        **kwargs: Additional pipeline configuration
    
    Returns:
        Configured pipeline
    
    Example:
        >>> pipeline = create_pipeline(
        ...     CSPExtractor(n_components=6),
        ...     BandPowerExtractor(bands={'mu': (8, 12)})
        ... )
    """
    pipeline = FeatureExtractionPipeline(mode=mode)
    
    for ext in extractors:
        if isinstance(ext, tuple) and len(ext) == 2:
            name, extractor = ext
            pipeline.add_extractor(extractor, name=name)
        else:
            pipeline.add_extractor(ext)
    
    return pipeline


def create_motor_imagery_pipeline(
    n_csp_components: int = 6,
    sampling_rate: float = 250.0
) -> FeatureExtractionPipeline:
    """
    Create a standard pipeline for motor imagery classification.
    
    Combines CSP features with band power features for mu and beta bands.
    
    Args:
        n_csp_components: Number of CSP components
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Configured pipeline for motor imagery
    """
    config = {
        'mode': 'concatenate',
        'init_config': {'sampling_rate': sampling_rate},
        'extractors': [
            {
                'type': 'csp',
                'name': 'csp',
                'params': {'n_components': n_csp_components}
            },
            {
                'type': 'band_power',
                'name': 'band_power',
                'params': {
                    'bands': {'mu': (8, 12), 'beta': (12, 30)},
                    'average_channels': True
                }
            }
        ]
    }
    
    return FeatureExtractionPipeline.from_config(config)
