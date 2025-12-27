"""
Classifier Factory - Dynamic Classifier Creation
=================================================

This module provides a factory pattern for creating classifier instances
dynamically based on configuration. It supports both traditional ML
classifiers and deep learning models.

Design Pattern:
    Strategy + Factory pattern for flexible classifier selection and
    easy extension with new classifier types.

Supported Classifiers:
    Traditional ML:
    - 'lda': Linear Discriminant Analysis
    - 'svm': Support Vector Machine
    
    Deep Learning:
    - 'eegnet': EEGNet compact CNN
    - 'eeg_dcnet': EEG-DCNet (future)
    - 'shallow_convnet': ShallowConvNet (future)
    - 'atcnet': ATCNet (future)

Example:
    ```python
    from src.classifiers.factory import ClassifierFactory
    
    # Create LDA classifier
    lda = ClassifierFactory.create('lda', n_classes=4)
    
    # Create SVM with custom parameters
    svm = ClassifierFactory.create('svm', kernel='rbf', C=10.0, n_classes=4)
    
    # Create EEGNet
    eegnet = ClassifierFactory.create('eegnet',
        n_classes=4,
        n_channels=22,
        n_samples=1000,
        device='cuda'
    )
    
    # Create from config dictionary
    config = {'name': 'eegnet', 'n_classes': 4, 'F1': 8, 'D': 2}
    clf = ClassifierFactory.from_config(config)
    ```

Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

from typing import Dict, Any, Optional, Type, List, Union
import logging

from .base import BaseClassifier
from .models.traditional.lda import LDAClassifier
from .models.traditional.svm import SVMClassifier
from .models.deep_learning.eegnet import EEGNetClassifier


# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# CLASSIFIER REGISTRY
# =============================================================================

# Registry mapping classifier names to classes
_CLASSIFIER_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    # Traditional ML
    'lda': LDAClassifier,
    'svm': SVMClassifier,
    
    # Deep Learning
    'eegnet': EEGNetClassifier,
}

# Default configurations for each classifier
_DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'lda': {
        'solver': 'svd',
        'shrinkage': None,
        'use_scaler': True,
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'probability': True,
        'use_scaler': True,
    },
    'eegnet': {
        'F1': 8,
        'D': 2,
        'F2': 16,
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping': True,
        'patience': 10,
    },
}

# Classifier type mapping
_CLASSIFIER_TYPES: Dict[str, str] = {
    'lda': 'traditional',
    'svm': 'traditional',
    'eegnet': 'deep_learning',
}


# =============================================================================
# CLASSIFIER FACTORY
# =============================================================================

class ClassifierFactory:
    """
    Factory class for creating classifier instances.
    
    Provides a unified interface for creating any registered classifier
    type with appropriate configuration.
    
    Class Methods:
        - create: Create classifier by name with kwargs
        - from_config: Create classifier from configuration dict
        - register: Register new classifier type
        - list_classifiers: List available classifiers
        - get_default_config: Get default config for classifier
        - get_classifier_info: Get information about classifier
    """
    
    @classmethod
    def create(cls, 
               name: str, 
               n_classes: int = 4,
               **kwargs) -> BaseClassifier:
        """
        Create and initialize a classifier by name.
        
        Args:
            name: Classifier name ('lda', 'svm', 'eegnet', etc.)
            n_classes: Number of output classes
            **kwargs: Classifier-specific parameters
        
        Returns:
            Initialized classifier instance
        
        Raises:
            ValueError: If classifier name is not registered
            
        Example:
            >>> lda = ClassifierFactory.create('lda', n_classes=4)
            >>> svm = ClassifierFactory.create('svm', kernel='rbf', C=10.0)
            >>> eegnet = ClassifierFactory.create('eegnet', 
            ...     n_classes=4, n_channels=22, n_samples=1000)
        """
        name = name.lower()
        
        if name not in _CLASSIFIER_REGISTRY:
            available = ', '.join(_CLASSIFIER_REGISTRY.keys())
            raise ValueError(
                f"Unknown classifier: '{name}'. "
                f"Available classifiers: {available}"
            )
        
        # Get classifier class
        classifier_cls = _CLASSIFIER_REGISTRY[name]
        
        # Build configuration
        config = _DEFAULT_CONFIGS.get(name, {}).copy()
        config['n_classes'] = n_classes
        config.update(kwargs)
        
        # Create and initialize
        classifier = classifier_cls()
        classifier.initialize(config)
        
        logger.info(f"Created {name} classifier with config: {config}")
        
        return classifier
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseClassifier:
        """
        Create classifier from configuration dictionary.
        
        Args:
            config: Configuration dict with 'name' key and parameters
        
        Returns:
            Initialized classifier instance
            
        Example:
            >>> config = {
            ...     'name': 'eegnet',
            ...     'n_classes': 4,
            ...     'n_channels': 22,
            ...     'n_samples': 1000,
            ...     'dropout_rate': 0.5
            ... }
            >>> clf = ClassifierFactory.from_config(config)
        """
        config = config.copy()
        name = config.pop('name', config.pop('type', None))
        
        if name is None:
            raise ValueError("Config must contain 'name' or 'type' key")
        
        return cls.create(name, **config)
    
    @classmethod
    def register(cls, 
                 name: str, 
                 classifier_cls: Type[BaseClassifier],
                 default_config: Optional[Dict[str, Any]] = None,
                 classifier_type: str = 'custom') -> None:
        """
        Register a new classifier type.
        
        Args:
            name: Unique name for the classifier
            classifier_cls: Classifier class (must inherit from BaseClassifier)
            default_config: Default configuration parameters
            classifier_type: Type classification ('traditional', 'deep_learning', 'custom')
            
        Example:
            >>> class MyClassifier(BaseClassifier):
            ...     ...
            >>> ClassifierFactory.register('my_classifier', MyClassifier,
            ...     default_config={'param1': 1.0})
        """
        name = name.lower()
        
        if not issubclass(classifier_cls, BaseClassifier):
            raise TypeError(
                f"Classifier class must inherit from BaseClassifier, "
                f"got {classifier_cls}"
            )
        
        _CLASSIFIER_REGISTRY[name] = classifier_cls
        _CLASSIFIER_TYPES[name] = classifier_type
        
        if default_config is not None:
            _DEFAULT_CONFIGS[name] = default_config
        
        logger.info(f"Registered classifier: {name}")
    
    @classmethod
    def list_classifiers(cls, 
                         classifier_type: Optional[str] = None) -> List[str]:
        """
        List available classifier names.
        
        Args:
            classifier_type: Filter by type ('traditional', 'deep_learning')
                           If None, return all classifiers
        
        Returns:
            List of classifier names
            
        Example:
            >>> ClassifierFactory.list_classifiers()
            ['lda', 'svm', 'eegnet']
            >>> ClassifierFactory.list_classifiers('traditional')
            ['lda', 'svm']
        """
        if classifier_type is None:
            return list(_CLASSIFIER_REGISTRY.keys())
        
        return [
            name for name, ctype in _CLASSIFIER_TYPES.items()
            if ctype == classifier_type
        ]
    
    @classmethod
    def get_default_config(cls, name: str) -> Dict[str, Any]:
        """
        Get default configuration for a classifier.
        
        Args:
            name: Classifier name
        
        Returns:
            Default configuration dictionary
        """
        name = name.lower()
        
        if name not in _CLASSIFIER_REGISTRY:
            raise ValueError(f"Unknown classifier: {name}")
        
        return _DEFAULT_CONFIGS.get(name, {}).copy()
    
    @classmethod
    def get_classifier_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a classifier.
        
        Args:
            name: Classifier name
        
        Returns:
            Dict with classifier information
        """
        name = name.lower()
        
        if name not in _CLASSIFIER_REGISTRY:
            raise ValueError(f"Unknown classifier: {name}")
        
        classifier_cls = _CLASSIFIER_REGISTRY[name]
        
        return {
            'name': name,
            'class': classifier_cls.__name__,
            'type': _CLASSIFIER_TYPES.get(name, 'unknown'),
            'default_config': _DEFAULT_CONFIGS.get(name, {}),
            'docstring': classifier_cls.__doc__
        }
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a classifier is registered."""
        return name.lower() in _CLASSIFIER_REGISTRY


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_classifier(name: str, n_classes: int = 4, **kwargs) -> BaseClassifier:
    """
    Create a classifier by name (convenience function).
    
    Shortcut for ClassifierFactory.create().
    
    Args:
        name: Classifier name
        n_classes: Number of output classes
        **kwargs: Classifier-specific parameters
    
    Returns:
        Initialized classifier
        
    Example:
        >>> clf = create_classifier('lda', n_classes=4)
    """
    return ClassifierFactory.create(name, n_classes=n_classes, **kwargs)


def create_traditional_classifier(
    name: str = 'lda',
    n_classes: int = 4,
    **kwargs
) -> BaseClassifier:
    """
    Create a traditional ML classifier.
    
    Args:
        name: Classifier name ('lda' or 'svm')
        n_classes: Number of output classes
        **kwargs: Classifier-specific parameters
    
    Returns:
        Initialized traditional classifier
    """
    if name not in ['lda', 'svm']:
        raise ValueError(f"Traditional classifier must be 'lda' or 'svm', got '{name}'")
    
    return ClassifierFactory.create(name, n_classes=n_classes, **kwargs)


def create_deep_learning_classifier(
    name: str = 'eegnet',
    n_classes: int = 4,
    n_channels: int = 22,
    n_samples: int = 1000,
    **kwargs
) -> BaseClassifier:
    """
    Create a deep learning classifier.
    
    Args:
        name: Classifier name ('eegnet', etc.)
        n_classes: Number of output classes
        n_channels: Number of EEG channels
        n_samples: Time samples per trial
        **kwargs: Additional parameters
    
    Returns:
        Initialized deep learning classifier
    """
    return ClassifierFactory.create(
        name, 
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        **kwargs
    )


def list_available_classifiers() -> Dict[str, List[str]]:
    """
    List all available classifiers grouped by type.
    
    Returns:
        Dict mapping classifier type to list of names
    """
    return {
        'traditional': ClassifierFactory.list_classifiers('traditional'),
        'deep_learning': ClassifierFactory.list_classifiers('deep_learning')
    }


def get_classifier_for_pipeline(
    pipeline_type: str = 'csp_lda',
    n_classes: int = 4,
    **kwargs
) -> BaseClassifier:
    """
    Get a classifier configured for a specific pipeline type.
    
    Predefined pipelines:
    - 'csp_lda': CSP features + LDA classifier
    - 'csp_svm': CSP features + SVM classifier
    - 'end_to_end': Raw EEG + EEGNet
    
    Args:
        pipeline_type: Pipeline configuration name
        n_classes: Number of output classes
        **kwargs: Override parameters
    
    Returns:
        Configured classifier
    """
    pipeline_configs = {
        'csp_lda': {
            'name': 'lda',
            'solver': 'svd',
            'shrinkage': None,
        },
        'csp_svm': {
            'name': 'svm',
            'kernel': 'rbf',
            'C': 1.0,
        },
        'end_to_end': {
            'name': 'eegnet',
            'F1': 8,
            'D': 2,
            'dropout_rate': 0.5,
        },
        'csp_svm_rbf': {
            'name': 'svm',
            'kernel': 'rbf',
            'C': 10.0,
            'gamma': 0.1,
        },
        'regularized_lda': {
            'name': 'lda',
            'solver': 'lsqr',
            'shrinkage': 'auto',
        }
    }
    
    if pipeline_type not in pipeline_configs:
        available = ', '.join(pipeline_configs.keys())
        raise ValueError(
            f"Unknown pipeline type: '{pipeline_type}'. "
            f"Available: {available}"
        )
    
    config = pipeline_configs[pipeline_type].copy()
    config['n_classes'] = n_classes
    config.update(kwargs)
    
    return ClassifierFactory.from_config(config)
