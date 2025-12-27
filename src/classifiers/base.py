"""
BaseClassifier - Abstract Base Class for EEG Classifiers
=========================================================

This module provides the abstract base class for all EEG classifiers
in the BCI framework. It implements the IClassifier interface and provides
common functionality shared by both traditional ML and deep learning classifiers.

Architecture:
- BaseClassifier (abstract) → IClassifier interface
    ├── BaseTraditionalClassifier → LDA, SVM, RandomForest
    └── BaseDeepClassifier → EEGNet, EEG-DCNet, ShallowConvNet

Design Principles:
- Common initialization and validation logic
- Standardized state management and persistence
- Hooks for subclass-specific behavior
- Full compatibility with Phase-4 Agent integration

Example:
    ```python
    class LDAClassifier(BaseClassifier):
        def __init__(self):
            super().__init__()
            
        @property
        def name(self) -> str:
            return "lda"
        
        def _fit_implementation(self, X, y, **kwargs):
            self._model = LinearDiscriminantAnalysis()
            self._model.fit(X, y)
    ```

Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import numpy as np
import pickle
import json
import logging

from ..core.interfaces.i_classifier import IClassifier
from ..core.registry import ComponentRegistry


# Setup logging
logger = logging.getLogger(__name__)


class BaseClassifier(IClassifier):
    """
    Abstract base class for all EEG classifiers.
    
    This class provides:
    - Common initialization and validation
    - State management (save/load)
    - Training history tracking
    - Integration hooks for Phase-4 agents
    
    Subclasses must implement:
    - name (property): Unique identifier
    - _fit_implementation(): Actual training logic
    - _predict_implementation(): Actual prediction logic
    - _predict_proba_implementation(): Probability prediction logic
    
    Attributes:
        _config (Dict): Configuration parameters
        _is_fitted (bool): Training status
        _n_classes (int): Number of output classes
        _classes_ (np.ndarray): Unique class labels
        _training_history (Dict): Training metrics per epoch
        _metadata (Dict): Additional metadata for tracking
    """
    
    def __init__(self):
        """Initialize the base classifier."""
        self._config: Dict[str, Any] = {}
        self._is_fitted: bool = False
        self._n_classes: int = 0
        self._classes_: Optional[np.ndarray] = None
        self._training_history: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Any] = {}
        self._random_state: Optional[int] = None
        
        # For Phase-4 integration
        self._feature_importance: Optional[np.ndarray] = None
        self._decision_boundary_info: Optional[Dict] = None
        
    # =========================================================================
    # ABSTRACT PROPERTIES - Must be implemented by subclasses
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this classifier type."""
        pass
    
    @property
    @abstractmethod
    def classifier_type(self) -> str:
        """
        Type of classifier: 'traditional' or 'deep_learning'.
        
        Used for:
        - Selecting appropriate save/load format
        - Input validation (features vs raw EEG)
        - Training parameter defaults
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def _fit_implementation(self, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                           **kwargs) -> None:
        """
        Actual training implementation.
        
        Args:
            X: Training data (already validated)
            y: Training labels
            validation_data: Optional validation set
            **kwargs: Classifier-specific training options
        """
        pass
    
    @abstractmethod
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Actual prediction implementation.
        
        Args:
            X: Input data (already validated)
            
        Returns:
            Predicted class labels
        """
        pass
    
    @abstractmethod
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Actual probability prediction implementation.
        
        Args:
            X: Input data (already validated)
            
        Returns:
            Class probability matrix (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """
        Get model-specific state for serialization.
        
        Returns:
            Dict containing model weights/parameters
        """
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """
        Restore model-specific state from serialization.
        
        Args:
            state: Previously saved model state
        """
        pass
    
    # =========================================================================
    # IMPLEMENTED PROPERTIES
    # =========================================================================
    
    @property
    def n_classes(self) -> int:
        """Number of output classes."""
        return self._n_classes
    
    @property
    def is_fitted(self) -> bool:
        """Whether classifier has been trained."""
        return self._is_fitted
    
    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Array of class labels (available after fitting)."""
        return self._classes_
    
    @property
    def config(self) -> Dict[str, Any]:
        """Current configuration parameters."""
        return self._config.copy()
    
    # =========================================================================
    # CORE API IMPLEMENTATION
    # =========================================================================
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize classifier with configuration.
        
        Args:
            config: Configuration dictionary
                Required keys vary by classifier type.
                Common keys:
                - 'n_classes': Number of output classes
                - 'random_state': Random seed for reproducibility
        
        Raises:
            ValueError: If required configuration is missing
        """
        logger.info(f"Initializing {self.name} classifier with config: {config}")
        
        # Store configuration
        self._config = config.copy()
        
        # Extract common parameters
        self._n_classes = config.get('n_classes', 0)
        self._random_state = config.get('random_state', None)
        
        # Set random state for reproducibility
        if self._random_state is not None:
            np.random.seed(self._random_state)
        
        # Call subclass-specific initialization
        self._initialize_implementation(config)
        
        logger.info(f"{self.name} classifier initialized successfully")
    
    def _initialize_implementation(self, config: Dict[str, Any]) -> None:
        """
        Subclass-specific initialization. Override if needed.
        
        Args:
            config: Configuration dictionary
        """
        pass
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            **kwargs) -> 'BaseClassifier':
        """
        Train the classifier.
        
        Args:
            X: Training features/data
            y: Training labels (0-indexed)
            validation_data: Optional (X_val, y_val) tuple
            **kwargs: Additional training options
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If data format is invalid
        """
        logger.info(f"Fitting {self.name} classifier on data with shape {X.shape}")
        
        # Validate input
        self._validate_training_data(X, y)
        
        # Store class information
        self._classes_ = np.unique(y)
        if self._n_classes == 0:
            self._n_classes = len(self._classes_)
        
        # Reset training history
        self._training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Store metadata
        self._metadata['n_training_samples'] = len(y)
        self._metadata['training_shape'] = X.shape
        self._metadata['class_distribution'] = {
            int(c): int(np.sum(y == c)) for c in self._classes_
        }
        
        # Call implementation
        self._fit_implementation(X, y, validation_data, **kwargs)
        
        # Mark as fitted
        self._is_fitted = True
        
        logger.info(f"{self.name} classifier fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input data
        
        Returns:
            Predicted class labels
        
        Raises:
            RuntimeError: If classifier not fitted
        """
        # Validate
        self.validate_input(X, for_training=False)
        X = self._ensure_2d(X)
        
        # Predict
        predictions = self._predict_implementation(X)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input data
        
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        # Validate
        self.validate_input(X, for_training=False)
        X = self._ensure_2d(X)
        
        # Predict probabilities
        probas = self._predict_proba_implementation(X)
        
        # Ensure proper shape
        if probas.ndim == 1:
            # Binary classification - expand to 2 columns
            probas = np.column_stack([1 - probas, probas])
        
        return probas
    
    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        return self._config.copy()
    
    def set_params(self, **params) -> 'BaseClassifier':
        """Set classifier parameters."""
        self._config.update(params)
        
        # Update specific attributes
        if 'n_classes' in params:
            self._n_classes = params['n_classes']
        if 'random_state' in params:
            self._random_state = params['random_state']
            if self._random_state is not None:
                np.random.seed(self._random_state)
        
        return self
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save classifier to disk.
        
        Args:
            path: File path for saving
        
        Raises:
            RuntimeError: If not fitted
            IOError: If save fails
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted classifier")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect full state
        state = {
            'name': self.name,
            'classifier_type': self.classifier_type,
            'config': self._config,
            'n_classes': self._n_classes,
            'classes_': self._classes_.tolist() if self._classes_ is not None else None,
            'is_fitted': self._is_fitted,
            'training_history': self._training_history,
            'metadata': self._metadata,
            'model_state': self._get_model_state()
        }
        
        # Save based on classifier type
        if self.classifier_type == 'traditional':
            with open(path, 'wb') as f:
                pickle.dump(state, f)
        else:
            # Deep learning - save as separate files
            self._save_deep_learning_model(path, state)
        
        logger.info(f"Saved {self.name} classifier to {path}")
    
    def load(self, path: Union[str, Path]) -> 'BaseClassifier':
        """
        Load classifier from disk.
        
        Args:
            path: File path to load from
        
        Returns:
            Self for method chaining
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load based on classifier type
        if self.classifier_type == 'traditional':
            with open(path, 'rb') as f:
                state = pickle.load(f)
        else:
            state = self._load_deep_learning_model(path)
        
        # Restore state
        self._config = state['config']
        self._n_classes = state['n_classes']
        self._classes_ = np.array(state['classes_']) if state['classes_'] else None
        self._is_fitted = state['is_fitted']
        self._training_history = state.get('training_history', {})
        self._metadata = state.get('metadata', {})
        
        # Restore model-specific state
        self._set_model_state(state['model_state'])
        
        logger.info(f"Loaded {self.name} classifier from {path}")
        return self
    
    def _save_deep_learning_model(self, path: Path, state: Dict) -> None:
        """Save deep learning model. Override in BaseDeepClassifier."""
        # Default: use pickle
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def _load_deep_learning_model(self, path: Path) -> Dict:
        """Load deep learning model. Override in BaseDeepClassifier."""
        # Default: use pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    # =========================================================================
    # EXTENDED FUNCTIONALITY
    # =========================================================================
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """Get training history (loss, accuracy per epoch)."""
        if not self._training_history:
            return None
        return self._training_history.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get classifier metadata."""
        return self._metadata.copy()
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores (if available).
        
        Returns:
            Feature importance array or None if not available
            
        Note:
            Available for:
            - LDA: Coefficient magnitudes
            - SVM: Coefficient magnitudes (linear kernel)
            - RandomForest: Gini importance
        """
        return self._feature_importance
    
    def get_decision_info(self) -> Optional[Dict[str, Any]]:
        """
        Get decision boundary information for DVA agent (Phase 4).
        
        Returns:
            Dict containing:
            - 'margin': Decision margin for SVM
            - 'class_means': Class means for LDA
            - 'decision_function_values': Raw decision values
            
        Note:
            Used by Decision Validation Agent for confidence estimation.
        """
        return self._decision_boundary_info
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete classifier state for serialization."""
        state = {
            'name': self.name,
            'classifier_type': self.classifier_type,
            'params': self.get_params(),
            'n_classes': self.n_classes,
            'is_fitted': self.is_fitted,
            'training_history': self._training_history,
            'metadata': self._metadata
        }
        
        if self._classes_ is not None:
            state['classes_'] = self._classes_.tolist()
        
        return state
    
    def summary(self) -> str:
        """
        Get human-readable summary of classifier.
        
        Returns:
            Formatted string with classifier information
        """
        lines = [
            f"{'='*50}",
            f"{self.name.upper()} Classifier Summary",
            f"{'='*50}",
            f"Type: {self.classifier_type}",
            f"Fitted: {self.is_fitted}",
            f"Number of classes: {self.n_classes}",
        ]
        
        if self._classes_ is not None:
            lines.append(f"Classes: {self._classes_.tolist()}")
        
        if self._metadata:
            lines.append(f"\nTraining Info:")
            for key, value in self._metadata.items():
                lines.append(f"  {key}: {value}")
        
        if self._training_history and self._training_history.get('train_accuracy'):
            final_acc = self._training_history['train_accuracy'][-1]
            lines.append(f"\nFinal Training Accuracy: {final_acc:.4f}")
            
            if self._training_history.get('val_accuracy'):
                final_val_acc = self._training_history['val_accuracy'][-1]
                lines.append(f"Final Validation Accuracy: {final_val_acc:.4f}")
        
        lines.append(f"{'='*50}")
        
        return '\n'.join(lines)
    
    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================
    
    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate training data format.
        
        Args:
            X: Training features
            y: Training labels
            
        Raises:
            ValueError: If data format is invalid
        """
        # Check dimensions
        if X.ndim < 2:
            raise ValueError(f"X must be at least 2D, got shape {X.shape}")
        
        # Check sample count match
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {len(X)}, y: {len(y)}"
            )
        
        # Check for NaN/Inf
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("X contains Inf values")
        
        # Check labels
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Need at least 2 classes for classification, "
                f"got {len(unique_labels)}"
            )
        
        # Check label format (0-indexed)
        if unique_labels.min() < 0:
            raise ValueError("Labels must be non-negative integers")
    
    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        """
        Ensure input is 2D for traditional classifiers.
        
        Args:
            X: Input array
            
        Returns:
            2D array
        """
        if X.ndim == 1:
            return X.reshape(1, -1)
        elif X.ndim == 3:
            # For deep learning input - flatten trials
            # (n_trials, n_channels, n_samples) -> (n_trials, n_channels * n_samples)
            return X.reshape(X.shape[0], -1)
        return X
    
    # =========================================================================
    # MAGIC METHODS
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"type='{self.classifier_type}', "
            f"n_classes={self.n_classes}, "
            f"{fitted_str})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        return self.summary()


# =============================================================================
# REGISTRY INTEGRATION
# =============================================================================

def register_classifier(cls):
    """
    Decorator to register a classifier class with the component registry.
    
    Usage:
        @register_classifier
        class MyClassifier(BaseClassifier):
            ...
    """
    registry = ComponentRegistry()
    
    # Create instance to get name
    try:
        instance = cls()
        name = instance.name
    except:
        # If can't instantiate, use class name
        name = cls.__name__.lower().replace('classifier', '')
    
    registry.register('classifier', name, cls, metadata={'type': 'classifier'})
    
    return cls
