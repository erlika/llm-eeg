"""
SVMClassifier - Support Vector Machine for EEG Classification
==============================================================

This module provides the Support Vector Machine classifier for 
EEG-based BCI applications. SVM with RBF kernel is a powerful
classifier that often achieves higher accuracy than LDA, especially
for complex, non-linear decision boundaries.

Key Features:
- Multiple kernel options (RBF, linear, polynomial, sigmoid)
- Hyperparameter tuning support (C, gamma)
- Class balancing for imbalanced datasets
- Decision function values for confidence estimation

Performance in BCI Competition IV-2a:
- CSP + SVM typically achieves 78-82% accuracy
- Better than LDA for non-linear feature relationships
- Slightly slower than LDA but more flexible

Example:
    ```python
    from src.classifiers.models.traditional.svm import SVMClassifier
    from src.features.extractors.csp import CSPExtractor
    
    # Extract CSP features
    csp = CSPExtractor(n_components=6)
    X_train_csp = csp.fit_extract(X_train, y_train)
    X_test_csp = csp.extract(X_test)
    
    # Train SVM classifier
    clf = SVMClassifier()
    clf.initialize({'n_classes': 4, 'kernel': 'rbf', 'C': 1.0})
    clf.fit(X_train_csp, y_train)
    predictions = clf.predict(X_test_csp)
    ```

Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import logging

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...base import BaseClassifier, register_classifier


# Setup logging
logger = logging.getLogger(__name__)


class SVMClassifier(BaseClassifier):
    """
    Support Vector Machine classifier for EEG classification.
    
    SVM finds an optimal hyperplane that maximizes the margin 
    between classes. With the RBF kernel, it can model complex
    non-linear decision boundaries.
    
    Advantages:
    - Effective for high-dimensional feature spaces
    - Robust to overfitting (with proper regularization)
    - Works well with CSP features
    - Multiple kernel options
    
    Limitations:
    - Slower training than LDA for large datasets
    - Requires feature scaling
    - Hyperparameter tuning often needed (C, gamma)
    
    Configuration Options:
        - kernel: 'rbf' (default), 'linear', 'poly', 'sigmoid'
        - C: Regularization parameter (default: 1.0)
        - gamma: Kernel coefficient (default: 'scale')
        - degree: Polynomial degree (only for poly kernel)
        - class_weight: None or 'balanced' for imbalanced data
        - probability: Enable probability estimates (default: True)
        
    Attributes:
        _model: Underlying sklearn SVC
        _scaler: StandardScaler for feature normalization
        _feature_importance: For linear kernel, coefficient magnitudes
    """
    
    def __init__(self):
        """Initialize the SVM classifier."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for SVMClassifier. "
                "Install with: pip install scikit-learn"
            )
        
        super().__init__()
        
        self._model: Optional[SVC] = None
        self._scaler: Optional[StandardScaler] = None
        
        # SVM-specific parameters
        self._kernel: str = 'rbf'
        self._C: float = 1.0
        self._gamma: Union[str, float] = 'scale'
        self._degree: int = 3
        self._class_weight: Optional[str] = None
        self._probability: bool = True
        self._use_scaler: bool = True
        
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Classifier name."""
        return 'svm'
    
    @property
    def classifier_type(self) -> str:
        """Traditional ML classifier."""
        return 'traditional'
    
    @property
    def support_vectors_(self) -> Optional[np.ndarray]:
        """
        Support vectors (data points closest to decision boundary).
        
        Returns:
            Array of support vectors or None if not fitted
        """
        if self._model is not None and hasattr(self._model, 'support_vectors_'):
            return self._model.support_vectors_
        return None
    
    @property
    def n_support_(self) -> Optional[np.ndarray]:
        """
        Number of support vectors for each class.
        
        Returns:
            Array of counts per class
        """
        if self._model is not None and hasattr(self._model, 'n_support_'):
            return self._model.n_support_
        return None
    
    @property
    def coef_(self) -> Optional[np.ndarray]:
        """
        Feature coefficients (only for linear kernel).
        
        Returns:
            Coefficient array or None if not linear kernel
        """
        if self._model is not None and self._kernel == 'linear':
            if hasattr(self._model, 'coef_'):
                return self._model.coef_
        return None
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _initialize_implementation(self, config: Dict[str, Any]) -> None:
        """
        Initialize SVM-specific parameters.
        
        Args:
            config: Configuration dictionary with keys:
                - kernel: 'rbf' (default), 'linear', 'poly', 'sigmoid'
                - C: Regularization (default: 1.0)
                - gamma: Kernel coefficient (default: 'scale')
                - degree: Polynomial degree (default: 3)
                - class_weight: None or 'balanced'
                - probability: Enable probabilities (default: True)
                - use_scaler: Standardize features (default: True)
        """
        # Extract SVM parameters
        self._kernel = config.get('kernel', 'rbf')
        self._C = config.get('C', 1.0)
        self._gamma = config.get('gamma', 'scale')
        self._degree = config.get('degree', 3)
        self._class_weight = config.get('class_weight', None)
        self._probability = config.get('probability', True)
        self._use_scaler = config.get('use_scaler', True)
        
        # Create the model
        self._model = SVC(
            kernel=self._kernel,
            C=self._C,
            gamma=self._gamma,
            degree=self._degree,
            class_weight=self._class_weight,
            probability=self._probability,
            random_state=self._random_state,
            cache_size=500  # Increase cache for faster training
        )
        
        # Create scaler
        if self._use_scaler:
            self._scaler = StandardScaler()
        else:
            self._scaler = None
        
        logger.info(
            f"Initialized SVM classifier with kernel='{self._kernel}', "
            f"C={self._C}, gamma={self._gamma}"
        )
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    def _fit_implementation(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                           **kwargs) -> None:
        """
        Train the SVM classifier.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels
            validation_data: Optional (X_val, y_val) for monitoring
            **kwargs: Additional options (ignored)
        """
        # Ensure 2D input
        X = self._ensure_2d(X)
        
        # Scale features
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        
        # Fit SVM
        logger.info(f"Training SVM with {len(X)} samples...")
        self._model.fit(X, y)
        
        # Calculate feature importance for linear kernel
        if self._kernel == 'linear' and self.coef_ is not None:
            if self.coef_.ndim == 2:
                self._feature_importance = np.mean(np.abs(self.coef_), axis=0)
            else:
                self._feature_importance = np.abs(self.coef_)
        
        # Store decision boundary info for DVA agent (Phase 4)
        self._decision_boundary_info = {
            'n_support': self.n_support_.tolist() if self.n_support_ is not None else None,
            'kernel': self._kernel,
            'C': self._C,
            'type': 'svm'
        }
        
        # Record training metrics
        train_acc = self._model.score(X, y)
        self._training_history['train_accuracy'].append(train_acc)
        self._training_history['train_loss'].append(0.0)  # No loss for SVM
        
        # Validation metrics if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = self._ensure_2d(X_val)
            if self._scaler is not None:
                X_val = self._scaler.transform(X_val)
            val_acc = self._model.score(X_val, y_val)
            self._training_history['val_accuracy'].append(val_acc)
            logger.info(f"SVM training complete. Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
        else:
            logger.info(f"SVM training complete. Train accuracy: {train_acc:.4f}")
        
        logger.info(f"Number of support vectors: {self.n_support_}")
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        X = self._ensure_2d(X)
        
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        return self._model.predict(X)
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Uses Platt scaling to convert decision function to probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        X = self._ensure_2d(X)
        
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        if self._probability:
            return self._model.predict_proba(X)
        else:
            # Fall back to decision function with softmax
            decision = self._model.decision_function(X)
            # Simple softmax approximation
            exp_decision = np.exp(decision - np.max(decision, axis=-1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=-1, keepdims=True)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.
        
        These values represent the signed distance to the hyperplane.
        Useful for Phase-4 DVA agent confidence estimation.
        
        Args:
            X: Input features
            
        Returns:
            Decision function values, shape depends on n_classes
        """
        self.validate_input(X, for_training=False)
        X = self._ensure_2d(X)
        
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        return self._model.decision_function(X)
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        state = {
            'model': self._model,
            'scaler': self._scaler,
            'kernel': self._kernel,
            'C': self._C,
            'gamma': self._gamma,
            'degree': self._degree,
            'class_weight': self._class_weight,
            'probability': self._probability,
            'use_scaler': self._use_scaler
        }
        return state
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Restore model state from serialization."""
        self._model = state.get('model')
        self._scaler = state.get('scaler')
        self._kernel = state.get('kernel', 'rbf')
        self._C = state.get('C', 1.0)
        self._gamma = state.get('gamma', 'scale')
        self._degree = state.get('degree', 3)
        self._class_weight = state.get('class_weight')
        self._probability = state.get('probability', True)
        self._use_scaler = state.get('use_scaler', True)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_support_info(self) -> Dict[str, Any]:
        """
        Get information about support vectors.
        
        Returns:
            Dict with:
            - 'n_support': Number of support vectors per class
            - 'support_ratio': Ratio of support vectors to training samples
            - 'support_indices': Indices of support vectors
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted first")
        
        result = {
            'n_support': self.n_support_.tolist() if self.n_support_ is not None else None,
            'total_support': sum(self.n_support_) if self.n_support_ is not None else None
        }
        
        if hasattr(self._model, 'support_'):
            result['support_indices'] = self._model.support_.tolist()
        
        # Support ratio
        if 'n_training_samples' in self._metadata:
            n_train = self._metadata['n_training_samples']
            result['support_ratio'] = result['total_support'] / n_train
        
        return result
    
    def get_margin_info(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get margin information for Phase-4 DVA agent.
        
        Args:
            X: Input features
            
        Returns:
            Dict with:
            - 'decision_values': Raw decision function values
            - 'margin': Minimum decision margin (confidence)
            - 'predictions': Predicted classes
        """
        self.validate_input(X, for_training=False)
        X = self._ensure_2d(X)
        
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        decision = self._model.decision_function(X)
        predictions = self._model.predict(X)
        
        # Calculate margin (distance to decision boundary)
        if decision.ndim == 1:
            margin = np.abs(decision)
        else:
            # Multi-class: use margin between top two classes
            sorted_decision = np.sort(decision, axis=1)
            margin = sorted_decision[:, -1] - sorted_decision[:, -2]
        
        return {
            'decision_values': decision,
            'margin': margin,
            'predictions': predictions
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_svm_classifier(
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: Union[str, float] = 'scale',
    class_weight: Optional[str] = None,
    n_classes: int = 4,
    random_state: Optional[int] = None
) -> SVMClassifier:
    """
    Create and initialize an SVM classifier with specified parameters.
    
    Args:
        kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
        C: Regularization parameter
        gamma: Kernel coefficient
        class_weight: None or 'balanced'
        n_classes: Number of output classes
        random_state: Random seed
        
    Returns:
        Initialized SVMClassifier
        
    Example:
        >>> clf = create_svm_classifier(kernel='rbf', C=10.0)
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
    """
    clf = SVMClassifier()
    clf.initialize({
        'kernel': kernel,
        'C': C,
        'gamma': gamma,
        'class_weight': class_weight,
        'n_classes': n_classes,
        'random_state': random_state
    })
    return clf


def create_linear_svm(
    C: float = 1.0,
    n_classes: int = 4
) -> SVMClassifier:
    """
    Create linear SVM (faster and interpretable).
    
    Linear SVM is useful when:
    - Feature space is already well-separated (e.g., after CSP)
    - Interpretability is needed (feature importance via coefficients)
    - Fast training/inference is required
    
    Args:
        C: Regularization parameter
        n_classes: Number of output classes
        
    Returns:
        Initialized linear SVMClassifier
    """
    return create_svm_classifier(
        kernel='linear',
        C=C,
        n_classes=n_classes
    )


def create_rbf_svm(
    C: float = 1.0,
    gamma: Union[str, float] = 'scale',
    n_classes: int = 4
) -> SVMClassifier:
    """
    Create RBF SVM (default for non-linear classification).
    
    RBF (Radial Basis Function) kernel is effective when:
    - Decision boundary is non-linear
    - Feature interactions are complex
    - No prior knowledge of data distribution
    
    Args:
        C: Regularization parameter (higher = less regularization)
        gamma: Kernel coefficient ('scale' or 'auto' recommended)
        n_classes: Number of output classes
        
    Returns:
        Initialized RBF SVMClassifier
    """
    return create_svm_classifier(
        kernel='rbf',
        C=C,
        gamma=gamma,
        n_classes=n_classes
    )
