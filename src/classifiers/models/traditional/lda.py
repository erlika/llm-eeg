"""
LDAClassifier - Linear Discriminant Analysis for EEG Classification
===================================================================

This module provides the Linear Discriminant Analysis classifier for 
EEG-based BCI applications. LDA is a classical and widely-used classifier
in motor imagery BCI systems, especially when combined with CSP features.

Key Features:
- Efficient for CSP features (works well with low-dimensional feature spaces)
- Fast training and inference
- Provides class probabilities via softmax of decision function
- Feature importance via coefficient magnitudes

Performance in BCI Competition IV-2a:
- CSP + LDA typically achieves 75-80% accuracy
- Works well for subject-dependent classification
- Simple baseline for comparison with deep learning methods

Example:
    ```python
    from src.classifiers.models.traditional.lda import LDAClassifier
    from src.features.extractors.csp import CSPExtractor
    
    # Extract CSP features
    csp = CSPExtractor(n_components=6)
    X_train_csp = csp.fit_extract(X_train, y_train)
    X_test_csp = csp.extract(X_test)
    
    # Train LDA classifier
    clf = LDAClassifier()
    clf.initialize({'n_classes': 4})
    clf.fit(X_train_csp, y_train)
    predictions = clf.predict(X_test_csp)
    ```

Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...base import BaseClassifier, register_classifier


# Setup logging
logger = logging.getLogger(__name__)


class LDAClassifier(BaseClassifier):
    """
    Linear Discriminant Analysis classifier for EEG classification.
    
    LDA finds a linear combination of features that best separates 
    multiple classes. It's particularly effective for BCI applications
    when combined with CSP features.
    
    Advantages:
    - Simple and fast (no hyperparameters to tune for basic use)
    - Works well with CSP features
    - Provides interpretable coefficients
    - Good for small training sets
    
    Limitations:
    - Assumes Gaussian distribution with equal covariance
    - May underperform with very high-dimensional features
    - Not suitable for end-to-end learning
    
    Configuration Options:
        - solver: 'svd' (default), 'lsqr', 'eigen'
        - shrinkage: None, 'auto', or float (0-1)
        - n_components: Number of components for dimensionality reduction
        - priors: Class prior probabilities
        
    Attributes:
        _model: Underlying sklearn LinearDiscriminantAnalysis
        _scaler: Optional StandardScaler for feature normalization
        _feature_importance: Coefficient magnitudes
    """
    
    def __init__(self):
        """Initialize the LDA classifier."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for LDAClassifier. "
                "Install with: pip install scikit-learn"
            )
        
        super().__init__()
        
        self._model: Optional[LinearDiscriminantAnalysis] = None
        self._scaler: Optional[StandardScaler] = None
        
        # LDA-specific parameters
        self._solver: str = 'svd'
        self._shrinkage: Optional[str] = None
        self._n_components: Optional[int] = None
        self._use_scaler: bool = True
        self._priors: Optional[np.ndarray] = None
        
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Classifier name."""
        return 'lda'
    
    @property
    def classifier_type(self) -> str:
        """Traditional ML classifier."""
        return 'traditional'
    
    @property
    def coef_(self) -> Optional[np.ndarray]:
        """
        LDA coefficients (weight vector).
        
        Returns:
            Array of shape (n_classes-1, n_features) for binary/multi-class
            or (n_features,) for 2-class classification
        """
        if self._model is not None and hasattr(self._model, 'coef_'):
            return self._model.coef_
        return None
    
    @property
    def class_means_(self) -> Optional[np.ndarray]:
        """
        Class means (centroids in feature space).
        
        Returns:
            Array of shape (n_classes, n_features)
        """
        if self._model is not None and hasattr(self._model, 'means_'):
            return self._model.means_
        return None
    
    @property
    def scalings_(self) -> Optional[np.ndarray]:
        """
        LDA scaling vectors (eigen vectors).
        
        Returns:
            Array of shape (n_features, min(n_classes-1, n_features))
        """
        if self._model is not None and hasattr(self._model, 'scalings_'):
            return self._model.scalings_
        return None
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _initialize_implementation(self, config: Dict[str, Any]) -> None:
        """
        Initialize LDA-specific parameters.
        
        Args:
            config: Configuration dictionary with keys:
                - solver: 'svd' (default), 'lsqr', 'eigen'
                - shrinkage: None, 'auto', or float (0-1)
                - n_components: Number of components (default: None = all)
                - use_scaler: Whether to standardize features (default: True)
                - priors: Class prior probabilities (default: None = use data)
        """
        # Extract LDA parameters
        self._solver = config.get('solver', 'svd')
        self._shrinkage = config.get('shrinkage', None)
        self._n_components = config.get('n_components', None)
        self._use_scaler = config.get('use_scaler', True)
        self._priors = config.get('priors', None)
        
        # Validate solver-shrinkage combination
        if self._shrinkage is not None and self._solver not in ['lsqr', 'eigen']:
            logger.warning(
                f"Shrinkage requires solver='lsqr' or 'eigen', "
                f"got solver='{self._solver}'. Setting solver='lsqr'."
            )
            self._solver = 'lsqr'
        
        # Create the model
        self._model = LinearDiscriminantAnalysis(
            solver=self._solver,
            shrinkage=self._shrinkage,
            n_components=self._n_components,
            priors=self._priors
        )
        
        # Create scaler if requested
        if self._use_scaler:
            self._scaler = StandardScaler()
        else:
            self._scaler = None
        
        logger.info(
            f"Initialized LDA classifier with solver='{self._solver}', "
            f"shrinkage={self._shrinkage}"
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
        Train the LDA classifier.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels
            validation_data: Ignored for LDA (no validation needed)
            **kwargs: Additional options (ignored)
        """
        # Ensure 2D input
        X = self._ensure_2d(X)
        
        # Scale features if requested
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        
        # Fit LDA
        self._model.fit(X, y)
        
        # Calculate feature importance (coefficient magnitudes)
        if self.coef_ is not None:
            # For multi-class, average across all pairs
            if self.coef_.ndim == 2:
                self._feature_importance = np.mean(np.abs(self.coef_), axis=0)
            else:
                self._feature_importance = np.abs(self.coef_)
        
        # Store decision boundary info for DVA agent (Phase 4)
        self._decision_boundary_info = {
            'class_means': self.class_means_.tolist() if self.class_means_ is not None else None,
            'type': 'lda'
        }
        
        # Record training metrics
        train_acc = self._model.score(X, y)
        self._training_history['train_accuracy'].append(train_acc)
        self._training_history['train_loss'].append(0.0)  # No loss for LDA
        
        logger.info(f"LDA training complete. Train accuracy: {train_acc:.4f}")
    
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
        
        Uses softmax of decision function values.
        
        Args:
            X: Input features
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        X = self._ensure_2d(X)
        
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        return self._model.predict_proba(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.
        
        These values represent the signed distance to the hyperplane.
        Useful for Phase-4 DVA agent confidence estimation.
        
        Args:
            X: Input features
            
        Returns:
            Decision function values, shape (n_samples,) for 2-class
            or (n_samples, n_classes) for multi-class
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
            'solver': self._solver,
            'shrinkage': self._shrinkage,
            'n_components': self._n_components,
            'use_scaler': self._use_scaler,
            'priors': self._priors.tolist() if self._priors is not None else None
        }
        return state
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Restore model state from serialization."""
        self._model = state.get('model')
        self._scaler = state.get('scaler')
        self._solver = state.get('solver', 'svd')
        self._shrinkage = state.get('shrinkage')
        self._n_components = state.get('n_components')
        self._use_scaler = state.get('use_scaler', True)
        
        priors = state.get('priors')
        self._priors = np.array(priors) if priors is not None else None
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto LDA components (dimensionality reduction).
        
        Args:
            X: Input features
            
        Returns:
            Transformed features, shape (n_samples, n_components)
        """
        self.validate_input(X, for_training=False)
        X = self._ensure_2d(X)
        
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        return self._model.transform(X)
    
    def get_discriminant_components(self) -> Dict[str, np.ndarray]:
        """
        Get LDA discriminant components for visualization.
        
        Returns:
            Dict with:
            - 'scalings': Projection vectors
            - 'class_means': Class centroids
            - 'explained_variance_ratio': Variance explained by each component
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted first")
        
        result = {}
        
        if self.scalings_ is not None:
            result['scalings'] = self.scalings_
        
        if self.class_means_ is not None:
            result['class_means'] = self.class_means_
        
        if hasattr(self._model, 'explained_variance_ratio_'):
            result['explained_variance_ratio'] = self._model.explained_variance_ratio_
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_lda_classifier(
    solver: str = 'svd',
    shrinkage: Optional[str] = None,
    n_components: Optional[int] = None,
    use_scaler: bool = True,
    n_classes: int = 4,
    random_state: Optional[int] = None
) -> LDAClassifier:
    """
    Create and initialize an LDA classifier with specified parameters.
    
    Args:
        solver: LDA solver ('svd', 'lsqr', 'eigen')
        shrinkage: Regularization (None, 'auto', or float)
        n_components: Number of components for reduction
        use_scaler: Whether to standardize features
        n_classes: Number of output classes
        random_state: Random seed for reproducibility
        
    Returns:
        Initialized LDAClassifier
        
    Example:
        >>> clf = create_lda_classifier(solver='lsqr', shrinkage='auto')
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
    """
    clf = LDAClassifier()
    clf.initialize({
        'solver': solver,
        'shrinkage': shrinkage,
        'n_components': n_components,
        'use_scaler': use_scaler,
        'n_classes': n_classes,
        'random_state': random_state
    })
    return clf


def create_regularized_lda(
    shrinkage: str = 'auto',
    n_classes: int = 4
) -> LDAClassifier:
    """
    Create LDA with automatic regularization (shrinkage).
    
    Regularized LDA is more robust when the number of features
    approaches or exceeds the number of samples.
    
    Args:
        shrinkage: Regularization parameter ('auto' recommended)
        n_classes: Number of output classes
        
    Returns:
        Initialized regularized LDAClassifier
    """
    return create_lda_classifier(
        solver='lsqr',
        shrinkage=shrinkage,
        n_classes=n_classes
    )
