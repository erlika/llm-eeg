"""
IClassifier Interface
=====================

This module defines the abstract interface for all classifiers in the EEG-BCI framework.

Classifiers are responsible for:
- Learning to map feature vectors to class labels
- Predicting class labels for new data
- Providing prediction confidence/probabilities
- Supporting model persistence (save/load)

Classifier Types Supported:
- Traditional ML: SVM, LDA, Random Forest, XGBoost
- Deep Learning: EEGNet, ShallowConvNet, DeepConvNet, EEG-DCNet
- Ensemble: Voting, Stacking, Boosting

Design Principles:
- All classifiers implement the same interface
- Supports both feature-based (traditional) and end-to-end (deep learning) approaches
- Provides both hard predictions (class labels) and soft predictions (probabilities)
- Standardized model persistence across all classifier types

Example Usage:
    ```python
    # Traditional classifier
    classifier = SVMClassifier()
    classifier.initialize({'kernel': 'rbf', 'C': 1.0})
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    
    # Deep learning classifier
    eegnet = EEGNetClassifier()
    eegnet.initialize({
        'n_classes': 4,
        'n_channels': 22,
        'n_samples': 1000,
        'learning_rate': 0.001
    })
    eegnet.fit(X_train, y_train, epochs=100)
    predictions = eegnet.predict(X_test)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from pathlib import Path

# Forward references
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.types.predictions import Prediction, PredictionBatch


class IClassifier(ABC):
    """
    Abstract interface for EEG classifiers.
    
    All classifier implementations must inherit from this class.
    This includes both traditional ML classifiers and deep learning models.
    
    Attributes:
        name (str): Unique identifier for this classifier
        n_classes (int): Number of output classes
        is_fitted (bool): Whether the classifier has been trained
        classes_ (np.ndarray): Array of class labels (set after fitting)
    
    Prediction Output:
        - predict(): Returns class labels (hard predictions)
        - predict_proba(): Returns class probabilities (soft predictions)
        - predict_with_confidence(): Returns predictions with confidence scores
    """
    
    # =========================================================================
    # ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this classifier.
        
        Returns:
            str: Classifier name (e.g., "svm", "lda", "eegnet", "eeg_dcnet")
        
        Example:
            >>> classifier.name
            'eegnet'
        """
        pass
    
    @property
    @abstractmethod
    def n_classes(self) -> int:
        """
        Number of output classes.
        
        Returns:
            int: Number of classes the classifier predicts
        
        Note:
            Set during initialization or inferred from training data.
        """
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """
        Check if classifier has been trained.
        
        Returns:
            bool: True if fit() has been called successfully
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the classifier with configuration.
        
        Sets up the classifier architecture and hyperparameters.
        Must be called before fit().
        
        Args:
            config: Dictionary containing classifier-specific settings
                Common keys:
                - 'n_classes': Number of output classes (required)
                - 'random_state': Random seed for reproducibility
                
                For traditional ML:
                - 'kernel', 'C', 'gamma' (SVM)
                - 'n_estimators', 'max_depth' (Random Forest)
                
                For deep learning:
                - 'n_channels': Number of EEG channels
                - 'n_samples': Number of time samples per trial
                - 'learning_rate': Optimizer learning rate
                - 'dropout_rate': Dropout probability
                - 'device': 'cpu' or 'cuda'
        
        Raises:
            ValueError: If required configuration is missing
            
        Example:
            >>> eegnet = EEGNetClassifier()
            >>> eegnet.initialize({
            ...     'n_classes': 4,
            ...     'n_channels': 22,
            ...     'n_samples': 1000,
            ...     'dropout_rate': 0.5
            ... })
        """
        pass
    
    @abstractmethod
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            **kwargs) -> 'IClassifier':
        """
        Train the classifier on data.
        
        Args:
            X: Training features/data
                - Traditional ML: Shape (n_samples, n_features)
                - Deep Learning: Shape (n_samples, n_channels, n_timepoints)
            y: Training labels
                Shape (n_samples,), values in range [0, n_classes-1]
            validation_data: Optional tuple of (X_val, y_val) for monitoring
            **kwargs: Additional training options
                - 'epochs': Number of training epochs (deep learning)
                - 'batch_size': Mini-batch size (deep learning)
                - 'early_stopping': Enable early stopping
                - 'patience': Early stopping patience
                - 'verbose': Verbosity level
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If data format is invalid
            RuntimeError: If classifier is not initialized
            
        Example:
            >>> classifier.fit(X_train, y_train, 
            ...               validation_data=(X_val, y_val),
            ...               epochs=100, 
            ...               batch_size=32)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Args:
            X: Input features/data
                - Traditional ML: Shape (n_samples, n_features)
                - Deep Learning: Shape (n_samples, n_channels, n_timepoints)
        
        Returns:
            np.ndarray: Predicted class labels
                Shape (n_samples,), values in range [0, n_classes-1]
        
        Raises:
            RuntimeError: If classifier is not fitted
            ValueError: If input format doesn't match training data
            
        Example:
            >>> predictions = classifier.predict(X_test)
            >>> predictions.shape
            (100,)  # 100 test samples
            >>> np.unique(predictions)
            array([0, 1, 2, 3])  # 4-class classification
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input features/data (same format as predict())
        
        Returns:
            np.ndarray: Class probability matrix
                Shape (n_samples, n_classes)
                Each row sums to 1.0
        
        Raises:
            RuntimeError: If classifier is not fitted
            
        Example:
            >>> probas = classifier.predict_proba(X_test)
            >>> probas.shape
            (100, 4)  # 100 samples, 4 classes
            >>> probas[0]
            array([0.1, 0.2, 0.6, 0.1])  # Probabilities for first sample
            >>> probas[0].sum()
            1.0
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get classifier parameters.
        
        Returns:
            Dict containing all classifier hyperparameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'IClassifier':
        """
        Set classifier parameters.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        
        Note:
            Some parameters may require re-initialization.
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained classifier to disk.
        
        Args:
            path: File path to save the model
                Extension determines format:
                - .pkl: Pickle (traditional ML)
                - .pt/.pth: PyTorch (deep learning)
                - .h5: HDF5/Keras (deep learning)
        
        Raises:
            RuntimeError: If classifier is not fitted
            IOError: If file cannot be written
            
        Example:
            >>> classifier.save("models/eegnet_subject01.pt")
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> 'IClassifier':
        """
        Load a trained classifier from disk.
        
        Args:
            path: File path to load the model from
        
        Returns:
            Self for method chaining
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
            
        Example:
            >>> classifier = EEGNetClassifier()
            >>> classifier.load("models/eegnet_subject01.pt")
            >>> predictions = classifier.predict(X_test)
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS - Override if needed
    # =========================================================================
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels with confidence scores.
        
        Default implementation uses max probability as confidence.
        Override for custom confidence estimation.
        
        Args:
            X: Input features/data
        
        Returns:
            Tuple of:
                - predictions: Class labels, shape (n_samples,)
                - confidences: Confidence scores, shape (n_samples,)
        
        Example:
            >>> preds, confs = classifier.predict_with_confidence(X_test)
            >>> preds[0], confs[0]
            (2, 0.85)  # Predicted class 2 with 85% confidence
        """
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)
        return predictions, confidences
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score on test data.
        
        Args:
            X: Test features/data
            y: True labels
        
        Returns:
            float: Accuracy (correct predictions / total predictions)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """
        Get training history (for deep learning classifiers).
        
        Returns:
            Dict containing training metrics per epoch:
                - 'loss': Training loss
                - 'accuracy': Training accuracy
                - 'val_loss': Validation loss (if available)
                - 'val_accuracy': Validation accuracy (if available)
            
            Returns None for classifiers without training history.
        """
        return getattr(self, '_training_history', None)
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get complete classifier state for serialization.
        
        Returns:
            Dict containing:
                - 'name': Classifier name
                - 'params': Hyperparameters
                - 'n_classes': Number of classes
                - 'is_fitted': Whether classifier is trained
                - 'classes_': Class labels (if fitted)
        """
        state = {
            'name': self.name,
            'params': self.get_params(),
            'n_classes': self.n_classes,
            'is_fitted': self.is_fitted
        }
        if self.is_fitted and hasattr(self, 'classes_'):
            state['classes_'] = self.classes_.tolist()
        return state
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def validate_input(self, X: np.ndarray, for_training: bool = False) -> None:
        """
        Validate input data format.
        
        Args:
            X: Input data to validate
            for_training: Whether validation is for training (fit) or inference (predict)
        
        Raises:
            ValueError: If data format is invalid
            RuntimeError: If classifier state is invalid
        """
        if X.ndim < 2:
            raise ValueError(f"Input must be at least 2D, got {X.ndim}D")
        
        if not for_training and not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction. Call fit() first.")
    
    def __repr__(self) -> str:
        """String representation of the classifier."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"n_classes={self.n_classes}, "
            f"{fitted_str})"
        )
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Allow using classifier as callable.
        
        Equivalent to predict().
        """
        return self.predict(X)
