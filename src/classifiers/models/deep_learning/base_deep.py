"""
BaseDeepClassifier - Base Class for Deep Learning EEG Classifiers
=================================================================

This module provides the abstract base class for PyTorch-based deep learning
classifiers in the EEG-BCI framework. It handles common training loops, device
management, and model persistence.

Architecture:
    BaseDeepClassifier (abstract) → BaseClassifier → IClassifier
        ├── EEGNetClassifier
        ├── EEGDCNetClassifier (future)
        ├── ShallowConvNetClassifier (future)
        └── ATCNetClassifier (future)

Features:
- PyTorch model management
- GPU/CPU device handling
- Training loop with early stopping
- Learning rate scheduling
- Training history tracking
- Model checkpointing
- Gradient clipping

Example:
    ```python
    class EEGNetClassifier(BaseDeepClassifier):
        def _build_model(self) -> nn.Module:
            return EEGNet(
                n_classes=self.n_classes,
                n_channels=self._n_channels,
                n_samples=self._n_samples
            )
    ```

Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING
from pathlib import Path
import numpy as np
import logging

# PyTorch imports - with graceful degradation
TORCH_AVAILABLE = False
torch = None
nn = None
optim = None
DataLoader = None
TensorDataset = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    pass  # PyTorch not available - will raise error at runtime if used

from ...base import BaseClassifier, register_classifier


# Setup logging
logger = logging.getLogger(__name__)


class BaseDeepClassifier(BaseClassifier):
    """
    Abstract base class for PyTorch-based deep learning classifiers.
    
    This class provides:
    - Automatic device selection (GPU/CPU)
    - Standard training loop with configurable parameters
    - Early stopping and learning rate scheduling
    - Model checkpointing and state management
    - Training history tracking for visualization
    
    Subclasses must implement:
    - name (property): Unique identifier (e.g., "eegnet")
    - _build_model(): Create and return the PyTorch model
    
    Default Configuration:
    - learning_rate: 0.001
    - batch_size: 32
    - epochs: 100
    - dropout_rate: 0.5
    - optimizer: Adam
    - loss: CrossEntropyLoss
    - early_stopping: True
    - patience: 10
    
    Attributes:
        _model (nn.Module): PyTorch model
        _optimizer: PyTorch optimizer
        _criterion: Loss function
        _device (torch.device): Computation device (cuda/cpu)
        _n_channels (int): Number of EEG channels
        _n_samples (int): Number of time samples per trial
    """
    
    def __init__(self):
        """Initialize the deep learning classifier."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for deep learning classifiers. "
                "Install with: pip install torch"
            )
        
        super().__init__()
        
        # PyTorch components
        self._model: Optional[Any] = None  # nn.Module
        self._optimizer: Optional[optim.Optimizer] = None
        self._criterion: Optional[Any] = None  # nn.Module
        self._scheduler: Optional[Any] = None
        
        # Device
        self._device = None  # Will be set during initialization
        
        # Architecture parameters
        self._n_channels: int = 22  # BCI Competition IV-2a default
        self._n_samples: int = 1000  # 4 seconds at 250 Hz
        
        # Training parameters (defaults)
        self._learning_rate: float = 0.001
        self._batch_size: int = 32
        self._epochs: int = 100
        self._dropout_rate: float = 0.5
        self._weight_decay: float = 0.0001
        
        # Early stopping
        self._early_stopping: bool = True
        self._patience: int = 10
        self._best_val_loss: float = float('inf')
        self._patience_counter: int = 0
        self._best_model_state: Optional[Dict] = None
        
        # Gradient clipping
        self._gradient_clip: Optional[float] = 1.0
        
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def classifier_type(self) -> str:
        """Deep learning classifier type."""
        return 'deep_learning'
    
    @property
    def model(self) -> Optional[Any]:  # nn.Module
        """Access the underlying PyTorch model."""
        return self._model
    
    @property
    def device(self):
        """Current computation device."""
        return self._device
    
    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return self._n_channels
    
    @property
    def n_samples(self) -> int:
        """Number of time samples per trial."""
        return self._n_samples
    
    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
    def _build_model(self):  # Returns nn.Module
        """
        Build and return the PyTorch model.
        
        This method is called during initialization after configuration
        parameters are set. The model should be returned without moving
        to device (handled automatically).
        
        Returns:
            nn.Module: The constructed PyTorch model
            
        Example:
            def _build_model(self):  # Returns nn.Module
                return EEGNet(
                    n_classes=self.n_classes,
                    n_channels=self._n_channels,
                    n_samples=self._n_samples,
                    dropout_rate=self._dropout_rate
                )
        """
        pass
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _initialize_implementation(self, config: Dict[str, Any]) -> None:
        """
        Initialize deep learning specific components.
        
        Args:
            config: Configuration dictionary with keys:
                - n_channels: Number of EEG channels (default: 22)
                - n_samples: Time samples per trial (default: 1000)
                - learning_rate: Optimizer LR (default: 0.001)
                - batch_size: Training batch size (default: 32)
                - epochs: Maximum training epochs (default: 100)
                - dropout_rate: Dropout probability (default: 0.5)
                - device: 'cuda', 'cpu', or 'auto' (default: 'auto')
                - early_stopping: Enable early stopping (default: True)
                - patience: Early stopping patience (default: 10)
                - weight_decay: L2 regularization (default: 0.0001)
                - gradient_clip: Max gradient norm (default: 1.0)
        """
        # Architecture parameters
        self._n_channels = config.get('n_channels', 22)
        self._n_samples = config.get('n_samples', 1000)
        
        # Training parameters
        self._learning_rate = config.get('learning_rate', 0.001)
        self._batch_size = config.get('batch_size', 32)
        self._epochs = config.get('epochs', 100)
        self._dropout_rate = config.get('dropout_rate', 0.5)
        self._weight_decay = config.get('weight_decay', 0.0001)
        
        # Early stopping
        self._early_stopping = config.get('early_stopping', True)
        self._patience = config.get('patience', 10)
        
        # Gradient clipping
        self._gradient_clip = config.get('gradient_clip', 1.0)
        
        # Setup device
        device_str = config.get('device', 'auto')
        self._setup_device(device_str)
        
        # Build model
        self._model = self._build_model()
        self._model = self._model.to(self._device)
        
        # Setup loss function
        self._criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self._setup_optimizer(config)
        
        # Setup scheduler (optional)
        self._setup_scheduler(config)
        
        logger.info(
            f"Initialized {self.name} on {self._device} "
            f"(channels={self._n_channels}, samples={self._n_samples})"
        )
    
    def _setup_device(self, device_str: str) -> None:
        """
        Setup computation device.
        
        Args:
            device_str: 'cuda', 'cpu', or 'auto'
        """
        if device_str == 'auto':
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device_str)
        
        logger.info(f"Using device: {self._device}")
    
    def _setup_optimizer(self, config: Dict[str, Any]) -> None:
        """
        Setup optimizer.
        
        Args:
            config: Configuration with optimizer settings
        """
        optimizer_name = config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            self._optimizer = optim.Adam(
                self._model.parameters(),
                lr=self._learning_rate,
                weight_decay=self._weight_decay
            )
        elif optimizer_name == 'adamw':
            self._optimizer = optim.AdamW(
                self._model.parameters(),
                lr=self._learning_rate,
                weight_decay=self._weight_decay
            )
        elif optimizer_name == 'sgd':
            self._optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._learning_rate,
                momentum=config.get('momentum', 0.9),
                weight_decay=self._weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self, config: Dict[str, Any]) -> None:
        """
        Setup learning rate scheduler (optional).
        
        Args:
            config: Configuration with scheduler settings
        """
        scheduler_name = config.get('scheduler', None)
        
        if scheduler_name is None:
            self._scheduler = None
        elif scheduler_name == 'step':
            self._scheduler = optim.lr_scheduler.StepLR(
                self._optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self._epochs
            )
        elif scheduler_name == 'reduce_on_plateau':
            self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                mode='min',
                factor=config.get('factor', 0.5),
                patience=config.get('scheduler_patience', 5)
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
        Train the deep learning model.
        
        Args:
            X: Training data (n_samples, n_channels, n_timepoints)
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
            **kwargs: Additional options
                - epochs: Override default epochs
                - batch_size: Override default batch size
                - verbose: Verbosity level (0, 1, or 2)
        """
        # Override training parameters if provided
        epochs = kwargs.get('epochs', self._epochs)
        batch_size = kwargs.get('batch_size', self._batch_size)
        verbose = kwargs.get('verbose', 1)
        
        # Prepare data
        X_tensor = self._prepare_input(X)
        y_tensor = torch.LongTensor(y)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        # Prepare validation data
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_tensor = self._prepare_input(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Reset early stopping
        self._best_val_loss = float('inf')
        self._patience_counter = 0
        self._best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            
            self._training_history['train_loss'].append(train_loss)
            self._training_history['train_accuracy'].append(train_acc)
            
            # Validation
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
                self._training_history['val_loss'].append(val_loss)
                self._training_history['val_accuracy'].append(val_acc)
            
            # Learning rate scheduling
            if self._scheduler is not None:
                if isinstance(self._scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self._scheduler.step()
            
            # Logging
            if verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                logger.info(msg)
            
            # Early stopping
            if self._early_stopping and val_loader is not None:
                if self._check_early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model if early stopping was used
        if self._best_model_state is not None:
            self._model.load_state_dict(self._best_model_state)
            logger.info("Restored best model from early stopping checkpoint")
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self._model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self._device)
            batch_y = batch_y.to(self._device)
            
            # Forward pass
            self._optimizer.zero_grad()
            outputs = self._model(batch_X)
            loss = self._criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self._gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), 
                    self._gradient_clip
                )
            
            self._optimizer.step()
            
            # Statistics
            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)
                
                outputs = self._model(batch_X)
                loss = self._criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check early stopping condition.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if should stop training
        """
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            # Save best model state
            self._best_model_state = self._model.state_dict().copy()
            return False
        else:
            self._patience_counter += 1
            return self._patience_counter >= self._patience
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input data
            
        Returns:
            Predicted labels
        """
        self._model.eval()
        X_tensor = self._prepare_input(X)
        
        with torch.no_grad():
            X_tensor = X_tensor.to(self._device)
            outputs = self._model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def _predict_proba_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        self._model.eval()
        X_tensor = self._prepare_input(X)
        
        with torch.no_grad():
            X_tensor = X_tensor.to(self._device)
            outputs = self._model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
        
        return probas.cpu().numpy()
    
    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    
    def _prepare_input(self, X: np.ndarray):
        """
        Prepare input data for the model.
        
        Handles shape conversion:
        - (n_samples, n_channels, n_timepoints) → as is
        - (n_samples, n_features) → reshape to (n_samples, n_channels, n_timepoints)
        - (n_channels, n_timepoints) → add batch dimension
        
        Args:
            X: Input data
            
        Returns:
            PyTorch tensor ready for the model
        """
        if X.ndim == 2:
            if X.shape[0] == self._n_channels:
                # Single trial: (n_channels, n_timepoints) → (1, n_channels, n_timepoints)
                X = X[np.newaxis, ...]
            else:
                # Feature matrix: (n_samples, n_features) → reshape
                # This assumes features are flattened channels * samples
                n_samples = X.shape[0]
                try:
                    X = X.reshape(n_samples, self._n_channels, -1)
                except ValueError:
                    raise ValueError(
                        f"Cannot reshape input of shape {X.shape} to "
                        f"(n_samples, {self._n_channels}, n_timepoints)"
                    )
        
        # Ensure correct dtype
        X = X.astype(np.float32)
        
        return torch.FloatTensor(X)
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            'model_state_dict': self._model.state_dict() if self._model else None,
            'optimizer_state_dict': self._optimizer.state_dict() if self._optimizer else None,
            'n_channels': self._n_channels,
            'n_samples': self._n_samples,
            'architecture_params': self._get_architecture_params()
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Restore model state from serialization."""
        # Restore architecture parameters
        self._n_channels = state.get('n_channels', self._n_channels)
        self._n_samples = state.get('n_samples', self._n_samples)
        
        # Set architecture params before building model
        arch_params = state.get('architecture_params', {})
        self._set_architecture_params(arch_params)
        
        # Rebuild model if needed
        if self._model is None:
            self._model = self._build_model()
            self._model = self._model.to(self._device)
        
        # Load weights
        if state.get('model_state_dict'):
            self._model.load_state_dict(state['model_state_dict'])
        
        # Load optimizer state
        if state.get('optimizer_state_dict') and self._optimizer:
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
    
    def _get_architecture_params(self) -> Dict[str, Any]:
        """
        Get architecture-specific parameters for serialization.
        Override in subclasses.
        """
        return {
            'dropout_rate': self._dropout_rate
        }
    
    def _set_architecture_params(self, params: Dict[str, Any]) -> None:
        """
        Set architecture-specific parameters from serialization.
        Override in subclasses.
        """
        self._dropout_rate = params.get('dropout_rate', self._dropout_rate)
    
    # =========================================================================
    # PERSISTENCE (Override for PyTorch format)
    # =========================================================================
    
    def _save_deep_learning_model(self, path: Path, state: Dict) -> None:
        """
        Save deep learning model in PyTorch format.
        
        Args:
            path: Save path
            state: Complete state dictionary
        """
        torch.save(state, path)
        logger.info(f"Saved PyTorch model to {path}")
    
    def _load_deep_learning_model(self, path: Path) -> Dict:
        """
        Load deep learning model from PyTorch format.
        
        Args:
            path: Load path
            
        Returns:
            State dictionary
        """
        state = torch.load(path, map_location=self._device)
        logger.info(f"Loaded PyTorch model from {path}")
        return state
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            String representation of model architecture
        """
        if self._model is None:
            return "Model not initialized"
        
        lines = [
            f"\n{'='*60}",
            f"{self.name.upper()} Model Summary",
            f"{'='*60}",
            f"Input shape: ({self._n_channels}, {self._n_samples})",
            f"Output classes: {self._n_classes}",
            f"Device: {self._device}",
            f"{'='*60}",
            str(self._model),
            f"{'='*60}",
        ]
        
        # Count parameters
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        
        lines.extend([
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable parameters: {total_params - trainable_params:,}",
            f"{'='*60}"
        ])
        
        return '\n'.join(lines)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.
        
        Returns:
            Dict with 'total', 'trainable', 'non_trainable' counts
        """
        if self._model is None:
            return {'total': 0, 'trainable': 0, 'non_trainable': 0}
        
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def to(self, device: Union[str, Any]) -> 'BaseDeepClassifier':
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cuda', 'cpu', or torch.device)
            
        Returns:
            Self for method chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        self._device = device
        
        if self._model is not None:
            self._model = self._model.to(device)
        
        return self
    
    def train_mode(self) -> 'BaseDeepClassifier':
        """Set model to training mode."""
        if self._model is not None:
            self._model.train()
        return self
    
    def eval_mode(self) -> 'BaseDeepClassifier':
        """Set model to evaluation mode."""
        if self._model is not None:
            self._model.eval()
        return self
    
    # =========================================================================
    # PHASE-4 INTEGRATION HOOKS
    # =========================================================================
    
    def get_feature_maps(self, X: np.ndarray, layer_name: Optional[str] = None) -> np.ndarray:
        """
        Extract intermediate feature maps (for Phase-4 DVA).
        
        Args:
            X: Input data
            layer_name: Specific layer to extract from (None = last before classifier)
            
        Returns:
            Feature maps from specified layer
            
        Note:
            Override in subclasses with specific hook implementations.
        """
        logger.warning(f"get_feature_maps not implemented for {self.name}")
        return np.array([])
    
    def get_attention_weights(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get attention weights (for attention-based models).
        
        Args:
            X: Input data
            
        Returns:
            Attention weights or None if not applicable
        """
        return None  # Override in attention-based models
