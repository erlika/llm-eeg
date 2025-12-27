"""
EEGNet Classifier - Compact CNN for EEG Classification
=======================================================

This module implements EEGNet, a compact convolutional neural network 
designed specifically for EEG-based brain-computer interfaces. EEGNet 
uses depthwise and separable convolutions to create an efficient model
that works well across different BCI paradigms.

Reference:
    Lawhern, V.J., Solon, A.J., Waytowich, N.R., Gordon, S.M., Hung, C.P. 
    and Lance, B.J., 2018. EEGNet: a compact convolutional neural network 
    for EEG-based brain–computer interfaces. Journal of neural engineering.

Architecture:
    1. Temporal Convolution: Learn frequency filters
    2. Depthwise Convolution: Learn spatial filters (per channel)
    3. Separable Convolution: Learn temporal patterns
    4. Classification Head: Dense layer for output

Key Features:
- End-to-end learning (no manual feature extraction needed)
- Works with raw preprocessed EEG data
- Compact model (~2.6k parameters for 22 channels, 4 classes)
- Effective across different BCI paradigms

Performance in BCI Competition IV-2a:
- Typically achieves 70-75% accuracy
- Can reach 82-85% when combined with CSP preprocessing
- Good generalization across subjects

Example:
    ```python
    from src.classifiers.models.deep_learning.eegnet import EEGNetClassifier
    
    # Create and train EEGNet
    clf = EEGNetClassifier()
    clf.initialize({
        'n_classes': 4,
        'n_channels': 22,
        'n_samples': 1000,  # 4 seconds at 250 Hz
        'F1': 8,
        'D': 2,
        'F2': 16,
        'dropout_rate': 0.5
    })
    
    clf.fit(X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=100)
    
    predictions = clf.predict(X_test)
    ```

Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from .base_deep import BaseDeepClassifier


# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# EEGNet PyTorch Model
# =============================================================================

class EEGNetModel(nn.Module):
    """
    EEGNet PyTorch implementation.
    
    Architecture:
        Block 1: 
            - Temporal convolution (across time)
            - Batch normalization
            - Depthwise convolution (per channel spatial filter)
            - Batch normalization
            - ELU activation
            - Average pooling
            - Dropout
            
        Block 2:
            - Separable convolution (depthwise + pointwise)
            - Batch normalization
            - ELU activation
            - Average pooling
            - Dropout
            
        Classification:
            - Flatten
            - Dense (fully connected)
    
    Args:
        n_classes: Number of output classes
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        F1: Number of temporal filters (default: 8)
        D: Depth multiplier for depthwise convolution (default: 2)
        F2: Number of pointwise filters (default: F1 * D = 16)
        kernel_length: Temporal conv kernel size (default: n_samples // 2)
        dropout_rate: Dropout probability (default: 0.5)
    """
    
    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_samples: int = 1000,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: Optional[int] = None,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate
        
        # Default kernel length: half the sampling rate (125 at 250 Hz)
        if kernel_length is None:
            kernel_length = max(n_samples // 4, 16)  # Minimum 16
        self.kernel_length = kernel_length
        
        # =====================================================================
        # Block 1: Temporal + Spatial (Depthwise) Convolution
        # =====================================================================
        
        # Temporal convolution: Learn frequency filters
        # Input: (batch, 1, channels, samples)
        # Output: (batch, F1, channels, samples)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise convolution: Learn spatial filters
        # Input: (batch, F1, channels, samples)
        # Output: (batch, F1*D, 1, samples)
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,  # Depthwise: each input channel has separate filters
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        # Pooling after block 1
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # =====================================================================
        # Block 2: Separable Convolution
        # =====================================================================
        
        # Calculate dimensions after pool1
        samples_after_pool1 = n_samples // 4
        
        # Separable convolution = depthwise + pointwise
        # Depthwise part: temporal filtering
        separable_kernel = min(16, max(4, samples_after_pool1 // 4))
        
        self.separable_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, separable_kernel),
            padding=(0, separable_kernel // 2),
            groups=F1 * D,  # Depthwise
            bias=False
        )
        
        # Pointwise part: channel mixing
        self.separable_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        
        # Pooling after block 2
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # =====================================================================
        # Classification Head
        # =====================================================================
        
        # Calculate flattened size
        samples_after_pool2 = samples_after_pool1 // 8
        if samples_after_pool2 < 1:
            samples_after_pool2 = 1
        
        flatten_size = F2 * samples_after_pool2
        
        self.fc = nn.Linear(flatten_size, n_classes)
        
        # Store for later reference
        self._flatten_size = flatten_size
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, samples)
               or (batch, 1, channels, samples)
        
        Returns:
            Output logits of shape (batch, n_classes)
        """
        # Ensure 4D input: (batch, 1, channels, samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Block 1: Temporal + Depthwise
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2: Separable
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor, layer: str = 'block2') -> torch.Tensor:
        """
        Extract intermediate feature maps.
        
        Args:
            x: Input tensor
            layer: Which layer to extract ('block1', 'block2')
        
        Returns:
            Feature maps at specified layer
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        
        if layer == 'block1':
            return x
        
        # Block 2
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        
        return x


# =============================================================================
# EEGNet Classifier Wrapper
# =============================================================================

class EEGNetClassifier(BaseDeepClassifier):
    """
    EEGNet classifier for end-to-end EEG classification.
    
    EEGNet is a compact CNN that can work directly with preprocessed
    EEG data without manual feature extraction. It's designed to learn
    both spatial and temporal patterns in EEG signals.
    
    Recommended for:
    - End-to-end learning experiments
    - When CSP+classifier underperforms
    - Cross-subject generalization
    - Small training datasets
    
    Configuration Options:
        - n_classes: Number of output classes (default: 4)
        - n_channels: Number of EEG channels (default: 22)
        - n_samples: Time samples per trial (default: 1000)
        - F1: Temporal filters (default: 8)
        - D: Depth multiplier (default: 2)
        - F2: Pointwise filters (default: 16)
        - kernel_length: Temporal conv kernel (default: n_samples // 4)
        - dropout_rate: Dropout probability (default: 0.5)
        - learning_rate: Adam LR (default: 0.001)
        
    Attributes:
        F1, D, F2: EEGNet architecture parameters
        kernel_length: Temporal convolution kernel size
    """
    
    def __init__(self):
        """Initialize EEGNet classifier."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for EEGNet. "
                "Install with: pip install torch"
            )
        
        super().__init__()
        
        # EEGNet-specific parameters
        self._F1: int = 8
        self._D: int = 2
        self._F2: int = 16
        self._kernel_length: Optional[int] = None
        
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Classifier name."""
        return 'eegnet'
    
    @property
    def F1(self) -> int:
        """Number of temporal filters."""
        return self._F1
    
    @property
    def D(self) -> int:
        """Depth multiplier."""
        return self._D
    
    @property
    def F2(self) -> int:
        """Number of pointwise filters."""
        return self._F2
    
    # =========================================================================
    # MODEL BUILDING
    # =========================================================================
    
    def _initialize_implementation(self, config: Dict[str, Any]) -> None:
        """
        Initialize EEGNet-specific parameters.
        
        Args:
            config: Configuration with EEGNet parameters
        """
        # Extract EEGNet-specific parameters before parent initialization
        self._F1 = config.get('F1', 8)
        self._D = config.get('D', 2)
        self._F2 = config.get('F2', self._F1 * self._D)
        self._kernel_length = config.get('kernel_length', None)
        
        # Call parent initialization (will call _build_model)
        super()._initialize_implementation(config)
        
        logger.info(
            f"EEGNet initialized with F1={self._F1}, D={self._D}, F2={self._F2}"
        )
    
    def _build_model(self):
        """Build the EEGNet model."""
        return EEGNetModel(
            n_classes=self._n_classes,
            n_channels=self._n_channels,
            n_samples=self._n_samples,
            F1=self._F1,
            D=self._D,
            F2=self._F2,
            kernel_length=self._kernel_length,
            dropout_rate=self._dropout_rate
        )
    
    # =========================================================================
    # ARCHITECTURE PARAMETERS
    # =========================================================================
    
    def _get_architecture_params(self) -> Dict[str, Any]:
        """Get EEGNet architecture parameters."""
        params = super()._get_architecture_params()
        params.update({
            'F1': self._F1,
            'D': self._D,
            'F2': self._F2,
            'kernel_length': self._kernel_length
        })
        return params
    
    def _set_architecture_params(self, params: Dict[str, Any]) -> None:
        """Set EEGNet architecture parameters."""
        super()._set_architecture_params(params)
        self._F1 = params.get('F1', 8)
        self._D = params.get('D', 2)
        self._F2 = params.get('F2', 16)
        self._kernel_length = params.get('kernel_length', None)
    
    # =========================================================================
    # FEATURE EXTRACTION (for Phase-4 DVA)
    # =========================================================================
    
    def get_feature_maps(self, X: np.ndarray, layer_name: str = 'block2') -> np.ndarray:
        """
        Extract intermediate feature maps for visualization/analysis.
        
        Useful for Phase-4 DVA agent to understand model decisions.
        
        Args:
            X: Input EEG data
            layer_name: Layer to extract ('block1', 'block2')
        
        Returns:
            Feature maps at specified layer
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted first")
        
        self._model.eval()
        X_tensor = self._prepare_input(X)
        
        with torch.no_grad():
            X_tensor = X_tensor.to(self._device)
            features = self._model.get_feature_maps(X_tensor, layer_name)
        
        return features.cpu().numpy()
    
    def get_temporal_patterns(self) -> np.ndarray:
        """
        Get learned temporal filter patterns.
        
        Returns:
            Temporal filter weights from first convolution
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        return self._model.conv1.weight.detach().cpu().numpy()
    
    def get_spatial_patterns(self) -> np.ndarray:
        """
        Get learned spatial filter patterns (depthwise conv).
        
        Returns:
            Spatial filter weights from depthwise convolution
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        return self._model.depthwise_conv.weight.detach().cpu().numpy()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_eegnet_classifier(
    n_classes: int = 4,
    n_channels: int = 22,
    n_samples: int = 1000,
    F1: int = 8,
    D: int = 2,
    F2: Optional[int] = None,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    device: str = 'auto'
) -> EEGNetClassifier:
    """
    Create and initialize an EEGNet classifier.
    
    Args:
        n_classes: Number of output classes
        n_channels: Number of EEG channels
        n_samples: Time samples per trial
        F1: Number of temporal filters
        D: Depth multiplier
        F2: Pointwise filters (default: F1 * D)
        dropout_rate: Dropout probability
        learning_rate: Adam learning rate
        device: 'cuda', 'cpu', or 'auto'
    
    Returns:
        Initialized EEGNetClassifier
        
    Example:
        >>> clf = create_eegnet_classifier(n_classes=4, n_channels=22)
        >>> clf.fit(X_train, y_train, epochs=100)
    """
    if F2 is None:
        F2 = F1 * D
    
    clf = EEGNetClassifier()
    clf.initialize({
        'n_classes': n_classes,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'F1': F1,
        'D': D,
        'F2': F2,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'device': device
    })
    
    return clf


def create_eegnet_for_motor_imagery(
    n_channels: int = 22,
    sampling_rate: int = 250,
    trial_duration: float = 4.0,
    dropout_rate: float = 0.5
) -> EEGNetClassifier:
    """
    Create EEGNet optimized for BCI Competition IV-2a motor imagery.
    
    Pre-configured for:
    - 4 classes (left hand, right hand, feet, tongue)
    - 22 EEG channels
    - 250 Hz sampling rate
    - 4 second trials
    
    Args:
        n_channels: Number of EEG channels
        sampling_rate: Sampling rate in Hz
        trial_duration: Trial duration in seconds
        dropout_rate: Dropout probability
    
    Returns:
        EEGNet configured for motor imagery
    """
    n_samples = int(sampling_rate * trial_duration)
    
    return create_eegnet_classifier(
        n_classes=4,
        n_channels=n_channels,
        n_samples=n_samples,
        F1=8,
        D=2,
        F2=16,
        dropout_rate=dropout_rate,
        learning_rate=0.001
    )
