"""
Deep Learning Classifiers for EEG-BCI
=====================================

This package provides PyTorch-based deep learning classifiers
for EEG brain-computer interface applications.

Available Classifiers:
- BaseDeepClassifier: Abstract base class for all DL classifiers
- EEGNetClassifier: Compact CNN for EEG (recommended starting point)

Future Additions (Phase 3+):
- EEGDCNetClassifier: EEG-DCNet for improved accuracy
- ShallowConvNetClassifier: Shallow ConvNet
- DeepConvNetClassifier: Deep ConvNet
- ATCNetClassifier: Attention Temporal Convolutional Network

These classifiers work with preprocessed EEG data:
- Input shape: (n_trials, n_channels, n_samples)
- Output: class predictions or probabilities

Typical Pipeline:
    1. Load EEG data (Phase 2)
    2. Preprocess (bandpass filter, normalization)
    3. Train deep learning classifier
    4. Predict and evaluate

Example:
    ```python
    from src.classifiers.models.deep_learning import (
        EEGNetClassifier,
        create_eegnet_classifier,
        create_eegnet_for_motor_imagery
    )
    
    # Create EEGNet for BCI Competition IV-2a
    clf = create_eegnet_for_motor_imagery()
    
    # Train
    clf.fit(X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=100)
    
    # Predict
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)
    ```

Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

from .base_deep import BaseDeepClassifier, TORCH_AVAILABLE
from .eegnet import (
    EEGNetClassifier,
    EEGNetModel,
    create_eegnet_classifier,
    create_eegnet_for_motor_imagery
)

__all__ = [
    # Base class
    'BaseDeepClassifier',
    'TORCH_AVAILABLE',
    
    # EEGNet
    'EEGNetClassifier',
    'EEGNetModel',
    'create_eegnet_classifier',
    'create_eegnet_for_motor_imagery',
]
