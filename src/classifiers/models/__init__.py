"""
Classifier Models Package
=========================

This package contains classifier model implementations organized by type.

Sub-packages:
- traditional/: Traditional ML classifiers (LDA, SVM)
- deep_learning/: Deep learning classifiers (EEGNet, etc.)

Author: EEG-BCI Framework
Date: 2024
Phase: 3
"""

from .traditional import (
    LDAClassifier,
    SVMClassifier,
    create_lda_classifier,
    create_svm_classifier
)

from .deep_learning import (
    BaseDeepClassifier,
    EEGNetClassifier,
    create_eegnet_classifier
)

__all__ = [
    # Traditional
    'LDAClassifier',
    'SVMClassifier',
    'create_lda_classifier',
    'create_svm_classifier',
    
    # Deep Learning
    'BaseDeepClassifier',
    'EEGNetClassifier',
    'create_eegnet_classifier',
]
