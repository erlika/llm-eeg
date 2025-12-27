"""
Classifiers Package for EEG-BCI Framework
==========================================

This package provides a comprehensive set of classifiers for EEG-based
brain-computer interface applications. It includes both traditional
machine learning classifiers and deep learning models.

Package Structure:
    classifiers/
    ├── __init__.py          # This file - public API
    ├── base.py              # BaseClassifier abstract class
    ├── factory.py           # ClassifierFactory for dynamic creation
    └── models/
        ├── traditional/     # LDA, SVM
        └── deep_learning/   # EEGNet, EEG-DCNet (future)

Classifier Types:

    Traditional ML (for extracted features):
    - LDAClassifier: Linear Discriminant Analysis
    - SVMClassifier: Support Vector Machine
    
    Deep Learning (for raw/preprocessed EEG):
    - EEGNetClassifier: Compact CNN for EEG

Typical Usage:

    1. Feature-based Classification (CSP + LDA/SVM):
        ```python
        from src.features import CSPExtractor
        from src.classifiers import create_lda_classifier
        
        # Extract features
        csp = CSPExtractor(n_components=6)
        X_train_csp = csp.fit_extract(X_train, y_train)
        X_test_csp = csp.extract(X_test)
        
        # Train classifier
        clf = create_lda_classifier(n_classes=4)
        clf.fit(X_train_csp, y_train)
        predictions = clf.predict(X_test_csp)
        ```
    
    2. End-to-end Classification (EEGNet):
        ```python
        from src.classifiers import create_eegnet_classifier
        
        # Create and train EEGNet
        clf = create_eegnet_classifier(
            n_classes=4, n_channels=22, n_samples=1000
        )
        clf.fit(X_train, y_train, epochs=100)
        predictions = clf.predict(X_test)
        ```
    
    3. Using ClassifierFactory:
        ```python
        from src.classifiers import ClassifierFactory
        
        # Create any classifier dynamically
        clf = ClassifierFactory.create('eegnet', n_classes=4, n_channels=22)
        
        # Or from config
        config = {'name': 'svm', 'kernel': 'rbf', 'C': 10.0, 'n_classes': 4}
        clf = ClassifierFactory.from_config(config)
        ```

Performance Benchmarks (BCI Competition IV-2a):
    - CSP + LDA: ~75-80% accuracy
    - CSP + SVM: ~78-82% accuracy
    - EEGNet: ~70-75% accuracy
    - CSP + EEGNet: ~82-85% accuracy

Phase-4 Integration:
    - All classifiers provide decision_function() for DVA agent
    - Feature importance available for interpretability
    - Training history for analysis

Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

# Base classes
from .base import BaseClassifier, register_classifier

# Factory
from .factory import (
    ClassifierFactory,
    create_classifier,
    create_traditional_classifier,
    create_deep_learning_classifier,
    list_available_classifiers,
    get_classifier_for_pipeline
)

# Traditional ML classifiers
from .models.traditional import (
    LDAClassifier,
    SVMClassifier,
    create_lda_classifier,
    create_regularized_lda,
    create_svm_classifier,
    create_linear_svm,
    create_rbf_svm
)

# Deep learning classifiers
from .models.deep_learning import (
    BaseDeepClassifier,
    TORCH_AVAILABLE,
    EEGNetClassifier,
    EEGNetModel,
    create_eegnet_classifier,
    create_eegnet_for_motor_imagery
)

__all__ = [
    # Base classes
    'BaseClassifier',
    'BaseDeepClassifier',
    'register_classifier',
    'TORCH_AVAILABLE',
    
    # Factory
    'ClassifierFactory',
    'create_classifier',
    'create_traditional_classifier',
    'create_deep_learning_classifier',
    'list_available_classifiers',
    'get_classifier_for_pipeline',
    
    # Traditional ML
    'LDAClassifier',
    'SVMClassifier',
    'create_lda_classifier',
    'create_regularized_lda',
    'create_svm_classifier',
    'create_linear_svm',
    'create_rbf_svm',
    
    # Deep Learning
    'EEGNetClassifier',
    'EEGNetModel',
    'create_eegnet_classifier',
    'create_eegnet_for_motor_imagery',
]

# Version info
__version__ = '3.0.0'
__phase__ = 3
