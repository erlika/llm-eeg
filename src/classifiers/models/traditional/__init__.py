"""
Traditional ML Classifiers for EEG-BCI
======================================

This package provides traditional machine learning classifiers
optimized for EEG-based brain-computer interface applications.

Available Classifiers:
- LDAClassifier: Linear Discriminant Analysis
- SVMClassifier: Support Vector Machine

These classifiers work best with extracted features (e.g., CSP features)
rather than raw EEG data. For end-to-end learning, use deep learning
classifiers in the deep_learning package.

Typical Pipeline:
    1. Load EEG data (Phase 2)
    2. Preprocess (Phase 2)
    3. Extract features (CSP, Band Power, etc.)
    4. Train classifier (LDA or SVM)
    5. Predict and evaluate

Example:
    ```python
    from src.classifiers.models.traditional import (
        LDAClassifier, 
        SVMClassifier,
        create_lda_classifier,
        create_svm_classifier
    )
    
    # LDA with CSP features
    lda = create_lda_classifier()
    lda.fit(X_csp_train, y_train)
    
    # SVM with CSP features
    svm = create_svm_classifier(kernel='rbf', C=1.0)
    svm.fit(X_csp_train, y_train)
    ```

Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

from .lda import (
    LDAClassifier,
    create_lda_classifier,
    create_regularized_lda
)

from .svm import (
    SVMClassifier,
    create_svm_classifier,
    create_linear_svm,
    create_rbf_svm
)

__all__ = [
    # Classes
    'LDAClassifier',
    'SVMClassifier',
    
    # LDA factory functions
    'create_lda_classifier',
    'create_regularized_lda',
    
    # SVM factory functions
    'create_svm_classifier',
    'create_linear_svm',
    'create_rbf_svm',
]
