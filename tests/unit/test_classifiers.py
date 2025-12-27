"""
Unit Tests for Classifiers
===========================

This module contains comprehensive unit tests for all classifiers
implemented in Phase 3.

Test Coverage:
- LDAClassifier: Training, prediction, probabilities, save/load
- SVMClassifier: Different kernels, decision function, margins
- EEGNetClassifier: Training, inference, model architecture
- ClassifierFactory: Dynamic creation, configuration
- Integration tests: Pipeline with feature extraction

Run tests:
    pytest tests/unit/test_classifiers.py -v
    
Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

import pytest
import numpy as np
import tempfile
import os

# Import classifiers
from src.classifiers import (
    LDAClassifier,
    SVMClassifier,
    EEGNetClassifier,
    ClassifierFactory,
    create_lda_classifier,
    create_svm_classifier,
    create_eegnet_classifier,
    list_available_classifiers,
    get_classifier_for_pipeline
)

# Import features for integration tests
from src.features import CSPExtractor


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def feature_data():
    """Generate synthetic feature data (e.g., after CSP extraction)."""
    np.random.seed(42)
    n_samples = 200
    n_features = 6  # Typical CSP feature count
    n_classes = 4
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Split train/test
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    return X_train, y_train, X_test, y_test


@pytest.fixture
def eeg_data():
    """Generate synthetic EEG data for deep learning classifiers."""
    np.random.seed(42)
    n_trials = 160
    n_channels = 22
    n_samples = 250  # 1 second at 250 Hz
    n_classes = 4
    
    X = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_trials // n_classes)
    
    # Split train/val/test
    X_train, y_train = X[:100], y[:100]
    X_val, y_val = X[100:130], y[100:130]
    X_test, y_test = X[130:], y[130:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture
def binary_feature_data():
    """Generate binary classification feature data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.repeat([0, 1], n_samples // 2)
    
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    return X_train, y_train, X_test, y_test


# =============================================================================
# LDA CLASSIFIER TESTS
# =============================================================================

class TestLDAClassifier:
    """Test suite for LDA classifier."""
    
    def test_creation(self):
        """Test LDA classifier creation."""
        lda = LDAClassifier()
        assert lda.name == 'lda'
        assert lda.classifier_type == 'traditional'
        assert lda.is_fitted == False
    
    def test_initialization(self):
        """Test LDA initialization with config."""
        lda = create_lda_classifier(n_classes=4, solver='svd')
        assert lda._solver == 'svd'
        assert lda._n_classes == 4
    
    def test_fit_predict(self, feature_data):
        """Test LDA fitting and prediction."""
        X_train, y_train, X_test, y_test = feature_data
        
        lda = create_lda_classifier(n_classes=4)
        lda.fit(X_train, y_train)
        
        assert lda.is_fitted == True
        
        predictions = lda.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert all(p in [0, 1, 2, 3] for p in predictions)
    
    def test_predict_proba(self, feature_data):
        """Test probability prediction."""
        X_train, y_train, X_test, y_test = feature_data
        
        lda = create_lda_classifier(n_classes=4)
        lda.fit(X_train, y_train)
        
        probas = lda.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 4)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all(probas >= 0) and np.all(probas <= 1)
    
    def test_decision_function(self, feature_data):
        """Test decision function for DVA integration."""
        X_train, y_train, X_test, y_test = feature_data
        
        lda = create_lda_classifier(n_classes=4)
        lda.fit(X_train, y_train)
        
        decision = lda.decision_function(X_test)
        
        assert decision.shape[0] == len(X_test)
    
    def test_transform(self, feature_data):
        """Test LDA transform (dimensionality reduction)."""
        X_train, y_train, X_test, y_test = feature_data
        
        lda = create_lda_classifier(n_classes=4)
        lda.fit(X_train, y_train)
        
        X_transformed = lda.transform(X_test)
        
        # LDA can reduce to at most n_classes - 1 dimensions
        assert X_transformed.shape[1] <= 3
    
    def test_save_load(self, feature_data):
        """Test save and load functionality."""
        X_train, y_train, X_test, y_test = feature_data
        
        lda = create_lda_classifier(n_classes=4)
        lda.fit(X_train, y_train)
        predictions_original = lda.predict(X_test)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            lda.save(f.name)
            
            lda_loaded = LDAClassifier()
            lda_loaded.initialize({'n_classes': 4})
            lda_loaded.load(f.name)
            
            predictions_loaded = lda_loaded.predict(X_test)
            
            np.testing.assert_array_equal(predictions_original, predictions_loaded)
            
            os.unlink(f.name)
    
    def test_regularized_lda(self, feature_data):
        """Test regularized LDA (shrinkage)."""
        X_train, y_train, X_test, y_test = feature_data
        
        lda = create_lda_classifier(
            n_classes=4, 
            solver='lsqr', 
            shrinkage='auto'
        )
        lda.fit(X_train, y_train)
        
        predictions = lda.predict(X_test)
        assert predictions.shape == (len(X_test),)


# =============================================================================
# SVM CLASSIFIER TESTS
# =============================================================================

class TestSVMClassifier:
    """Test suite for SVM classifier."""
    
    def test_creation(self):
        """Test SVM classifier creation."""
        svm = SVMClassifier()
        assert svm.name == 'svm'
        assert svm.classifier_type == 'traditional'
        assert svm.is_fitted == False
    
    def test_rbf_kernel(self, feature_data):
        """Test SVM with RBF kernel."""
        X_train, y_train, X_test, y_test = feature_data
        
        svm = create_svm_classifier(kernel='rbf', C=1.0, n_classes=4)
        svm.fit(X_train, y_train)
        
        predictions = svm.predict(X_test)
        assert predictions.shape == (len(X_test),)
    
    def test_linear_kernel(self, feature_data):
        """Test SVM with linear kernel."""
        X_train, y_train, X_test, y_test = feature_data
        
        svm = create_svm_classifier(kernel='linear', C=1.0, n_classes=4)
        svm.fit(X_train, y_train)
        
        predictions = svm.predict(X_test)
        assert predictions.shape == (len(X_test),)
    
    def test_predict_proba(self, feature_data):
        """Test probability prediction."""
        X_train, y_train, X_test, y_test = feature_data
        
        svm = create_svm_classifier(kernel='rbf', n_classes=4)
        svm.fit(X_train, y_train)
        
        probas = svm.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 4)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_decision_function(self, feature_data):
        """Test decision function."""
        X_train, y_train, X_test, y_test = feature_data
        
        svm = create_svm_classifier(kernel='rbf', n_classes=4)
        svm.fit(X_train, y_train)
        
        decision = svm.decision_function(X_test)
        assert decision.shape[0] == len(X_test)
    
    def test_support_vectors(self, feature_data):
        """Test support vector information."""
        X_train, y_train, X_test, y_test = feature_data
        
        svm = create_svm_classifier(kernel='rbf', n_classes=4)
        svm.fit(X_train, y_train)
        
        support_info = svm.get_support_info()
        assert 'n_support' in support_info
        assert 'total_support' in support_info
        assert support_info['total_support'] > 0
    
    def test_margin_info(self, feature_data):
        """Test margin information for DVA."""
        X_train, y_train, X_test, y_test = feature_data
        
        svm = create_svm_classifier(kernel='rbf', n_classes=4)
        svm.fit(X_train, y_train)
        
        margin_info = svm.get_margin_info(X_test)
        
        assert 'decision_values' in margin_info
        assert 'margin' in margin_info
        assert 'predictions' in margin_info
        assert margin_info['margin'].shape == (len(X_test),)
    
    def test_save_load(self, feature_data):
        """Test save and load functionality."""
        X_train, y_train, X_test, y_test = feature_data
        
        svm = create_svm_classifier(kernel='rbf', n_classes=4)
        svm.fit(X_train, y_train)
        predictions_original = svm.predict(X_test)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            svm.save(f.name)
            
            svm_loaded = SVMClassifier()
            svm_loaded.initialize({'n_classes': 4, 'kernel': 'rbf'})
            svm_loaded.load(f.name)
            
            predictions_loaded = svm_loaded.predict(X_test)
            
            np.testing.assert_array_equal(predictions_original, predictions_loaded)
            
            os.unlink(f.name)


# =============================================================================
# EEGNET CLASSIFIER TESTS
# =============================================================================

class TestEEGNetClassifier:
    """Test suite for EEGNet classifier."""
    
    def test_creation(self):
        """Test EEGNet classifier creation."""
        eegnet = EEGNetClassifier()
        assert eegnet.name == 'eegnet'
        assert eegnet.classifier_type == 'deep_learning'
    
    def test_initialization(self):
        """Test EEGNet initialization."""
        eegnet = create_eegnet_classifier(
            n_classes=4,
            n_channels=22,
            n_samples=250,
            F1=8,
            D=2,
            device='cpu'
        )
        
        assert eegnet._n_classes == 4
        assert eegnet._n_channels == 22
        assert eegnet._F1 == 8
        assert eegnet._D == 2
    
    def test_model_architecture(self):
        """Test model architecture creation."""
        eegnet = create_eegnet_classifier(
            n_classes=4,
            n_channels=22,
            n_samples=250,
            device='cpu'
        )
        
        params = eegnet.count_parameters()
        assert params['total'] > 0
        assert params['trainable'] > 0
    
    def test_fit_predict(self, eeg_data):
        """Test EEGNet fitting and prediction."""
        X_train, y_train, X_val, y_val, X_test, y_test = eeg_data
        
        eegnet = create_eegnet_classifier(
            n_classes=4,
            n_channels=22,
            n_samples=250,
            device='cpu'
        )
        
        eegnet.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=2,  # Quick test
            batch_size=16,
            verbose=0
        )
        
        assert eegnet.is_fitted == True
        
        predictions = eegnet.predict(X_test)
        assert predictions.shape == (len(X_test),)
    
    def test_predict_proba(self, eeg_data):
        """Test probability prediction."""
        X_train, y_train, X_val, y_val, X_test, y_test = eeg_data
        
        eegnet = create_eegnet_classifier(
            n_classes=4,
            n_channels=22,
            n_samples=250,
            device='cpu'
        )
        
        eegnet.fit(X_train, y_train, epochs=2, verbose=0)
        
        probas = eegnet.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 4)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_training_history(self, eeg_data):
        """Test training history tracking."""
        X_train, y_train, X_val, y_val, X_test, y_test = eeg_data
        
        eegnet = create_eegnet_classifier(
            n_classes=4,
            n_channels=22,
            n_samples=250,
            device='cpu'
        )
        
        eegnet.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3,
            verbose=0
        )
        
        history = eegnet.get_training_history()
        
        assert 'train_loss' in history
        assert 'train_accuracy' in history
        assert 'val_loss' in history
        assert 'val_accuracy' in history
        assert len(history['train_loss']) == 3
    
    def test_feature_maps(self, eeg_data):
        """Test feature map extraction."""
        X_train, y_train, X_val, y_val, X_test, y_test = eeg_data
        
        eegnet = create_eegnet_classifier(
            n_classes=4,
            n_channels=22,
            n_samples=250,
            device='cpu'
        )
        
        eegnet.fit(X_train, y_train, epochs=2, verbose=0)
        
        features = eegnet.get_feature_maps(X_test[:5], layer_name='block2')
        assert features.shape[0] == 5
    
    def test_save_load(self, eeg_data):
        """Test save and load functionality."""
        X_train, y_train, X_val, y_val, X_test, y_test = eeg_data
        
        eegnet = create_eegnet_classifier(
            n_classes=4,
            n_channels=22,
            n_samples=250,
            device='cpu'
        )
        
        eegnet.fit(X_train, y_train, epochs=2, verbose=0)
        predictions_original = eegnet.predict(X_test)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            eegnet.save(f.name)
            
            eegnet_loaded = EEGNetClassifier()
            eegnet_loaded.initialize({
                'n_classes': 4,
                'n_channels': 22,
                'n_samples': 250,
                'device': 'cpu'
            })
            eegnet_loaded.load(f.name)
            
            predictions_loaded = eegnet_loaded.predict(X_test)
            
            np.testing.assert_array_equal(predictions_original, predictions_loaded)
            
            os.unlink(f.name)


# =============================================================================
# CLASSIFIER FACTORY TESTS
# =============================================================================

class TestClassifierFactory:
    """Test suite for Classifier Factory."""
    
    def test_list_classifiers(self):
        """Test listing available classifiers."""
        available = list_available_classifiers()
        
        assert 'traditional' in available
        assert 'deep_learning' in available
        assert 'lda' in available['traditional']
        assert 'svm' in available['traditional']
        assert 'eegnet' in available['deep_learning']
    
    def test_create_lda(self):
        """Test LDA creation via factory."""
        lda = ClassifierFactory.create('lda', n_classes=4)
        assert lda.name == 'lda'
    
    def test_create_svm(self):
        """Test SVM creation via factory."""
        svm = ClassifierFactory.create('svm', kernel='rbf', n_classes=4)
        assert svm.name == 'svm'
    
    def test_create_eegnet(self):
        """Test EEGNet creation via factory."""
        eegnet = ClassifierFactory.create(
            'eegnet',
            n_classes=4,
            n_channels=22,
            n_samples=250,
            device='cpu'
        )
        assert eegnet.name == 'eegnet'
    
    def test_from_config(self):
        """Test creation from config dictionary."""
        config = {
            'name': 'svm',
            'kernel': 'linear',
            'C': 10.0,
            'n_classes': 4
        }
        
        clf = ClassifierFactory.from_config(config)
        assert clf.name == 'svm'
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = ClassifierFactory.get_default_config('eegnet')
        
        assert 'F1' in config
        assert 'D' in config
        assert 'dropout_rate' in config
    
    def test_get_classifier_info(self):
        """Test getting classifier information."""
        info = ClassifierFactory.get_classifier_info('svm')
        
        assert 'name' in info
        assert 'class' in info
        assert 'type' in info
        assert info['type'] == 'traditional'
    
    def test_invalid_classifier(self):
        """Test creating invalid classifier raises error."""
        with pytest.raises(ValueError):
            ClassifierFactory.create('invalid_classifier', n_classes=4)
    
    def test_pipeline_classifiers(self):
        """Test pipeline classifier creation."""
        clf_csp_lda = get_classifier_for_pipeline('csp_lda', n_classes=4)
        assert clf_csp_lda.name == 'lda'
        
        clf_csp_svm = get_classifier_for_pipeline('csp_svm', n_classes=4)
        assert clf_csp_svm.name == 'svm'
        
        clf_end_to_end = get_classifier_for_pipeline(
            'end_to_end',
            n_classes=4,
            n_channels=22,
            n_samples=250,
            device='cpu'
        )
        assert clf_end_to_end.name == 'eegnet'


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining features and classifiers."""
    
    def test_csp_lda_pipeline(self):
        """Test CSP + LDA pipeline."""
        np.random.seed(42)
        
        # Generate EEG data
        n_trials = 100
        n_channels = 22
        n_samples = 500
        
        X = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
        y = np.repeat([0, 1], n_trials // 2)
        
        # Split
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Feature extraction
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        X_train_csp = csp.fit_extract(X_train, y_train)
        X_test_csp = csp.extract(X_test)
        
        # Classification
        lda = create_lda_classifier(n_classes=2)
        lda.fit(X_train_csp, y_train)
        predictions = lda.predict(X_test_csp)
        
        assert predictions.shape == (len(X_test),)
        assert all(p in [0, 1] for p in predictions)
    
    def test_csp_svm_pipeline(self):
        """Test CSP + SVM pipeline."""
        np.random.seed(42)
        
        n_trials = 100
        n_channels = 22
        n_samples = 500
        
        X = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
        y = np.repeat([0, 1], n_trials // 2)
        
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        X_train_csp = csp.fit_extract(X_train, y_train)
        X_test_csp = csp.extract(X_test)
        
        svm = create_svm_classifier(kernel='rbf', C=1.0, n_classes=2)
        svm.fit(X_train_csp, y_train)
        predictions = svm.predict(X_test_csp)
        
        assert predictions.shape == (len(X_test),)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_unfitted_predict(self, feature_data):
        """Test prediction without fitting raises error."""
        X_train, y_train, X_test, y_test = feature_data
        
        lda = create_lda_classifier(n_classes=4)
        
        with pytest.raises(RuntimeError):
            lda.predict(X_test)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        lda = create_lda_classifier(n_classes=4)
        
        with pytest.raises(ValueError):
            lda.fit(np.array([]), np.array([]))
    
    def test_single_class(self, feature_data):
        """Test handling of single class raises error."""
        X_train, y_train, X_test, y_test = feature_data
        
        lda = create_lda_classifier(n_classes=4)
        
        with pytest.raises(ValueError):
            lda.fit(X_train, np.zeros(len(X_train)))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
