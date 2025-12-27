"""
Unit Tests for Feature Extractors
==================================

This module contains comprehensive unit tests for all feature extractors
implemented in Phase 3.

Test Coverage:
- CSPExtractor: Fit, transform, save/load, multi-class
- BandPowerExtractor: Frequency band extraction, PSD
- TimeDomainExtractor: Statistical features, Hjorth parameters
- FeatureExtractorFactory: Dynamic creation
- FeatureExtractionPipeline: Multi-extractor pipeline

Run tests:
    pytest tests/unit/test_feature_extractors.py -v
    
Author: EEG-BCI Framework
Date: 2024
Phase: 3 - Feature Extraction & Classification
"""

import pytest
import numpy as np
import tempfile
import os

# Import feature extractors
from src.features import (
    CSPExtractor,
    BandPowerExtractor,
    TimeDomainExtractor,
    FeatureExtractorFactory,
    FeatureExtractionPipeline,
    create_csp_extractor,
    create_band_power_extractor,
    create_motor_imagery_pipeline
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_eeg_data():
    """Generate synthetic EEG data for testing."""
    np.random.seed(42)
    n_trials = 100
    n_channels = 22
    n_samples = 1000  # 4 seconds at 250 Hz
    n_classes = 4
    
    X = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_trials // n_classes)
    
    return X, y


@pytest.fixture
def binary_eeg_data():
    """Generate binary classification EEG data."""
    np.random.seed(42)
    n_trials = 50
    n_channels = 22
    n_samples = 500
    
    X = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
    y = np.repeat([0, 1], n_trials // 2)
    
    return X, y


@pytest.fixture
def single_trial_data():
    """Generate single trial EEG data."""
    np.random.seed(42)
    n_channels = 22
    n_samples = 1000
    
    X = np.random.randn(n_channels, n_samples).astype(np.float32)
    return X


# =============================================================================
# CSP EXTRACTOR TESTS
# =============================================================================

class TestCSPExtractor:
    """Test suite for CSP feature extractor."""
    
    def test_creation(self):
        """Test CSP extractor creation."""
        csp = CSPExtractor(n_components=6)
        assert csp.name == 'csp'
        assert csp.is_trainable == True
        assert csp._n_components == 6
    
    def test_fit_binary(self, binary_eeg_data):
        """Test CSP fitting with binary classification."""
        X, y = binary_eeg_data
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        csp.fit(X, y)
        
        assert csp._is_fitted == True
        assert csp._filters is not None
        assert csp._filters.shape[0] == 4
    
    def test_extract_binary(self, binary_eeg_data):
        """Test feature extraction for binary classification."""
        X, y = binary_eeg_data
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        features = csp.fit_extract(X, y)
        
        assert features.shape == (len(X), 4)
        assert not np.any(np.isnan(features))
    
    def test_extract_multiclass(self, synthetic_eeg_data):
        """Test feature extraction for multi-class (OVR)."""
        X, y = synthetic_eeg_data
        
        # CSP with multi-class data uses one-vs-rest internally
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        features = csp.fit_extract(X, y)
        
        # For multi-class, CSP produces features based on n_components
        # The exact number depends on implementation (pairwise or OVR)
        assert features.shape[0] == len(X)
        assert features.shape[1] >= 4  # At least n_components features
    
    def test_single_trial_extraction(self, binary_eeg_data, single_trial_data):
        """Test extracting features from a single trial."""
        X_train, y_train = binary_eeg_data
        X_single = single_trial_data
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        csp.fit(X_train, y_train)
        
        features = csp.extract(X_single)
        assert features.shape == (4,)
    
    def test_save_load(self, binary_eeg_data):
        """Test save and load functionality."""
        X, y = binary_eeg_data
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        csp.fit(X, y)
        features_original = csp.extract(X)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            csp.save(f.name)
            
            # Use classmethod load instead of instance method
            csp_loaded = CSPExtractor.load(f.name)
            
            features_loaded = csp_loaded.extract(X)
            
            np.testing.assert_array_almost_equal(features_original, features_loaded)
            
            os.unlink(f.name)
    
    def test_feature_names(self, binary_eeg_data):
        """Test feature name generation."""
        X, y = binary_eeg_data
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        csp.fit(X, y)
        
        names = csp.get_feature_names()
        assert len(names) == 4
        assert all('csp' in name.lower() for name in names)


# =============================================================================
# BAND POWER EXTRACTOR TESTS
# =============================================================================

class TestBandPowerExtractor:
    """Test suite for Band Power feature extractor."""
    
    def test_creation(self):
        """Test band power extractor creation."""
        bp = BandPowerExtractor()
        assert bp.name == 'band_power'
        assert bp.is_trainable == False
    
    def test_default_bands(self, synthetic_eeg_data):
        """Test extraction with default frequency bands."""
        X, y = synthetic_eeg_data
        
        bp = BandPowerExtractor()
        bp.initialize({'sampling_rate': 250})
        features = bp.extract(X)
        
        assert features.shape[0] == len(X)
        assert features.shape[1] > 0
        assert not np.any(np.isnan(features))
    
    def test_custom_bands(self, synthetic_eeg_data):
        """Test extraction with custom frequency bands."""
        X, y = synthetic_eeg_data
        
        bands = {'mu': (8, 12), 'beta': (12, 30)}
        bp = BandPowerExtractor(bands=bands)
        bp.initialize({'sampling_rate': 250})
        features = bp.extract(X)
        
        n_channels = X.shape[1]
        expected_features = n_channels * len(bands)  # One feature per band per channel
        assert features.shape[1] == expected_features
    
    def test_average_channels(self, synthetic_eeg_data):
        """Test averaging across channels."""
        X, y = synthetic_eeg_data
        
        bands = {'mu': (8, 12), 'beta': (12, 30)}
        bp = BandPowerExtractor(bands=bands, average_channels=True)
        bp.initialize({'sampling_rate': 250})
        features = bp.extract(X)
        
        # Should have one feature per band
        assert features.shape[1] == len(bands)
    
    def test_single_trial(self, single_trial_data):
        """Test single trial extraction."""
        bp = BandPowerExtractor()
        bp.initialize({'sampling_rate': 250})
        features = bp.extract(single_trial_data)
        
        assert features.ndim == 1
    
    def test_feature_names(self, synthetic_eeg_data):
        """Test feature name generation."""
        X, y = synthetic_eeg_data
        
        bands = {'mu': (8, 12), 'beta': (12, 30)}
        # Test with average_channels=True which generates predictable feature names
        bp = BandPowerExtractor(bands=bands, average_channels=True)
        bp.initialize({'sampling_rate': 250})
        
        # Extract features first to populate feature names
        bp.extract(X)
        names = bp.get_feature_names()
        assert len(names) > 0
        assert any('mu' in name.lower() for name in names)


# =============================================================================
# TIME DOMAIN EXTRACTOR TESTS
# =============================================================================

class TestTimeDomainExtractor:
    """Test suite for Time Domain feature extractor."""
    
    def test_creation(self):
        """Test time domain extractor creation."""
        td = TimeDomainExtractor()
        assert td.name == 'time_domain'
        assert td.is_trainable == False
    
    def test_default_features(self, synthetic_eeg_data):
        """Test extraction with default features."""
        X, y = synthetic_eeg_data
        
        td = TimeDomainExtractor()
        td.initialize({'sampling_rate': 250})
        features = td.extract(X)
        
        assert features.shape[0] == len(X)
        assert not np.any(np.isnan(features))
    
    def test_specific_features(self, synthetic_eeg_data):
        """Test extraction with specific features."""
        X, y = synthetic_eeg_data
        
        feature_list = ['mean', 'variance', 'rms']
        td = TimeDomainExtractor(features=feature_list)
        td.initialize({'sampling_rate': 250})
        features = td.extract(X)
        
        n_channels = X.shape[1]
        expected_features = n_channels * len(feature_list)
        assert features.shape[1] == expected_features
    
    def test_hjorth_parameters(self, synthetic_eeg_data):
        """Test Hjorth parameter extraction."""
        X, y = synthetic_eeg_data
        
        feature_list = ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']
        td = TimeDomainExtractor(features=feature_list)
        td.initialize({'sampling_rate': 250})
        features = td.extract(X)
        
        n_channels = X.shape[1]
        expected_features = n_channels * 3  # 3 Hjorth params
        assert features.shape[1] == expected_features
    
    def test_single_trial(self, single_trial_data):
        """Test single trial extraction."""
        td = TimeDomainExtractor(features=['variance', 'rms'])
        td.initialize({'sampling_rate': 250})
        features = td.extract(single_trial_data)
        
        assert features.ndim == 1


# =============================================================================
# FEATURE EXTRACTOR FACTORY TESTS
# =============================================================================

class TestFeatureExtractorFactory:
    """Test suite for Feature Extractor Factory."""
    
    def test_create_csp(self):
        """Test CSP creation via factory."""
        csp = FeatureExtractorFactory.create('csp', n_components=6)
        assert csp.name == 'csp'
        assert csp._n_components == 6
    
    def test_create_band_power(self):
        """Test band power creation via factory."""
        bp = FeatureExtractorFactory.create('band_power', config={'sampling_rate': 250})
        assert bp.name == 'band_power'
    
    def test_create_time_domain(self):
        """Test time domain creation via factory."""
        td = FeatureExtractorFactory.create('time_domain', config={'sampling_rate': 250})
        assert td.name == 'time_domain'
    
    def test_list_extractors(self):
        """Test listing available extractors."""
        extractors = FeatureExtractorFactory.list_available()
        assert 'csp' in extractors
        assert 'band_power' in extractors
        assert 'time_domain' in extractors
    
    def test_invalid_extractor(self):
        """Test creating invalid extractor raises error."""
        with pytest.raises(ValueError):
            FeatureExtractorFactory.create('invalid_extractor')


# =============================================================================
# FEATURE EXTRACTION PIPELINE TESTS
# =============================================================================

class TestFeatureExtractionPipeline:
    """Test suite for Feature Extraction Pipeline."""
    
    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = FeatureExtractionPipeline()
        assert pipeline.mode == 'concatenate'
    
    def test_add_extractors(self):
        """Test adding extractors to pipeline."""
        pipeline = FeatureExtractionPipeline()
        
        csp = CSPExtractor(n_components=4)
        bp = BandPowerExtractor()
        
        pipeline.add_extractor(csp, 'csp')
        pipeline.add_extractor(bp, 'band_power')
        
        assert len(pipeline._extractors) == 2
    
    def test_pipeline_fit_extract(self, binary_eeg_data):
        """Test pipeline fit and extract."""
        X, y = binary_eeg_data
        
        pipeline = FeatureExtractionPipeline()
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        
        bp = BandPowerExtractor(bands={'mu': (8, 12)}, average_channels=True)
        bp.initialize({'sampling_rate': 250})
        
        pipeline.add_extractor(csp, 'csp')
        pipeline.add_extractor(bp, 'band_power')
        
        features = pipeline.fit_extract(X, y)
        
        assert features.shape[0] == len(X)
        assert features.shape[1] == 4 + 1  # CSP features + 1 band power
    
    def test_pipeline_save_load(self, binary_eeg_data):
        """Test pipeline save and load."""
        X, y = binary_eeg_data
        
        pipeline = FeatureExtractionPipeline()
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        pipeline.add_extractor(csp, 'csp')
        
        features_original = pipeline.fit_extract(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pipeline.save(f.name)
            
            # Create and load pipeline
            pipeline_loaded = FeatureExtractionPipeline.load(f.name)
            
            features_loaded = pipeline_loaded.extract(X)
            
            np.testing.assert_array_almost_equal(features_original, features_loaded)
            
            os.unlink(f.name)
    
    def test_motor_imagery_pipeline(self, binary_eeg_data):
        """Test motor imagery pipeline convenience function."""
        X, y = binary_eeg_data
        
        pipeline = create_motor_imagery_pipeline(
            n_csp_components=4,
            sampling_rate=250
        )
        
        features = pipeline.fit_extract(X, y)
        
        assert features.shape[0] == len(X)
        assert features.shape[1] > 0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_create_csp_extractor(self):
        """Test CSP extractor convenience function."""
        csp = create_csp_extractor(n_components=6, sampling_rate=250)
        assert csp.name == 'csp'
        assert csp._n_components == 6
    
    def test_create_band_power_extractor(self):
        """Test band power extractor convenience function."""
        bp = create_band_power_extractor(
            bands={'mu': (8, 12), 'beta': (12, 30)},
            sampling_rate=250
        )
        assert bp.name == 'band_power'


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        
        with pytest.raises(ValueError):
            csp.fit(np.array([]), np.array([]))
    
    def test_mismatched_shapes(self, binary_eeg_data):
        """Test handling of mismatched X and y shapes."""
        X, y = binary_eeg_data
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        
        with pytest.raises(ValueError):
            csp.fit(X, y[:-10])  # Mismatched lengths
    
    def test_unfitted_extract(self, binary_eeg_data):
        """Test extraction without fitting raises error."""
        X, y = binary_eeg_data
        
        csp = CSPExtractor(n_components=4)
        csp.initialize({'sampling_rate': 250})
        
        with pytest.raises(RuntimeError):
            csp.extract(X)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
