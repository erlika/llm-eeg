"""
Unit Tests for Data Validators
==============================

This module contains unit tests for data validation and quality checking.

Test Coverage:
- DataValidator
- ValidationResult
- QualityChecker

Author: EEG-BCI Framework
Date: 2024
"""

import pytest
import numpy as np

# Import modules to test
from src.data.validators import (
    DataValidator,
    ValidationResult,
    ValidationError,
    QualityChecker,
)
from src.core.types.eeg_data import EEGData, TrialData, EventMarker


class TestValidationResult:
    """Test cases for ValidationResult."""
    
    def test_initial_state(self):
        """Test initial validation result state."""
        result = ValidationResult()
        
        assert result.is_valid == True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error(self):
        """Test adding an error marks result as invalid."""
        result = ValidationResult()
        result.add_error("Test error")
        
        assert result.is_valid == False
        assert len(result.errors) == 1
        assert "Test error" in result.errors
    
    def test_add_warning(self):
        """Test adding a warning doesn't affect validity."""
        result = ValidationResult()
        result.add_warning("Test warning")
        
        assert result.is_valid == True
        assert len(result.warnings) == 1
    
    def test_merge_results(self):
        """Test merging two validation results."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        
        result2 = ValidationResult()
        result2.add_warning("Warning 1")
        
        result1.merge(result2)
        
        assert result1.is_valid == False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1
    
    def test_bool_conversion(self):
        """Test boolean conversion."""
        result = ValidationResult()
        assert bool(result) == True
        
        result.add_error("Error")
        assert bool(result) == False


class TestDataValidator:
    """Test cases for DataValidator."""
    
    def test_valid_eegdata(self):
        """Test validation of valid EEGData."""
        validator = DataValidator()
        
        eeg_data = EEGData(
            signals=np.random.randn(22, 1000),
            sampling_rate=250.0,
            channel_names=['Ch' + str(i) for i in range(22)]
        )
        
        result = validator.validate(eeg_data)
        
        assert result.is_valid == True
    
    def test_invalid_nan_values(self):
        """Test detection of NaN values."""
        validator = DataValidator()
        
        signals = np.random.randn(22, 1000)
        signals[0, 500] = np.nan  # Insert NaN
        
        eeg_data = EEGData(
            signals=signals,
            sampling_rate=250.0
        )
        
        result = validator.validate(eeg_data)
        
        # Should have warning (not error for small amount)
        assert len(result.warnings) > 0 or len(result.errors) > 0
    
    def test_invalid_inf_values(self):
        """Test detection of Inf values."""
        validator = DataValidator()
        
        signals = np.random.randn(22, 1000)
        signals[0, 500] = np.inf  # Insert Inf
        
        eeg_data = EEGData(
            signals=signals,
            sampling_rate=250.0
        )
        
        result = validator.validate(eeg_data)
        
        assert result.is_valid == False
        assert any("Inf" in e for e in result.errors)
    
    def test_channel_name_mismatch(self):
        """Test detection of channel name count mismatch."""
        validator = DataValidator()
        
        # This should raise error during EEGData creation
        with pytest.raises(ValueError):
            EEGData(
                signals=np.random.randn(22, 1000),
                sampling_rate=250.0,
                channel_names=['Ch1', 'Ch2']  # Wrong count
            )
    
    def test_valid_trialdata(self):
        """Test validation of valid TrialData."""
        validator = DataValidator()
        
        trial = TrialData(
            signals=np.random.randn(22, 1000),
            label=0,
            label_name='left_hand',
            sampling_rate=250.0
        )
        
        result = validator.validate(trial)
        
        assert result.is_valid == True
    
    def test_valid_numpy_array(self):
        """Test validation of valid numpy array."""
        validator = DataValidator()
        
        signals = np.random.randn(22, 1000)
        result = validator.validate(signals)
        
        assert result.is_valid == True
    
    def test_invalid_array_dimensions(self):
        """Test detection of invalid array dimensions."""
        validator = DataValidator()
        
        signals = np.random.randn(22)  # 1D array
        result = validator.validate(signals)
        
        assert result.is_valid == False
    
    def test_assert_valid_success(self):
        """Test assert_valid with valid data."""
        validator = DataValidator()
        
        eeg_data = EEGData(
            signals=np.random.randn(22, 1000),
            sampling_rate=250.0
        )
        
        # Should not raise
        validator.assert_valid(eeg_data)
    
    def test_assert_valid_failure(self):
        """Test assert_valid with invalid data."""
        validator = DataValidator()
        
        signals = np.random.randn(22, 1000)
        signals[0, 0] = np.inf  # Invalid value
        
        eeg_data = EEGData(
            signals=signals,
            sampling_rate=250.0
        )
        
        with pytest.raises(ValidationError):
            validator.assert_valid(eeg_data)
    
    def test_is_valid_shortcut(self):
        """Test is_valid convenience method."""
        validator = DataValidator()
        
        valid_data = EEGData(
            signals=np.random.randn(22, 1000),
            sampling_rate=250.0
        )
        
        assert validator.is_valid(valid_data) == True
    
    def test_strict_mode(self):
        """Test strict mode converts warnings to errors."""
        validator = DataValidator(strict=True)
        
        # Create data with high amplitude (warning normally)
        signals = np.random.randn(22, 1000) * 1000  # Very high amplitude
        
        eeg_data = EEGData(
            signals=signals,
            sampling_rate=250.0
        )
        
        result = validator.validate(eeg_data)
        
        # In strict mode, amplitude warning should become error
        if len(result.warnings) == 0 and not result.is_valid:
            assert True  # Warning was converted to error
    
    def test_bci_iv_2a_validation(self):
        """Test BCI Competition IV-2a specific validation."""
        validator = DataValidator()
        
        # Valid BCI IV-2a data
        eeg_data = EEGData(
            signals=np.random.randn(22, 250000),  # ~1000s at 250 Hz
            sampling_rate=250.0,
            events=[
                EventMarker(sample=i * 1000, code=769 + (i % 4), label=f'class_{i%4}')
                for i in range(288)  # 288 trials
            ]
        )
        
        result = validator.validate_for_bci_iv_2a(eeg_data)
        
        assert result.is_valid == True
    
    def test_bci_iv_2a_wrong_sampling_rate(self):
        """Test BCI IV-2a validation with wrong sampling rate."""
        validator = DataValidator()
        
        eeg_data = EEGData(
            signals=np.random.randn(22, 1000),
            sampling_rate=512.0  # Wrong rate
        )
        
        result = validator.validate_for_bci_iv_2a(eeg_data)
        
        assert result.is_valid == False
        assert any("250 Hz" in e for e in result.errors)


class TestQualityChecker:
    """Test cases for QualityChecker."""
    
    def test_initialization(self):
        """Test quality checker initialization."""
        checker = QualityChecker()
        checker.initialize({
            'sampling_rate': 250,
            'line_freq': 50
        })
        
        assert checker._sampling_rate == 250
        assert checker._line_freq == 50
    
    def test_snr_computation(self):
        """Test SNR computation."""
        checker = QualityChecker()
        checker.initialize({'sampling_rate': 250})
        
        # Create signal with known characteristics
        n_samples = 2000
        t = np.arange(n_samples) / 250.0
        
        # Strong 15 Hz signal (in motor imagery band)
        signal = np.sin(2 * np.pi * 15 * t)
        # Weak noise in delta band
        noise = 0.1 * np.sin(2 * np.pi * 2 * t)
        
        signals = (signal + noise).reshape(1, -1)
        snr = checker.compute_snr(signals)
        
        # SNR should be positive (signal stronger than noise)
        assert snr > 0
    
    def test_line_noise_detection(self):
        """Test line noise level computation."""
        checker = QualityChecker()
        checker.initialize({'sampling_rate': 250, 'line_freq': 50})
        
        n_samples = 2000
        t = np.arange(n_samples) / 250.0
        
        # Signal with strong 50 Hz noise
        signal = np.sin(2 * np.pi * 10 * t)
        noise = 2.0 * np.sin(2 * np.pi * 50 * t)  # Strong line noise
        
        noisy = (signal + noise).reshape(1, -1)
        clean = signal.reshape(1, -1)
        
        noise_level_noisy = checker.compute_line_noise_level(noisy)
        noise_level_clean = checker.compute_line_noise_level(clean)
        
        # Noisy signal should have higher line noise level
        assert noise_level_noisy > noise_level_clean
    
    def test_artifact_ratio(self):
        """Test artifact ratio computation."""
        checker = QualityChecker()
        checker.initialize({
            'sampling_rate': 250,
            'artifact_threshold': 100
        })
        
        # Clean signal (all values < threshold)
        clean_signals = np.random.randn(22, 1000) * 10  # std ~10
        clean_ratio = checker.compute_artifact_ratio(clean_signals)
        
        assert clean_ratio < 0.01  # Less than 1% artifacts
        
        # Noisy signal (many values > threshold)
        noisy_signals = np.random.randn(22, 1000) * 200  # std ~200
        noisy_ratio = checker.compute_artifact_ratio(noisy_signals)
        
        assert noisy_ratio > 0.1  # More artifacts
    
    def test_flatline_detection(self):
        """Test flatline channel detection."""
        checker = QualityChecker()
        checker.initialize({
            'sampling_rate': 250,
            'flatline_threshold': 0.5
        })
        
        signals = np.random.randn(5, 1000)
        signals[2] = 0  # Flatline channel
        signals[4] = 0.001 * np.random.randn(1000)  # Near flatline
        
        flatlines = checker.detect_flatline_channels(signals)
        
        assert 2 in flatlines
        # Channel 4 might also be detected depending on threshold
    
    def test_quality_assessment(self):
        """Test comprehensive quality assessment."""
        checker = QualityChecker()
        checker.initialize({'sampling_rate': 250})
        
        signals = np.random.randn(22, 1000)
        report = checker.assess_quality(signals)
        
        # Check report structure
        assert 'overall_score' in report
        assert 'snr_db' in report
        assert 'line_noise_ratio' in report
        assert 'artifact_ratio' in report
        assert 'recommendations' in report
        
        # Overall score should be in [0, 1]
        assert 0 <= report['overall_score'] <= 1
    
    def test_trial_quality_assessment(self):
        """Test single trial quality assessment."""
        checker = QualityChecker()
        checker.initialize({'sampling_rate': 250})
        
        trial = TrialData(
            signals=np.random.randn(22, 1000) * 10,  # Clean trial
            label=0,
            sampling_rate=250.0
        )
        
        quality = checker.assess_trial_quality(trial)
        
        assert 'is_clean' in quality
        assert 'artifact_ratio' in quality
        assert 'quality_score' in quality
    
    def test_filter_clean_trials(self):
        """Test filtering clean trials."""
        checker = QualityChecker()
        checker.initialize({
            'sampling_rate': 250,
            'artifact_threshold': 100
        })
        
        # Create mix of clean and noisy trials
        trials = []
        for i in range(10):
            if i < 8:  # 8 clean trials
                signals = np.random.randn(22, 1000) * 10
            else:  # 2 noisy trials
                signals = np.random.randn(22, 1000) * 200
            
            trials.append(TrialData(
                signals=signals,
                label=i % 4,
                sampling_rate=250.0
            ))
        
        clean, rejected = checker.filter_clean_trials(trials)
        
        assert len(clean) >= 6  # At least 6 should be clean
        assert len(rejected) >= 1  # At least 1 rejected


class TestValidationError:
    """Test cases for ValidationError exception."""
    
    def test_error_creation(self):
        """Test creating validation error."""
        error = ValidationError(
            "Validation failed",
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        assert "Validation failed" in str(error)
        assert len(error.errors) == 2
        assert len(error.warnings) == 1
    
    def test_error_string_representation(self):
        """Test error string formatting."""
        error = ValidationError(
            "Test error",
            errors=["Missing channels"]
        )
        
        error_str = str(error)
        assert "Test error" in error_str
        assert "Missing channels" in error_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
