"""
Unit Tests for Preprocessing Components
=======================================

This module contains unit tests for the preprocessing pipeline and steps.

Test Coverage:
- BandpassFilter
- NotchFilter
- Normalization
- PreprocessingPipeline

Author: EEG-BCI Framework
Date: 2024
"""

import pytest
import numpy as np
from scipy import signal as scipy_signal

# Import modules to test
from src.preprocessing import (
    BandpassFilter,
    NotchFilter,
    Normalization,
    PreprocessingPipeline,
    create_standard_pipeline,
)
from src.core.types.eeg_data import EEGData


class TestBandpassFilter:
    """Test cases for BandpassFilter."""
    
    def test_initialization(self):
        """Test filter initialization."""
        bp = BandpassFilter()
        bp.initialize({
            'sampling_rate': 250,
            'low_freq': 8.0,
            'high_freq': 30.0,
            'filter_order': 5
        })
        
        params = bp.get_params()
        assert params['low_freq'] == 8.0
        assert params['high_freq'] == 30.0
        assert params['filter_order'] == 5
        assert params['sampling_rate'] == 250
    
    def test_properties(self):
        """Test filter properties."""
        bp = BandpassFilter()
        
        assert bp.name == 'bandpass_filter'
        assert bp.is_trainable == False
    
    def test_process_2d_array(self):
        """Test filtering 2D array (channels, samples)."""
        bp = BandpassFilter()
        bp.initialize({'sampling_rate': 250, 'low_freq': 8, 'high_freq': 30})
        
        # Create test signal with multiple frequency components
        n_channels = 3
        n_samples = 1000
        t = np.arange(n_samples) / 250.0
        
        # Signal: 5 Hz (should be filtered) + 15 Hz (should pass) + 50 Hz (should be filtered)
        signals = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            signals[ch] = (
                np.sin(2 * np.pi * 5 * t) +    # Below passband
                np.sin(2 * np.pi * 15 * t) +   # In passband
                np.sin(2 * np.pi * 50 * t)     # Above passband
            )
        
        filtered = bp.process(signals)
        
        # Check output shape
        assert filtered.shape == signals.shape
        
        # Check that 15 Hz is preserved (roughly)
        # Power spectrum should show peak at 15 Hz
        freqs, psd = scipy_signal.welch(filtered[0], fs=250, nperseg=256)
        peak_freq = freqs[np.argmax(psd)]
        assert 10 < peak_freq < 20  # Peak should be around 15 Hz
    
    def test_process_3d_array(self):
        """Test filtering 3D array (trials, channels, samples)."""
        bp = BandpassFilter()
        bp.initialize({'sampling_rate': 250, 'low_freq': 8, 'high_freq': 30})
        
        signals = np.random.randn(10, 22, 1000)  # 10 trials
        filtered = bp.process(signals)
        
        assert filtered.shape == signals.shape
    
    def test_process_eegdata(self):
        """Test filtering EEGData object."""
        bp = BandpassFilter()
        bp.initialize({'sampling_rate': 250, 'low_freq': 8, 'high_freq': 30})
        
        eeg_data = EEGData(
            signals=np.random.randn(22, 1000),
            sampling_rate=250.0,
            channel_names=['Ch' + str(i) for i in range(22)]
        )
        
        filtered = bp.process(eeg_data)
        
        assert isinstance(filtered, EEGData)
        assert filtered.shape == eeg_data.shape
        assert filtered.metadata.get('bandpass_filtered') == True
    
    def test_invalid_frequencies(self):
        """Test validation of invalid frequency parameters."""
        bp = BandpassFilter()
        
        # high_freq >= Nyquist
        with pytest.raises(ValueError):
            bp.initialize({
                'sampling_rate': 250,
                'low_freq': 8,
                'high_freq': 130  # >= 125 Hz Nyquist
            })
        
        # low_freq >= high_freq
        with pytest.raises(ValueError):
            bp.initialize({
                'sampling_rate': 250,
                'low_freq': 30,
                'high_freq': 8
            })
    
    def test_get_passband(self):
        """Test getting passband frequencies."""
        bp = BandpassFilter()
        bp.initialize({'sampling_rate': 250, 'low_freq': 8, 'high_freq': 30})
        
        low, high = bp.get_passband()
        assert low == 8
        assert high == 30


class TestNotchFilter:
    """Test cases for NotchFilter."""
    
    def test_initialization(self):
        """Test notch filter initialization."""
        notch = NotchFilter()
        notch.initialize({
            'sampling_rate': 250,
            'notch_freq': 50.0,
            'quality_factor': 30
        })
        
        params = notch.get_params()
        assert params['notch_freq'] == 50.0
        assert params['quality_factor'] == 30
    
    def test_properties(self):
        """Test notch filter properties."""
        notch = NotchFilter()
        
        assert notch.name == 'notch_filter'
        assert notch.is_trainable == False
    
    def test_line_noise_removal(self):
        """Test removal of line noise."""
        notch = NotchFilter()
        notch.initialize({
            'sampling_rate': 250,
            'notch_freq': 50.0,
            'quality_factor': 30
        })
        
        # Create signal with 50 Hz noise
        n_samples = 2000
        t = np.arange(n_samples) / 250.0
        
        clean_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
        noise = 0.5 * np.sin(2 * np.pi * 50 * t)   # 50 Hz noise
        noisy_signal = clean_signal + noise
        
        signals = noisy_signal.reshape(1, -1)
        filtered = notch.process(signals)
        
        # Check that 50 Hz is attenuated
        freqs, psd_before = scipy_signal.welch(signals[0], fs=250, nperseg=256)
        freqs, psd_after = scipy_signal.welch(filtered[0], fs=250, nperseg=256)
        
        idx_50 = np.argmin(np.abs(freqs - 50))
        
        # 50 Hz power should be reduced
        assert psd_after[idx_50] < psd_before[idx_50] * 0.5
    
    def test_harmonic_removal(self):
        """Test removal of harmonics."""
        notch = NotchFilter()
        notch.initialize({
            'sampling_rate': 250,
            'notch_freq': 50.0,
            'remove_harmonics': True,
            'max_harmonic': 2
        })
        
        frequencies = notch.get_notch_frequencies()
        
        assert 50 in frequencies
        assert 100 in frequencies  # 2nd harmonic


class TestNormalization:
    """Test cases for Normalization."""
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        norm = Normalization()
        norm.initialize({'method': 'zscore', 'axis': 'global'})
        
        signals = np.random.randn(3, 100) * 50 + 100  # Mean ~100, std ~50
        normalized = norm.process(signals)
        
        # After z-score, should have mean ~0, std ~1
        assert np.abs(np.mean(normalized)) < 0.1
        assert np.abs(np.std(normalized) - 1) < 0.1
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        norm = Normalization()
        norm.initialize({
            'method': 'minmax',
            'axis': 'global',
            'feature_range': (0, 1)
        })
        
        signals = np.random.randn(3, 100)
        normalized = norm.process(signals)
        
        # Should be in [0, 1] range
        assert np.min(normalized) >= 0 - 1e-6
        assert np.max(normalized) <= 1 + 1e-6
    
    def test_channel_wise_normalization(self):
        """Test channel-wise normalization."""
        norm = Normalization()
        norm.initialize({'method': 'zscore', 'axis': 'channel'})
        
        # Create signals with different channel scales
        signals = np.zeros((3, 1000))
        signals[0] = np.random.randn(1000) * 10 + 50   # Mean 50, std 10
        signals[1] = np.random.randn(1000) * 100 + 200  # Mean 200, std 100
        signals[2] = np.random.randn(1000) * 1          # Mean 0, std 1
        
        normalized = norm.process(signals)
        
        # Each channel should have mean ~0, std ~1
        for ch in range(3):
            assert np.abs(np.mean(normalized[ch])) < 0.1
            assert np.abs(np.std(normalized[ch]) - 1) < 0.1
    
    def test_properties(self):
        """Test normalization properties."""
        norm = Normalization()
        
        assert norm.name == 'normalization'
        assert norm.is_trainable == False
    
    def test_invalid_method(self):
        """Test validation of invalid method."""
        norm = Normalization()
        
        with pytest.raises(ValueError):
            norm.initialize({'method': 'invalid_method'})


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline."""
    
    def test_empty_pipeline(self):
        """Test empty pipeline returns data unchanged."""
        pipeline = PreprocessingPipeline()
        pipeline.initialize({})
        
        signals = np.random.randn(22, 1000)
        output = pipeline.process(signals)
        
        np.testing.assert_array_equal(signals, output)
    
    def test_single_step_pipeline(self):
        """Test pipeline with single step."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(
            Normalization(),
            {'method': 'zscore', 'axis': 'global'}
        )
        pipeline.initialize({})
        
        signals = np.random.randn(22, 1000) * 50 + 100
        output = pipeline.process(signals)
        
        # Should be normalized
        assert np.abs(np.mean(output)) < 0.1
    
    def test_multi_step_pipeline(self):
        """Test pipeline with multiple steps."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(
            NotchFilter(),
            {'notch_freq': 50, 'quality_factor': 30}
        )
        pipeline.add_step(
            BandpassFilter(),
            {'low_freq': 8, 'high_freq': 30}
        )
        pipeline.add_step(
            Normalization(),
            {'method': 'zscore'}
        )
        pipeline.initialize({'sampling_rate': 250})
        
        signals = np.random.randn(22, 1000)
        output = pipeline.process(signals)
        
        assert output.shape == signals.shape
    
    def test_step_ordering(self):
        """Test that steps are executed in order."""
        pipeline = PreprocessingPipeline()
        
        pipeline.add_step(NotchFilter(), name='step1')
        pipeline.add_step(BandpassFilter(), name='step2')
        pipeline.add_step(Normalization(), name='step3')
        
        steps = pipeline.get_steps()
        assert steps == ['step1', 'step2', 'step3']
    
    def test_remove_step(self):
        """Test removing a step from pipeline."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(NotchFilter(), name='notch')
        pipeline.add_step(BandpassFilter(), name='bandpass')
        
        pipeline.remove_step('notch')
        
        steps = pipeline.get_steps()
        assert 'notch' not in steps
        assert 'bandpass' in steps
    
    def test_process_eegdata(self):
        """Test pipeline processing EEGData."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(Normalization(), {'method': 'zscore'})
        pipeline.initialize({'sampling_rate': 250})
        
        eeg_data = EEGData(
            signals=np.random.randn(22, 1000),
            sampling_rate=250.0,
            channel_names=['Ch' + str(i) for i in range(22)]
        )
        
        output = pipeline.process(eeg_data)
        
        assert isinstance(output, EEGData)
    
    def test_timing_tracking(self):
        """Test execution time tracking."""
        pipeline = PreprocessingPipeline(timing=True)
        pipeline.add_step(Normalization(), name='norm')
        pipeline.initialize({'sampling_rate': 250})
        
        signals = np.random.randn(22, 1000)
        pipeline.process(signals)
        
        times = pipeline.get_execution_times()
        assert 'norm' in times
        assert times['norm'] > 0


class TestCreateStandardPipeline:
    """Test cases for create_standard_pipeline factory."""
    
    def test_default_pipeline(self):
        """Test creating default pipeline."""
        pipeline = create_standard_pipeline(sampling_rate=250)
        
        steps = pipeline.get_steps()
        assert 'notch' in steps
        assert 'bandpass' in steps
        assert 'normalize' in steps
    
    def test_custom_frequencies(self):
        """Test custom frequency parameters."""
        pipeline = create_standard_pipeline(
            sampling_rate=250,
            notch_freq=60,
            bandpass_low=4,
            bandpass_high=40
        )
        
        # Get bandpass params
        bp, _ = pipeline.get_step('bandpass')
        params = bp.get_params()
        
        assert params['low_freq'] == 4
        assert params['high_freq'] == 40
    
    def test_no_normalization(self):
        """Test pipeline without normalization."""
        pipeline = create_standard_pipeline(
            sampling_rate=250,
            normalize=False
        )
        
        steps = pipeline.get_steps()
        assert 'normalize' not in steps


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
