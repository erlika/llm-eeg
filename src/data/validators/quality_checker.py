"""
Signal Quality Checker
======================

This module provides signal quality assessment for EEG data.
It analyzes various aspects of signal quality that may affect
classification performance.

Quality Metrics:
---------------
1. Signal-to-Noise Ratio (SNR)
2. Line noise level (50/60 Hz contamination)
3. Artifact presence (amplitude-based)
4. Flatline detection (dead channels)
5. Correlation between channels (for artifact detection)

Use Cases:
---------
- Pre-processing quality check
- Artifact trial rejection
- Channel quality assessment
- Data quality reporting
- Adaptive preprocessing selection (APA agent)

BCI Competition IV-2a Considerations:
------------------------------------
- Artifact trials are marked with event code 1023
- EOG channels (23-25) should be excluded from EEG quality checks
- High-quality trials are essential for accurate classification

Usage Example:
    ```python
    from src.data.validators import QualityChecker
    
    checker = QualityChecker()
    checker.initialize({'sampling_rate': 250})
    
    # Get comprehensive quality report
    report = checker.assess_quality(eeg_data)
    print(f"Overall quality: {report['overall_score']:.2f}")
    
    # Check specific metrics
    snr = checker.compute_snr(signals)
    line_noise = checker.compute_line_noise_level(signals)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats
import logging

from src.core.types.eeg_data import EEGData, TrialData


# Configure module logger
logger = logging.getLogger(__name__)


class QualityChecker:
    """
    Signal quality assessment for EEG data.
    
    This class provides methods to assess various aspects of
    EEG signal quality, useful for preprocessing decisions
    and artifact detection.
    
    Attributes:
        _sampling_rate (float): Signal sampling rate
        _artifact_threshold (float): Amplitude threshold for artifact detection (µV)
        _flatline_threshold (float): Standard deviation threshold for flatline (µV)
        _line_freq (float): Power line frequency (50 or 60 Hz)
    
    Quality Score Components:
        - SNR: Signal-to-noise ratio (higher is better)
        - Line noise: 50/60 Hz contamination (lower is better)
        - Artifact ratio: Proportion of artifact samples (lower is better)
        - Flatline ratio: Proportion of flatline channels (lower is better)
    
    Example:
        >>> checker = QualityChecker()
        >>> checker.initialize({'sampling_rate': 250, 'line_freq': 50})
        >>> report = checker.assess_quality(eeg_data)
    """
    
    # Default thresholds
    DEFAULT_ARTIFACT_THRESHOLD: float = 100.0  # µV
    DEFAULT_FLATLINE_THRESHOLD: float = 0.5    # µV (very low std)
    DEFAULT_LINE_FREQ: float = 50.0            # Hz (European)
    
    def __init__(self):
        """Initialize the quality checker."""
        # Configuration
        self._sampling_rate: float = 250.0
        self._artifact_threshold: float = self.DEFAULT_ARTIFACT_THRESHOLD
        self._flatline_threshold: float = self.DEFAULT_FLATLINE_THRESHOLD
        self._line_freq: float = self.DEFAULT_LINE_FREQ
        
        # State
        self._is_initialized: bool = False
        
        logger.debug("QualityChecker instantiated")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the quality checker with configuration.
        
        Args:
            config: Configuration dictionary with keys:
                - 'sampling_rate' (float, required): Signal sampling rate in Hz
                - 'artifact_threshold' (float, optional): Artifact amplitude threshold
                - 'flatline_threshold' (float, optional): Flatline std threshold
                - 'line_freq' (float, optional): Power line frequency (50 or 60)
        
        Raises:
            ValueError: If required config is missing
        """
        logger.info("Initializing QualityChecker")
        
        if 'sampling_rate' not in config:
            raise ValueError("sampling_rate is required")
        
        self._sampling_rate = float(config['sampling_rate'])
        self._artifact_threshold = float(
            config.get('artifact_threshold', self.DEFAULT_ARTIFACT_THRESHOLD)
        )
        self._flatline_threshold = float(
            config.get('flatline_threshold', self.DEFAULT_FLATLINE_THRESHOLD)
        )
        self._line_freq = float(
            config.get('line_freq', self.DEFAULT_LINE_FREQ)
        )
        
        self._is_initialized = True
        logger.info(
            f"QualityChecker initialized: fs={self._sampling_rate}Hz, "
            f"line_freq={self._line_freq}Hz"
        )
    
    # =========================================================================
    # MAIN QUALITY ASSESSMENT
    # =========================================================================
    
    def assess_quality(
        self,
        data: Union[EEGData, TrialData, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment.
        
        Analyzes multiple quality metrics and returns a detailed report.
        
        Args:
            data: EEG data to assess
        
        Returns:
            Dict containing:
                - 'overall_score': Combined quality score (0-1)
                - 'snr_db': Signal-to-noise ratio in dB
                - 'line_noise_ratio': Line noise power ratio
                - 'artifact_ratio': Proportion of artifact samples
                - 'flatline_channels': List of potential flatline channels
                - 'channel_quality': Per-channel quality scores
                - 'recommendations': List of recommendations
        
        Example:
            >>> report = checker.assess_quality(eeg_data)
            >>> print(f"Quality: {report['overall_score']:.2%}")
        """
        if not self._is_initialized:
            raise RuntimeError(
                "QualityChecker not initialized. Call initialize() first."
            )
        
        # Extract signals
        if isinstance(data, (EEGData, TrialData)):
            signals = data.signals
        else:
            signals = data
        
        # Ensure 2D
        if signals.ndim == 3:
            # Average across trials for overall assessment
            signals = np.mean(signals, axis=0)
        
        # Compute individual metrics
        snr_db = self.compute_snr(signals)
        line_noise_ratio = self.compute_line_noise_level(signals)
        artifact_ratio = self.compute_artifact_ratio(signals)
        flatline_channels = self.detect_flatline_channels(signals)
        channel_quality = self.compute_channel_quality(signals)
        
        # Compute overall score (weighted combination)
        overall_score = self._compute_overall_score(
            snr_db=snr_db,
            line_noise_ratio=line_noise_ratio,
            artifact_ratio=artifact_ratio,
            flatline_ratio=len(flatline_channels) / signals.shape[0]
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            snr_db=snr_db,
            line_noise_ratio=line_noise_ratio,
            artifact_ratio=artifact_ratio,
            flatline_channels=flatline_channels
        )
        
        report = {
            'overall_score': overall_score,
            'snr_db': snr_db,
            'line_noise_ratio': line_noise_ratio,
            'artifact_ratio': artifact_ratio,
            'flatline_channels': flatline_channels,
            'channel_quality': channel_quality,
            'recommendations': recommendations,
            'n_channels': signals.shape[0],
            'n_samples': signals.shape[1]
        }
        
        logger.info(
            f"Quality assessment complete: score={overall_score:.2f}, "
            f"SNR={snr_db:.1f}dB"
        )
        
        return report
    
    # =========================================================================
    # INDIVIDUAL QUALITY METRICS
    # =========================================================================
    
    def compute_snr(
        self,
        signals: np.ndarray,
        signal_band: Tuple[float, float] = (8, 30),
        noise_band: Tuple[float, float] = (1, 4)
    ) -> float:
        """
        Compute Signal-to-Noise Ratio in dB.
        
        SNR is computed as the ratio of power in the signal band
        (8-30 Hz for motor imagery) to power in the noise band.
        
        Args:
            signals: EEG signals (channels, samples)
            signal_band: Frequency band for signal (default: 8-30 Hz)
            noise_band: Frequency band for noise (default: 1-4 Hz)
        
        Returns:
            float: SNR in dB (higher is better)
        
        Example:
            >>> snr = checker.compute_snr(signals)
            >>> print(f"SNR: {snr:.1f} dB")
        """
        # Compute power spectral density
        freqs, psd = scipy_signal.welch(
            signals,
            fs=self._sampling_rate,
            nperseg=min(256, signals.shape[1]),
            axis=1
        )
        
        # Average PSD across channels
        psd_mean = np.mean(psd, axis=0)
        
        # Find frequency indices
        signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
        noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
        
        # Compute band powers
        signal_power = np.mean(psd_mean[signal_mask])
        noise_power = np.mean(psd_mean[noise_mask])
        
        # Avoid division by zero
        if noise_power < 1e-10:
            noise_power = 1e-10
        
        # SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        return float(snr_db)
    
    def compute_line_noise_level(
        self,
        signals: np.ndarray
    ) -> float:
        """
        Compute power line noise contamination level.
        
        Measures the power at the line frequency (50/60 Hz) relative
        to surrounding frequencies.
        
        Args:
            signals: EEG signals (channels, samples)
        
        Returns:
            float: Line noise ratio (lower is better, <0.1 is good)
        
        Example:
            >>> noise_ratio = checker.compute_line_noise_level(signals)
            >>> if noise_ratio > 0.2:
            ...     print("Significant line noise detected")
        """
        # Compute PSD
        freqs, psd = scipy_signal.welch(
            signals,
            fs=self._sampling_rate,
            nperseg=min(256, signals.shape[1]),
            axis=1
        )
        
        # Average across channels
        psd_mean = np.mean(psd, axis=0)
        
        # Find line frequency power
        line_idx = np.argmin(np.abs(freqs - self._line_freq))
        line_power = psd_mean[line_idx]
        
        # Find surrounding frequencies (5 Hz on each side, excluding notch)
        surround_mask = (
            ((freqs >= self._line_freq - 10) & (freqs < self._line_freq - 2)) |
            ((freqs > self._line_freq + 2) & (freqs <= self._line_freq + 10))
        )
        surround_power = np.mean(psd_mean[surround_mask])
        
        # Ratio
        if surround_power < 1e-10:
            surround_power = 1e-10
        
        noise_ratio = line_power / surround_power
        
        # Normalize to 0-1 range (cap at 10x)
        noise_ratio = min(noise_ratio / 10, 1.0)
        
        return float(noise_ratio)
    
    def compute_artifact_ratio(
        self,
        signals: np.ndarray
    ) -> float:
        """
        Compute proportion of samples exceeding artifact threshold.
        
        Args:
            signals: EEG signals (channels, samples)
        
        Returns:
            float: Artifact ratio (0-1, lower is better)
        
        Example:
            >>> artifact_ratio = checker.compute_artifact_ratio(signals)
            >>> if artifact_ratio > 0.1:
            ...     print("High artifact contamination")
        """
        # Count samples exceeding threshold
        artifact_mask = np.abs(signals) > self._artifact_threshold
        artifact_count = np.sum(artifact_mask)
        total_samples = signals.size
        
        ratio = artifact_count / total_samples if total_samples > 0 else 0
        
        return float(ratio)
    
    def detect_flatline_channels(
        self,
        signals: np.ndarray
    ) -> List[int]:
        """
        Detect channels with very low variance (potential flatlines).
        
        Flatline channels may indicate electrode disconnection or
        hardware issues.
        
        Args:
            signals: EEG signals (channels, samples)
        
        Returns:
            List[int]: Indices of potential flatline channels
        
        Example:
            >>> flatlines = checker.detect_flatline_channels(signals)
            >>> if flatlines:
            ...     print(f"Flatline channels detected: {flatlines}")
        """
        n_channels = signals.shape[0]
        flatlines = []
        
        for ch in range(n_channels):
            std = np.std(signals[ch])
            if std < self._flatline_threshold:
                flatlines.append(ch)
        
        return flatlines
    
    def compute_channel_quality(
        self,
        signals: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-channel quality scores.
        
        Each channel gets a score from 0 (poor) to 1 (excellent).
        
        Args:
            signals: EEG signals (channels, samples)
        
        Returns:
            np.ndarray: Quality scores for each channel (0-1)
        
        Example:
            >>> quality = checker.compute_channel_quality(signals)
            >>> bad_channels = np.where(quality < 0.5)[0]
        """
        n_channels = signals.shape[0]
        quality = np.ones(n_channels)
        
        for ch in range(n_channels):
            ch_signal = signals[ch]
            
            # Factor 1: Artifact ratio (lower is better)
            artifact_ratio = np.mean(np.abs(ch_signal) > self._artifact_threshold)
            
            # Factor 2: Standard deviation (very low or very high is bad)
            std = np.std(ch_signal)
            std_score = 1.0
            if std < self._flatline_threshold:
                std_score = 0.1  # Flatline
            elif std > self._artifact_threshold * 2:
                std_score = 0.5  # High noise
            
            # Factor 3: Kurtosis (very high indicates artifacts)
            kurtosis = scipy_stats.kurtosis(ch_signal) if len(ch_signal) > 4 else 0
            kurtosis_score = 1.0 - min(abs(kurtosis) / 10, 0.5)
            
            # Combine factors
            quality[ch] = (
                (1 - artifact_ratio) * 0.4 +
                std_score * 0.4 +
                kurtosis_score * 0.2
            )
        
        return quality
    
    # =========================================================================
    # TRIAL-LEVEL QUALITY
    # =========================================================================
    
    def assess_trial_quality(
        self,
        trial: Union[TrialData, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Assess quality of a single trial.
        
        Useful for trial rejection in training/evaluation.
        
        Args:
            trial: Trial data to assess
        
        Returns:
            Dict with trial quality metrics
        
        Example:
            >>> quality = checker.assess_trial_quality(trial)
            >>> if quality['is_clean']:
            ...     # Use trial for training
        """
        if isinstance(trial, TrialData):
            signals = trial.signals
        else:
            signals = trial
        
        # Ensure 2D
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        
        # Compute metrics
        artifact_ratio = self.compute_artifact_ratio(signals)
        max_amplitude = float(np.max(np.abs(signals)))
        std = float(np.std(signals))
        
        # Determine if clean
        is_clean = (
            artifact_ratio < 0.1 and
            max_amplitude < self._artifact_threshold * 1.5 and
            std > self._flatline_threshold
        )
        
        return {
            'is_clean': is_clean,
            'artifact_ratio': artifact_ratio,
            'max_amplitude': max_amplitude,
            'std': std,
            'quality_score': 1.0 - artifact_ratio if is_clean else 0.0
        }
    
    def filter_clean_trials(
        self,
        trials: List[TrialData],
        threshold: float = 0.1
    ) -> Tuple[List[TrialData], List[int]]:
        """
        Filter trials to keep only clean ones.
        
        Args:
            trials: List of trials to filter
            threshold: Maximum artifact ratio for clean trials
        
        Returns:
            Tuple of (clean_trials, rejected_indices)
        
        Example:
            >>> clean, rejected = checker.filter_clean_trials(trials)
            >>> print(f"Kept {len(clean)}/{len(trials)} trials")
        """
        clean_trials = []
        rejected_indices = []
        
        for i, trial in enumerate(trials):
            quality = self.assess_trial_quality(trial)
            if quality['artifact_ratio'] < threshold:
                clean_trials.append(trial)
            else:
                rejected_indices.append(i)
        
        logger.info(
            f"Trial filtering: {len(clean_trials)}/{len(trials)} clean "
            f"({len(rejected_indices)} rejected)"
        )
        
        return clean_trials, rejected_indices
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _compute_overall_score(
        self,
        snr_db: float,
        line_noise_ratio: float,
        artifact_ratio: float,
        flatline_ratio: float
    ) -> float:
        """
        Compute overall quality score from individual metrics.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            line_noise_ratio: Line noise contamination (0-1)
            artifact_ratio: Artifact sample ratio (0-1)
            flatline_ratio: Flatline channel ratio (0-1)
        
        Returns:
            float: Overall quality score (0-1)
        """
        # Convert SNR to 0-1 score (10dB = 0.5, 20dB = 1.0, 0dB = 0)
        snr_score = np.clip(snr_db / 20, 0, 1)
        
        # Convert line noise to score (inverted, lower is better)
        line_score = 1 - line_noise_ratio
        
        # Convert artifact ratio to score (inverted)
        artifact_score = 1 - np.clip(artifact_ratio * 5, 0, 1)  # 20% = 0
        
        # Convert flatline ratio to score (inverted)
        flatline_score = 1 - flatline_ratio
        
        # Weighted combination
        overall = (
            snr_score * 0.3 +
            line_score * 0.2 +
            artifact_score * 0.3 +
            flatline_score * 0.2
        )
        
        return float(np.clip(overall, 0, 1))
    
    def _generate_recommendations(
        self,
        snr_db: float,
        line_noise_ratio: float,
        artifact_ratio: float,
        flatline_channels: List[int]
    ) -> List[str]:
        """
        Generate recommendations based on quality metrics.
        
        Args:
            snr_db: Signal-to-noise ratio
            line_noise_ratio: Line noise level
            artifact_ratio: Artifact ratio
            flatline_channels: List of flatline channels
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if snr_db < 5:
            recommendations.append(
                "Low SNR detected. Consider checking electrode impedances "
                "or recording environment."
            )
        
        if line_noise_ratio > 0.3:
            recommendations.append(
                f"High line noise ({self._line_freq} Hz). "
                "Apply notch filter before further processing."
            )
        
        if artifact_ratio > 0.1:
            recommendations.append(
                f"High artifact contamination ({artifact_ratio:.1%}). "
                "Consider artifact rejection or ICA cleaning."
            )
        
        if flatline_channels:
            recommendations.append(
                f"Flatline detected in channels {flatline_channels}. "
                "Check electrode connections or exclude these channels."
            )
        
        if not recommendations:
            recommendations.append("Signal quality is good. No issues detected.")
        
        return recommendations
    
    def __repr__(self) -> str:
        """String representation."""
        if self._is_initialized:
            return (
                f"QualityChecker(fs={self._sampling_rate}, "
                f"line_freq={self._line_freq})"
            )
        return "QualityChecker(not initialized)"
