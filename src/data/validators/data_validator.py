"""
EEG Data Validator
==================

This module provides validation functionality for EEG data objects.
It ensures data integrity, format correctness, and compatibility
with the BCI framework.

Validation Levels:
-----------------
1. Structure: Correct array shapes and types
2. Format: Valid channel names, sampling rate, events
3. Integrity: No NaN/Inf values, reasonable value ranges
4. Compatibility: Matches expected dataset specifications

BCI Competition IV-2a Validation:
--------------------------------
For the BCI Competition IV-2a dataset, validators check:
- 22 EEG channels (or 25 with EOG)
- 250 Hz sampling rate
- Valid event codes (768, 769, 770, 771, 772)
- Reasonable signal amplitude ranges

Usage Example:
    ```python
    from src.data.validators import DataValidator
    
    validator = DataValidator()
    
    # Validate EEGData object
    is_valid, errors = validator.validate(eeg_data)
    
    if not is_valid:
        for error in errors:
            print(f"Validation error: {error}")
    
    # Quick check with assertion
    validator.assert_valid(eeg_data)  # Raises ValidationError if invalid
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import logging

from src.core.types.eeg_data import EEGData, TrialData


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT CLASS
# =============================================================================

class ValidationResult:
    """
    Container for validation results.
    
    Attributes:
        is_valid (bool): Overall validation result
        errors (List[str]): List of error messages
        warnings (List[str]): List of warning messages
        info (Dict): Additional validation information
    """
    
    def __init__(self):
        """Initialize validation result."""
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}
    
    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.update(other.info)
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid
    
    def __repr__(self) -> str:
        """String representation."""
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"ValidationResult({status}, "
            f"errors={len(self.errors)}, "
            f"warnings={len(self.warnings)})"
        )


# =============================================================================
# VALIDATION ERROR
# =============================================================================

class ValidationError(Exception):
    """
    Exception raised when data validation fails.
    
    Attributes:
        errors (List[str]): List of validation errors
        warnings (List[str]): List of validation warnings
    """
    
    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ):
        """Initialize validation error."""
        super().__init__(message)
        self.errors = errors or []
        self.warnings = warnings or []
    
    def __str__(self) -> str:
        """String representation with details."""
        base = super().__str__()
        if self.errors:
            base += "\nErrors:\n" + "\n".join(f"  - {e}" for e in self.errors)
        return base


# =============================================================================
# DATA VALIDATOR
# =============================================================================

class DataValidator:
    """
    Comprehensive validator for EEG data.
    
    This class provides validation methods for EEGData and TrialData objects,
    checking structure, format, integrity, and compatibility.
    
    Attributes:
        _config (Dict): Validation configuration
        _strict (bool): Enable strict mode (warnings become errors)
    
    Validation Categories:
        - Structure: Array dimensions, shapes, types
        - Values: NaN, Inf, amplitude ranges
        - Format: Channel names, sampling rate, events
        - Compatibility: Dataset-specific requirements
    
    Example:
        >>> validator = DataValidator(strict=False)
        >>> result = validator.validate(eeg_data)
        >>> print(result.is_valid)
        True
    """
    
    # Default expected values for BCI Competition IV-2a
    DEFAULT_SAMPLING_RATE: float = 250.0
    DEFAULT_N_EEG_CHANNELS: int = 22
    DEFAULT_N_ALL_CHANNELS: int = 25  # Including EOG
    DEFAULT_AMPLITUDE_RANGE: Tuple[float, float] = (-500, 500)  # µV
    
    # Valid event codes for BCI Competition IV-2a
    VALID_EVENT_CODES: set = {768, 769, 770, 771, 772, 783, 1023, 32766, 276, 277, 1072}
    
    # Motor imagery class codes
    MI_CLASS_CODES: set = {769, 770, 771, 772}
    
    def __init__(
        self,
        strict: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the data validator.
        
        Args:
            strict: If True, warnings are treated as errors
            config: Optional configuration dictionary with keys:
                - 'sampling_rate': Expected sampling rate
                - 'n_channels': Expected number of channels
                - 'amplitude_range': Tuple of (min, max) amplitude
                - 'check_events': Whether to validate events
        """
        self._strict = strict
        self._config = config or {}
        
        # Extract config values with defaults
        self._expected_sr = self._config.get(
            'sampling_rate', self.DEFAULT_SAMPLING_RATE
        )
        self._expected_channels = self._config.get(
            'n_channels', None  # None means accept any reasonable number
        )
        self._amplitude_range = self._config.get(
            'amplitude_range', self.DEFAULT_AMPLITUDE_RANGE
        )
        self._check_events = self._config.get('check_events', True)
        
        logger.debug(
            f"DataValidator initialized (strict={strict})"
        )
    
    # =========================================================================
    # MAIN VALIDATION METHODS
    # =========================================================================
    
    def validate(
        self,
        data: Union[EEGData, TrialData, np.ndarray]
    ) -> ValidationResult:
        """
        Validate EEG data.
        
        Performs comprehensive validation including structure, values,
        format, and compatibility checks.
        
        Args:
            data: Data to validate (EEGData, TrialData, or numpy array)
        
        Returns:
            ValidationResult with is_valid, errors, and warnings
        
        Example:
            >>> result = validator.validate(eeg_data)
            >>> if result.is_valid:
            ...     print("Data is valid!")
            >>> else:
            ...     for error in result.errors:
            ...         print(f"Error: {error}")
        """
        result = ValidationResult()
        
        if isinstance(data, EEGData):
            self._validate_eegdata(data, result)
        elif isinstance(data, TrialData):
            self._validate_trialdata(data, result)
        elif isinstance(data, np.ndarray):
            self._validate_array(data, result)
        else:
            result.add_error(
                f"Unsupported data type: {type(data).__name__}. "
                f"Expected EEGData, TrialData, or numpy array."
            )
        
        # Convert warnings to errors in strict mode
        if self._strict and result.warnings:
            for warning in result.warnings:
                result.add_error(f"[strict] {warning}")
            result.warnings.clear()
        
        return result
    
    def assert_valid(
        self,
        data: Union[EEGData, TrialData, np.ndarray],
        message: str = "Data validation failed"
    ) -> None:
        """
        Assert that data is valid, raising exception if not.
        
        Args:
            data: Data to validate
            message: Error message prefix
        
        Raises:
            ValidationError: If data is invalid
        
        Example:
            >>> validator.assert_valid(eeg_data)  # Raises if invalid
        """
        result = self.validate(data)
        
        if not result.is_valid:
            raise ValidationError(
                message,
                errors=result.errors,
                warnings=result.warnings
            )
    
    def is_valid(self, data: Union[EEGData, TrialData, np.ndarray]) -> bool:
        """
        Quick check if data is valid.
        
        Args:
            data: Data to check
        
        Returns:
            bool: True if valid
        """
        return self.validate(data).is_valid
    
    # =========================================================================
    # SPECIFIC VALIDATORS
    # =========================================================================
    
    def _validate_eegdata(self, data: EEGData, result: ValidationResult) -> None:
        """
        Validate EEGData object.
        
        Args:
            data: EEGData to validate
            result: ValidationResult to populate
        """
        logger.debug(f"Validating EEGData: {data}")
        
        # Validate signals array
        self._validate_signals_array(data.signals, result)
        
        # Validate sampling rate
        self._validate_sampling_rate(data.sampling_rate, result)
        
        # Validate channel names
        self._validate_channel_names(
            data.channel_names, 
            data.n_channels, 
            result
        )
        
        # Validate events
        if self._check_events and data.events:
            self._validate_events(data.events, data.n_samples, result)
        
        # Store info
        result.info['n_channels'] = data.n_channels
        result.info['n_samples'] = data.n_samples
        result.info['duration_seconds'] = data.duration_seconds
        result.info['n_events'] = data.n_events
    
    def _validate_trialdata(self, data: TrialData, result: ValidationResult) -> None:
        """
        Validate TrialData object.
        
        Args:
            data: TrialData to validate
            result: ValidationResult to populate
        """
        logger.debug(f"Validating TrialData: {data}")
        
        # Validate signals array
        self._validate_signals_array(data.signals, result)
        
        # Validate label
        if data.label < 0 or data.label > 3:
            result.add_warning(
                f"Label {data.label} outside expected range [0-3] for 4-class MI"
            )
        
        # Validate sampling rate
        self._validate_sampling_rate(data.sampling_rate, result)
        
        # Validate channel names
        self._validate_channel_names(
            data.channel_names,
            data.n_channels,
            result
        )
        
        # Store info
        result.info['n_channels'] = data.n_channels
        result.info['n_samples'] = data.n_samples
        result.info['label'] = data.label
        result.info['duration_seconds'] = data.duration_seconds
    
    def _validate_array(self, data: np.ndarray, result: ValidationResult) -> None:
        """
        Validate numpy array.
        
        Args:
            data: Array to validate
            result: ValidationResult to populate
        """
        logger.debug(f"Validating array with shape {data.shape}")
        
        # Check dimensions
        if data.ndim not in [2, 3]:
            result.add_error(
                f"Expected 2D (channels, samples) or 3D (trials, channels, samples) "
                f"array, got {data.ndim}D with shape {data.shape}"
            )
            return
        
        # Validate values
        self._validate_signal_values(data, result)
        
        # Store info
        result.info['shape'] = data.shape
        result.info['dtype'] = str(data.dtype)
    
    # =========================================================================
    # COMPONENT VALIDATORS
    # =========================================================================
    
    def _validate_signals_array(
        self,
        signals: np.ndarray,
        result: ValidationResult
    ) -> None:
        """
        Validate signals array structure and values.
        
        Args:
            signals: Signal array to validate
            result: ValidationResult to populate
        """
        # Check array type
        if not isinstance(signals, np.ndarray):
            result.add_error(
                f"Signals must be numpy array, got {type(signals).__name__}"
            )
            return
        
        # Check dimensions
        if signals.ndim != 2:
            result.add_error(
                f"Signals must be 2D (channels, samples), got {signals.ndim}D"
            )
            return
        
        n_channels, n_samples = signals.shape
        
        # Check channel count
        if self._expected_channels is not None:
            if n_channels != self._expected_channels:
                result.add_error(
                    f"Expected {self._expected_channels} channels, got {n_channels}"
                )
        else:
            # Check reasonable channel count
            if n_channels < 1:
                result.add_error(f"Invalid channel count: {n_channels}")
            elif n_channels > 256:
                result.add_warning(
                    f"Unusually high channel count: {n_channels}"
                )
        
        # Check sample count
        if n_samples < 1:
            result.add_error(f"Invalid sample count: {n_samples}")
        
        # Validate values
        self._validate_signal_values(signals, result)
    
    def _validate_signal_values(
        self,
        signals: np.ndarray,
        result: ValidationResult
    ) -> None:
        """
        Validate signal values (NaN, Inf, amplitude).
        
        Args:
            signals: Signal array to validate
            result: ValidationResult to populate
        """
        # Check for NaN values
        nan_count = np.sum(np.isnan(signals))
        if nan_count > 0:
            nan_ratio = nan_count / signals.size
            if nan_ratio > 0.1:  # More than 10% NaN
                result.add_error(
                    f"High proportion of NaN values: {nan_ratio:.1%}"
                )
            else:
                result.add_warning(
                    f"Contains NaN values: {nan_count} ({nan_ratio:.2%})"
                )
        
        # Check for Inf values
        inf_count = np.sum(np.isinf(signals))
        if inf_count > 0:
            result.add_error(
                f"Contains {inf_count} Inf values"
            )
        
        # Check amplitude range (skip if has NaN/Inf)
        if nan_count == 0 and inf_count == 0:
            min_val = np.min(signals)
            max_val = np.max(signals)
            
            min_expected, max_expected = self._amplitude_range
            
            if min_val < min_expected or max_val > max_expected:
                result.add_warning(
                    f"Signal amplitude [{min_val:.1f}, {max_val:.1f}] "
                    f"outside expected range [{min_expected}, {max_expected}] µV"
                )
            
            # Store statistics
            result.info['signal_min'] = float(min_val)
            result.info['signal_max'] = float(max_val)
            result.info['signal_mean'] = float(np.mean(signals))
            result.info['signal_std'] = float(np.std(signals))
    
    def _validate_sampling_rate(
        self,
        sampling_rate: float,
        result: ValidationResult
    ) -> None:
        """
        Validate sampling rate.
        
        Args:
            sampling_rate: Sampling rate to validate
            result: ValidationResult to populate
        """
        if sampling_rate <= 0:
            result.add_error(
                f"Invalid sampling rate: {sampling_rate}"
            )
            return
        
        # Check against expected
        if self._expected_sr and sampling_rate != self._expected_sr:
            result.add_warning(
                f"Sampling rate {sampling_rate} Hz differs from "
                f"expected {self._expected_sr} Hz"
            )
        
        # Check reasonable range
        if sampling_rate < 100:
            result.add_warning(
                f"Low sampling rate: {sampling_rate} Hz"
            )
        elif sampling_rate > 10000:
            result.add_warning(
                f"Unusually high sampling rate: {sampling_rate} Hz"
            )
    
    def _validate_channel_names(
        self,
        channel_names: List[str],
        n_channels: int,
        result: ValidationResult
    ) -> None:
        """
        Validate channel names.
        
        Args:
            channel_names: List of channel names
            n_channels: Number of channels in data
            result: ValidationResult to populate
        """
        if not channel_names:
            result.add_warning("No channel names provided")
            return
        
        # Check count matches
        if len(channel_names) != n_channels:
            result.add_error(
                f"Channel name count ({len(channel_names)}) doesn't match "
                f"channel count ({n_channels})"
            )
        
        # Check for duplicates
        if len(channel_names) != len(set(channel_names)):
            duplicates = [
                name for name in channel_names 
                if channel_names.count(name) > 1
            ]
            result.add_warning(
                f"Duplicate channel names: {set(duplicates)}"
            )
        
        # Check for empty names
        empty_names = [i for i, name in enumerate(channel_names) if not name.strip()]
        if empty_names:
            result.add_warning(
                f"Empty channel names at indices: {empty_names}"
            )
    
    def _validate_events(
        self,
        events: List,
        n_samples: int,
        result: ValidationResult
    ) -> None:
        """
        Validate event markers.
        
        Args:
            events: List of EventMarker objects
            n_samples: Total number of samples
            result: ValidationResult to populate
        """
        if not events:
            return
        
        mi_event_count = 0
        invalid_positions = []
        unknown_codes = set()
        
        for event in events:
            # Check position bounds
            if event.sample < 0:
                invalid_positions.append((event.sample, "negative"))
            elif event.sample >= n_samples:
                invalid_positions.append((event.sample, "exceeds data length"))
            
            # Check event code
            if event.code not in self.VALID_EVENT_CODES:
                unknown_codes.add(event.code)
            
            # Count MI events
            if event.code in self.MI_CLASS_CODES:
                mi_event_count += 1
        
        # Report issues
        if invalid_positions:
            result.add_error(
                f"Invalid event positions: {invalid_positions[:5]}..."
                if len(invalid_positions) > 5 else
                f"Invalid event positions: {invalid_positions}"
            )
        
        if unknown_codes:
            result.add_warning(
                f"Unknown event codes: {unknown_codes}"
            )
        
        # Store event info
        result.info['n_events'] = len(events)
        result.info['n_mi_events'] = mi_event_count
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def validate_for_bci_iv_2a(
        self,
        data: EEGData
    ) -> ValidationResult:
        """
        Validate data specifically for BCI Competition IV-2a compatibility.
        
        Applies stricter validation for the standard dataset format.
        
        Args:
            data: EEGData to validate
        
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        # Must be EEGData
        if not isinstance(data, EEGData):
            result.add_error("Expected EEGData object")
            return result
        
        # Check sampling rate
        if data.sampling_rate != 250:
            result.add_error(
                f"BCI IV-2a requires 250 Hz sampling rate, got {data.sampling_rate}"
            )
        
        # Check channel count
        if data.n_channels not in [22, 25]:
            result.add_error(
                f"BCI IV-2a requires 22 (EEG) or 25 (EEG+EOG) channels, "
                f"got {data.n_channels}"
            )
        
        # Check for MI events
        mi_events = [e for e in data.events if e.code in self.MI_CLASS_CODES]
        if len(mi_events) == 0:
            result.add_warning("No motor imagery events found")
        elif len(mi_events) != 288:
            result.add_warning(
                f"Expected 288 MI trials, found {len(mi_events)} MI events"
            )
        
        # Check class distribution
        class_counts = {}
        for e in mi_events:
            class_counts[e.code] = class_counts.get(e.code, 0) + 1
        
        result.info['class_distribution'] = class_counts
        
        # Check for balanced classes
        if class_counts:
            expected_per_class = len(mi_events) / 4
            for code, count in class_counts.items():
                if abs(count - expected_per_class) > expected_per_class * 0.2:
                    result.add_warning(
                        f"Imbalanced class {code}: {count} trials "
                        f"(expected ~{expected_per_class:.0f})"
                    )
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DataValidator(strict={self._strict})"
