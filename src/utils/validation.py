"""
Validation Utilities
====================

This module provides validation functions and decorators for the EEG-BCI framework.

Features:
---------
- Input validation for functions and classes
- Data shape and type checking
- Configuration validation
- Parameter range validation

Validation Categories:
---------------------
1. Type Validation: Check argument types
2. Shape Validation: Check numpy array shapes
3. Range Validation: Check numeric ranges
4. Config Validation: Validate configuration dictionaries

Example Usage:
    ```python
    from src.utils.validation import (
        validate_array, validate_config, 
        check_type, check_range
    )
    
    # Validate array shape
    validate_array(data, expected_ndim=3, name='input_data')
    
    # Validate configuration
    validate_config(config, required_keys=['learning_rate', 'epochs'])
    
    # Using decorators
    @validate_params(data=ArrayValidator(ndim=3))
    def process_data(data):
        ...
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import (
    Dict, List, Optional, Any, Union, Tuple, 
    Callable, Type, TypeVar, Sequence
)
from functools import wraps
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')


# =============================================================================
# BASIC TYPE CHECKING
# =============================================================================

def check_type(value: Any, 
               expected_type: Union[Type, Tuple[Type, ...]],
               name: str = 'value') -> None:
    """
    Check if value is of expected type.
    
    Args:
        value: Value to check
        expected_type: Expected type(s)
        name: Name of the value for error message
    
    Raises:
        TypeError: If type doesn't match
    
    Example:
        >>> check_type(data, np.ndarray, 'data')
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected_str = ' or '.join(t.__name__ for t in expected_type)
        else:
            expected_str = expected_type.__name__
        
        raise TypeError(
            f"'{name}' must be {expected_str}, got {type(value).__name__}"
        )


def check_not_none(value: Any, name: str = 'value') -> None:
    """
    Check if value is not None.
    
    Args:
        value: Value to check
        name: Name of the value
    
    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(f"'{name}' cannot be None")


# =============================================================================
# NUMERIC VALIDATION
# =============================================================================

def check_range(value: Union[int, float],
                min_val: Optional[Union[int, float]] = None,
                max_val: Optional[Union[int, float]] = None,
                name: str = 'value',
                inclusive: bool = True) -> None:
    """
    Check if value is within range.
    
    Args:
        value: Value to check
        min_val: Minimum allowed value (or None for no minimum)
        max_val: Maximum allowed value (or None for no maximum)
        name: Name of the value
        inclusive: Whether range is inclusive
    
    Raises:
        ValueError: If value is out of range
    
    Example:
        >>> check_range(learning_rate, min_val=0.0, max_val=1.0, name='learning_rate')
    """
    if min_val is not None:
        if inclusive and value < min_val:
            raise ValueError(f"'{name}' must be >= {min_val}, got {value}")
        elif not inclusive and value <= min_val:
            raise ValueError(f"'{name}' must be > {min_val}, got {value}")
    
    if max_val is not None:
        if inclusive and value > max_val:
            raise ValueError(f"'{name}' must be <= {max_val}, got {value}")
        elif not inclusive and value >= max_val:
            raise ValueError(f"'{name}' must be < {max_val}, got {value}")


def check_positive(value: Union[int, float], 
                   name: str = 'value',
                   allow_zero: bool = False) -> None:
    """
    Check if value is positive.
    
    Args:
        value: Value to check
        name: Name of the value
        allow_zero: Whether zero is allowed
    
    Raises:
        ValueError: If value is not positive
    """
    if allow_zero:
        if value < 0:
            raise ValueError(f"'{name}' must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValueError(f"'{name}' must be positive, got {value}")


def check_probability(value: float, name: str = 'value') -> None:
    """
    Check if value is a valid probability (0-1).
    
    Args:
        value: Value to check
        name: Name of the value
    """
    check_range(value, min_val=0.0, max_val=1.0, name=name)


# =============================================================================
# ARRAY VALIDATION
# =============================================================================

def validate_array(array: np.ndarray,
                   expected_ndim: Optional[int] = None,
                   expected_shape: Optional[Tuple[int, ...]] = None,
                   min_samples: Optional[int] = None,
                   allow_nan: bool = False,
                   allow_inf: bool = False,
                   dtype: Optional[Union[np.dtype, Type]] = None,
                   name: str = 'array') -> None:
    """
    Validate numpy array properties.
    
    Args:
        array: Array to validate
        expected_ndim: Expected number of dimensions
        expected_shape: Expected shape (use -1 for any size in that dimension)
        min_samples: Minimum number of samples (first dimension)
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether Inf values are allowed
        dtype: Expected data type
        name: Name of the array
    
    Raises:
        TypeError: If not a numpy array
        ValueError: If array doesn't meet criteria
    
    Example:
        >>> validate_array(X, expected_ndim=3, min_samples=10, name='X')
    """
    # Type check
    if not isinstance(array, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy array, got {type(array).__name__}")
    
    # Dimension check
    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(
            f"'{name}' must be {expected_ndim}D, got {array.ndim}D (shape={array.shape})"
        )
    
    # Shape check
    if expected_shape is not None:
        for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValueError(
                    f"'{name}' shape mismatch at dimension {i}: "
                    f"expected {expected}, got {actual} (shape={array.shape})"
                )
    
    # Minimum samples check
    if min_samples is not None and array.shape[0] < min_samples:
        raise ValueError(
            f"'{name}' must have at least {min_samples} samples, "
            f"got {array.shape[0]}"
        )
    
    # NaN check
    if not allow_nan and np.any(np.isnan(array)):
        raise ValueError(f"'{name}' contains NaN values")
    
    # Inf check
    if not allow_inf and np.any(np.isinf(array)):
        raise ValueError(f"'{name}' contains Inf values")
    
    # Dtype check
    if dtype is not None and array.dtype != dtype:
        raise ValueError(
            f"'{name}' must have dtype {dtype}, got {array.dtype}"
        )


def validate_eeg_data(data: np.ndarray,
                      n_channels: Optional[int] = None,
                      n_samples: Optional[int] = None,
                      sampling_rate: Optional[float] = None,
                      name: str = 'eeg_data') -> None:
    """
    Validate EEG data format.
    
    Expected format: (n_channels, n_samples) or (n_trials, n_channels, n_samples)
    
    Args:
        data: EEG data array
        n_channels: Expected number of channels
        n_samples: Expected number of samples
        sampling_rate: Sampling rate (for duration validation)
        name: Name of the data
    
    Raises:
        ValueError: If data format is invalid
    """
    if data.ndim == 2:
        # Continuous recording: (channels, samples)
        ch_dim, samp_dim = 0, 1
    elif data.ndim == 3:
        # Epoched: (trials, channels, samples)
        ch_dim, samp_dim = 1, 2
    else:
        raise ValueError(
            f"'{name}' must be 2D (channels, samples) or 3D (trials, channels, samples), "
            f"got {data.ndim}D"
        )
    
    if n_channels is not None and data.shape[ch_dim] != n_channels:
        raise ValueError(
            f"'{name}' expected {n_channels} channels, got {data.shape[ch_dim]}"
        )
    
    if n_samples is not None and data.shape[samp_dim] != n_samples:
        raise ValueError(
            f"'{name}' expected {n_samples} samples, got {data.shape[samp_dim]}"
        )


def validate_labels(labels: np.ndarray,
                    n_samples: Optional[int] = None,
                    n_classes: Optional[int] = None,
                    name: str = 'labels') -> None:
    """
    Validate classification labels.
    
    Args:
        labels: Label array
        n_samples: Expected number of samples
        n_classes: Expected number of classes
        name: Name of the labels
    """
    if labels.ndim != 1:
        raise ValueError(f"'{name}' must be 1D, got {labels.ndim}D")
    
    if n_samples is not None and len(labels) != n_samples:
        raise ValueError(
            f"'{name}' length ({len(labels)}) doesn't match data samples ({n_samples})"
        )
    
    unique_labels = np.unique(labels)
    
    if n_classes is not None and len(unique_labels) != n_classes:
        raise ValueError(
            f"'{name}' has {len(unique_labels)} unique classes, expected {n_classes}"
        )
    
    # Check if labels are consecutive integers starting from 0
    expected = np.arange(len(unique_labels))
    if not np.array_equal(unique_labels, expected):
        logger.warning(
            f"'{name}' labels are not consecutive integers starting from 0: {unique_labels}"
        )


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config(config: Dict[str, Any],
                    required_keys: Optional[List[str]] = None,
                    optional_keys: Optional[List[str]] = None,
                    strict: bool = False,
                    name: str = 'config') -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: Keys that must be present
        optional_keys: Keys that are allowed but not required
        strict: If True, raise error for unexpected keys
        name: Name of the config
    
    Raises:
        TypeError: If config is not a dict
        ValueError: If required keys missing or unknown keys present (strict mode)
    
    Example:
        >>> validate_config(
        ...     config,
        ...     required_keys=['learning_rate', 'epochs'],
        ...     optional_keys=['batch_size', 'patience']
        ... )
    """
    if not isinstance(config, dict):
        raise TypeError(f"'{name}' must be a dictionary, got {type(config).__name__}")
    
    # Check required keys
    if required_keys:
        missing = set(required_keys) - set(config.keys())
        if missing:
            raise ValueError(f"'{name}' missing required keys: {missing}")
    
    # Check for unknown keys in strict mode
    if strict:
        allowed = set(required_keys or []) | set(optional_keys or [])
        unknown = set(config.keys()) - allowed
        if unknown:
            raise ValueError(f"'{name}' has unknown keys: {unknown}")


def validate_config_value(config: Dict[str, Any],
                          key: str,
                          expected_type: Optional[Type] = None,
                          min_val: Optional[Union[int, float]] = None,
                          max_val: Optional[Union[int, float]] = None,
                          choices: Optional[List[Any]] = None) -> None:
    """
    Validate a specific configuration value.
    
    Args:
        config: Configuration dictionary
        key: Key to validate
        expected_type: Expected type of the value
        min_val: Minimum value (for numbers)
        max_val: Maximum value (for numbers)
        choices: List of valid choices
    """
    if key not in config:
        return  # Key not present, skip validation
    
    value = config[key]
    
    if expected_type is not None:
        check_type(value, expected_type, name=key)
    
    if min_val is not None or max_val is not None:
        check_range(value, min_val=min_val, max_val=max_val, name=key)
    
    if choices is not None and value not in choices:
        raise ValueError(f"'{key}' must be one of {choices}, got {value}")


# =============================================================================
# DECORATOR-BASED VALIDATION
# =============================================================================

class Validator:
    """Base class for parameter validators."""
    
    def validate(self, value: Any, name: str) -> None:
        """Validate a value. Override in subclasses."""
        raise NotImplementedError


class ArrayValidator(Validator):
    """Validator for numpy arrays."""
    
    def __init__(self,
                 ndim: Optional[int] = None,
                 min_samples: Optional[int] = None,
                 allow_nan: bool = False):
        self.ndim = ndim
        self.min_samples = min_samples
        self.allow_nan = allow_nan
    
    def validate(self, value: Any, name: str) -> None:
        validate_array(
            value, 
            expected_ndim=self.ndim,
            min_samples=self.min_samples,
            allow_nan=self.allow_nan,
            name=name
        )


class RangeValidator(Validator):
    """Validator for numeric ranges."""
    
    def __init__(self, min_val: float = None, max_val: float = None):
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value: Any, name: str) -> None:
        check_range(value, min_val=self.min_val, max_val=self.max_val, name=name)


class TypeValidator(Validator):
    """Validator for types."""
    
    def __init__(self, expected_type: Union[Type, Tuple[Type, ...]]):
        self.expected_type = expected_type
    
    def validate(self, value: Any, name: str) -> None:
        check_type(value, self.expected_type, name=name)


def validate_params(**validators: Validator) -> Callable:
    """
    Decorator to validate function parameters.
    
    Args:
        **validators: Mapping of parameter names to Validator instances
    
    Example:
        >>> @validate_params(
        ...     data=ArrayValidator(ndim=3),
        ...     learning_rate=RangeValidator(min_val=0, max_val=1)
        ... )
        ... def train(data, learning_rate):
        ...     ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    validator.validate(bound.arguments[param_name], param_name)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def ensure_2d(array: np.ndarray, name: str = 'array') -> np.ndarray:
    """
    Ensure array is 2D, reshaping if needed.
    
    Args:
        array: Input array (1D or 2D)
        name: Array name for error messages
    
    Returns:
        2D array
    """
    if array.ndim == 1:
        return array.reshape(1, -1)
    elif array.ndim == 2:
        return array
    else:
        raise ValueError(f"'{name}' must be 1D or 2D, got {array.ndim}D")


def ensure_3d(array: np.ndarray, name: str = 'array') -> np.ndarray:
    """
    Ensure array is 3D (for batched processing).
    
    Args:
        array: Input array (2D or 3D)
        name: Array name
    
    Returns:
        3D array with shape (n_trials, n_channels, n_samples)
    """
    if array.ndim == 2:
        return array[np.newaxis, :, :]
    elif array.ndim == 3:
        return array
    else:
        raise ValueError(f"'{name}' must be 2D or 3D, got {array.ndim}D")


def validate_same_length(*arrays: np.ndarray, 
                          names: Optional[List[str]] = None) -> None:
    """
    Validate that all arrays have the same length (first dimension).
    
    Args:
        *arrays: Arrays to compare
        names: Optional names for error messages
    """
    if len(arrays) < 2:
        return
    
    lengths = [len(a) for a in arrays]
    
    if len(set(lengths)) > 1:
        names = names or [f'array_{i}' for i in range(len(arrays))]
        length_info = ', '.join(f'{n}={l}' for n, l in zip(names, lengths))
        raise ValueError(f"Arrays must have same length: {length_info}")
