"""
Utilities Module
================

This module provides common utility functions for the EEG-BCI framework.

Available Modules:
-----------------
- logging: Centralized logging configuration
- validation: Input validation functions and decorators

Author: EEG-BCI Framework
Date: 2024
"""

# =============================================================================
# Logging Utilities (using relative imports)
# =============================================================================
from .logging import (
    setup_logging,
    get_logger,
    set_level,
    log_execution_time,
    log_entry_exit,
    LogLevel,
    ProgressLogger,
    log_exception,
    create_log_dir
)

# =============================================================================
# Validation Utilities
# =============================================================================
from .validation import (
    # Type checking
    check_type,
    check_not_none,
    
    # Numeric validation
    check_range,
    check_positive,
    check_probability,
    
    # Array validation
    validate_array,
    validate_eeg_data,
    validate_labels,
    
    # Config validation
    validate_config,
    validate_config_value,
    
    # Validators
    Validator,
    ArrayValidator,
    RangeValidator,
    TypeValidator,
    validate_params,
    
    # Utilities
    ensure_2d,
    ensure_3d,
    validate_same_length
)

# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    'set_level',
    'log_execution_time',
    'log_entry_exit',
    'LogLevel',
    'ProgressLogger',
    'log_exception',
    'create_log_dir',
    
    # Validation - Type
    'check_type',
    'check_not_none',
    
    # Validation - Numeric
    'check_range',
    'check_positive',
    'check_probability',
    
    # Validation - Array
    'validate_array',
    'validate_eeg_data',
    'validate_labels',
    
    # Validation - Config
    'validate_config',
    'validate_config_value',
    
    # Validators
    'Validator',
    'ArrayValidator',
    'RangeValidator',
    'TypeValidator',
    'validate_params',
    
    # Utilities
    'ensure_2d',
    'ensure_3d',
    'validate_same_length'
]
