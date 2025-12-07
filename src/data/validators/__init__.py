"""
Data Validators Module
======================

This module provides validation and quality assessment functionality
for EEG data in the BCI framework.

Components:
----------
- DataValidator: Validates data structure, format, and integrity
- QualityChecker: Assesses signal quality metrics
- ValidationResult: Container for validation results
- ValidationError: Exception for validation failures

Validation Levels:
-----------------
1. **Structure Validation**: Array shapes, types, dimensions
2. **Format Validation**: Channel names, sampling rate, events
3. **Integrity Validation**: NaN/Inf values, amplitude ranges
4. **Quality Assessment**: SNR, line noise, artifacts

BCI Competition IV-2a Support:
-----------------------------
Validators include specific checks for the BCI Competition IV-2a dataset:
- 22 EEG channels (or 25 with EOG)
- 250 Hz sampling rate
- Valid event codes (768, 769, 770, 771, 772)
- Artifact markers (1023)

Usage Examples:
    ```python
    # Validate data structure
    from src.data.validators import DataValidator
    
    validator = DataValidator()
    result = validator.validate(eeg_data)
    
    if result.is_valid:
        print("Data is valid!")
    else:
        for error in result.errors:
            print(f"Error: {error}")
    
    # Check signal quality
    from src.data.validators import QualityChecker
    
    checker = QualityChecker()
    checker.initialize({'sampling_rate': 250})
    
    report = checker.assess_quality(eeg_data)
    print(f"Quality score: {report['overall_score']:.2%}")
    print(f"SNR: {report['snr_db']:.1f} dB")
    
    # Quick validation with assertion
    validator.assert_valid(eeg_data)  # Raises ValidationError if invalid
    ```

Author: EEG-BCI Framework
Date: 2024
"""

# Import validators
from src.data.validators.data_validator import (
    DataValidator,
    ValidationResult,
    ValidationError,
)

from src.data.validators.quality_checker import (
    QualityChecker,
)

# Define public API
__all__ = [
    # Data validation
    'DataValidator',
    'ValidationResult',
    'ValidationError',
    
    # Quality assessment
    'QualityChecker',
]
