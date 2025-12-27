"""
Feature Extraction Module
=========================

This module provides feature extraction functionality for the EEG-BCI framework.

The feature extraction module includes:
- Base classes for creating custom extractors
- Built-in extractors (CSP, Band Power, Time Domain)
- Feature extraction pipeline for combining extractors
- Factory for easy extractor creation

Module Structure:
----------------
- base.py: BaseFeatureExtractor class
- factory.py: FeatureExtractorFactory
- pipeline.py: FeatureExtractionPipeline
- extractors/: Individual extractor implementations

Built-in Extractors:
-------------------
1. **CSPExtractor**: Common Spatial Patterns for motor imagery
   - Best for binary/multi-class motor imagery classification
   - Trainable: Yes (learns spatial filters)
   - Output: log-variance features

2. **BandPowerExtractor**: Frequency band power
   - Computes power in specified frequency bands
   - Trainable: No
   - Methods: Welch, FFT, Multitaper

3. **TimeDomainExtractor**: Statistical features
   - Mean, variance, skewness, kurtosis, Hjorth parameters
   - Trainable: No (optional normalization)

Pipeline:
---------
Combine multiple extractors for richer feature sets:

    ```python
    from src.features import FeatureExtractionPipeline, CSPExtractor, BandPowerExtractor
    
    # Create pipeline
    pipeline = FeatureExtractionPipeline()
    pipeline.add_extractor(CSPExtractor(n_components=6))
    pipeline.add_extractor(BandPowerExtractor(bands={'mu': (8, 12)}))
    
    # Fit and extract
    pipeline.fit(X_train, y_train)
    features = pipeline.extract(X_test)
    ```

Quick Start:
-----------
    ```python
    from src.features import create_extractor, create_motor_imagery_pipeline
    
    # Single extractor
    csp = create_extractor('csp', n_components=6)
    csp.fit(X_train, y_train)
    features = csp.extract(X_test)
    
    # Pre-configured pipeline for motor imagery
    pipeline = create_motor_imagery_pipeline(n_csp_components=6)
    pipeline.fit(X_train, y_train)
    features = pipeline.extract(X_test)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

# Base class
from src.features.base import BaseFeatureExtractor

# Extractors
from src.features.extractors.csp import (
    CSPExtractor,
    create_csp_extractor,
)

from src.features.extractors.band_power import (
    BandPowerExtractor,
    create_band_power_extractor,
    create_motor_imagery_band_power,
    DEFAULT_BANDS,
    MOTOR_IMAGERY_BANDS,
    SIMPLE_BANDS,
)

from src.features.extractors.time_domain import (
    TimeDomainExtractor,
    create_time_domain_extractor,
    create_hjorth_extractor,
    ALL_FEATURES as TIME_DOMAIN_FEATURES,
    DEFAULT_FEATURES as TIME_DOMAIN_DEFAULT_FEATURES,
    HJORTH_FEATURES,
)

# Factory
from src.features.factory import (
    FeatureExtractorFactory,
    create_extractor,
    list_extractors,
)

# Pipeline
from src.features.pipeline import (
    FeatureExtractionPipeline,
    create_pipeline,
    create_motor_imagery_pipeline,
)

# Define public API
__all__ = [
    # Base
    'BaseFeatureExtractor',
    
    # Extractors
    'CSPExtractor',
    'BandPowerExtractor',
    'TimeDomainExtractor',
    
    # Factory functions for extractors
    'create_csp_extractor',
    'create_band_power_extractor',
    'create_motor_imagery_band_power',
    'create_time_domain_extractor',
    'create_hjorth_extractor',
    
    # Factory
    'FeatureExtractorFactory',
    'create_extractor',
    'list_extractors',
    
    # Pipeline
    'FeatureExtractionPipeline',
    'create_pipeline',
    'create_motor_imagery_pipeline',
    
    # Constants
    'DEFAULT_BANDS',
    'MOTOR_IMAGERY_BANDS',
    'SIMPLE_BANDS',
    'TIME_DOMAIN_FEATURES',
    'TIME_DOMAIN_DEFAULT_FEATURES',
    'HJORTH_FEATURES',
]
