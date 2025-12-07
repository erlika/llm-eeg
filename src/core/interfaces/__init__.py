"""
Core Interfaces Module
======================

This module exports all abstract interfaces for the EEG-BCI framework.

All components (data loaders, preprocessors, feature extractors, classifiers,
agents, LLM providers) must implement their respective interfaces to ensure
consistent behavior and enable the plugin architecture.

Available Interfaces:
--------------------
- IDataLoader: Interface for EEG data loading from various formats
- IPreprocessor: Interface for signal preprocessing steps
- IFeatureExtractor: Interface for feature extraction methods
- IClassifier: Interface for classification models
- IAgent: Interface for AI agents (APA, DVA)
- IPolicy: Interface for RL policies (used by APA)
- IReward: Interface for reward functions (used by APA)
- ILLMProvider: Interface for LLM providers (Phi-3, etc.)
- IStorageAdapter: Interface for data persistence

Design Pattern:
--------------
All interfaces follow the Abstract Factory pattern, enabling:
1. Dynamic component instantiation via the ComponentRegistry
2. Easy addition of new implementations without modifying existing code
3. Consistent configuration through the ConfigurationManager

Example Usage:
    ```python
    from src.core.interfaces import IDataLoader, IPreprocessor, IClassifier
    
    class MyCustomLoader(IDataLoader):
        # Implement all abstract methods
        ...
    
    # Register with ComponentRegistry
    registry.register('data_loader', 'custom', MyCustomLoader)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

# =============================================================================
# Data Loading Interface
# =============================================================================
from src.core.interfaces.i_data_loader import IDataLoader

# =============================================================================
# Signal Processing Interfaces
# =============================================================================
from src.core.interfaces.i_preprocessor import IPreprocessor
from src.core.interfaces.i_feature_extractor import IFeatureExtractor

# =============================================================================
# Classification Interface
# =============================================================================
from src.core.interfaces.i_classifier import IClassifier

# =============================================================================
# Agent System Interfaces
# =============================================================================
from src.core.interfaces.i_agent import IAgent
from src.core.interfaces.i_policy import IPolicy
from src.core.interfaces.i_reward import IReward, ICompositeReward, RewardType

# =============================================================================
# LLM Provider Interface
# =============================================================================
from src.core.interfaces.i_llm_provider import ILLMProvider, LLMProviderType

# =============================================================================
# Storage Interface
# =============================================================================
from src.core.interfaces.i_storage_adapter import IStorageAdapter, StorageBackendType

# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    # Data Loading
    'IDataLoader',
    
    # Signal Processing
    'IPreprocessor',
    'IFeatureExtractor',
    
    # Classification
    'IClassifier',
    
    # Agent System
    'IAgent',
    'IPolicy',
    'IReward',
    'ICompositeReward',
    'RewardType',
    
    # LLM
    'ILLMProvider',
    'LLMProviderType',
    
    # Storage
    'IStorageAdapter',
    'StorageBackendType',
]
