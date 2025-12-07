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

Author: EEG-BCI Framework
Date: 2024
"""

# =============================================================================
# Data Loading Interface (using relative imports)
# =============================================================================
from .i_data_loader import IDataLoader

# =============================================================================
# Signal Processing Interfaces
# =============================================================================
from .i_preprocessor import IPreprocessor
from .i_feature_extractor import IFeatureExtractor

# =============================================================================
# Classification Interface
# =============================================================================
from .i_classifier import IClassifier

# =============================================================================
# Agent System Interfaces
# =============================================================================
from .i_agent import IAgent
from .i_policy import IPolicy
from .i_reward import IReward, ICompositeReward, RewardType

# =============================================================================
# LLM Provider Interface
# =============================================================================
from .i_llm_provider import ILLMProvider, LLMProviderType

# =============================================================================
# Storage Interface
# =============================================================================
from .i_storage_adapter import IStorageAdapter, StorageBackendType

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
