"""
Core Module
===========

This is the core module of the EEG-BCI Framework, containing:
- Abstract interfaces for all components
- Data types for EEG signals and trials
- Configuration management
- Component registry for plugin architecture
- Custom exceptions

Quick Start:
-----------
```python
from src.core import (
    # Interfaces
    IDataLoader, IPreprocessor, IClassifier, IAgent,
    
    # Data Types
    EEGData, TrialData, EventMarker,
    
    # Configuration
    ConfigManager, get_config,
    
    # Registry
    ComponentRegistry, get_registry,
    
    # Exceptions
    DataLoadError, ModelNotFittedError
)

# Get configuration
config = get_config()
print(config.get('agents.dva.confidence_threshold'))  # 0.8

# Get component registry
registry = get_registry()
loader = registry.create('data_loader', 'mat', config={'channels': ['C3', 'C4']})

# Load and process data
eeg_data = loader.load('data.mat')
trials = eeg_data.extract_trials()
```

Author: EEG-BCI Framework
Date: 2024
"""

# =============================================================================
# Interfaces
# =============================================================================
from src.core.interfaces import (
    # Data Loading
    IDataLoader,
    
    # Signal Processing
    IPreprocessor,
    IFeatureExtractor,
    
    # Classification
    IClassifier,
    
    # Agent System
    IAgent,
    IPolicy,
    IReward,
    ICompositeReward,
    RewardType,
    
    # LLM
    ILLMProvider,
    LLMProviderType,
    
    # Storage
    IStorageAdapter,
    StorageBackendType,
)

# =============================================================================
# Data Types
# =============================================================================
from src.core.types import (
    EEGData,
    TrialData,
    EventMarker,
    DatasetInfo
)

# =============================================================================
# Configuration & Registry
# =============================================================================
from src.core.config import (
    ConfigManager,
    get_config,
    load_config
)

from src.core.registry import (
    ComponentRegistry,
    get_registry,
    register,
    create,
    registered
)

# =============================================================================
# Exceptions
# =============================================================================
from src.core.exceptions import (
    # Base
    BCIFrameworkError,
    
    # Data
    DataError,
    DataLoadError,
    DataValidationError,
    DataFormatError,
    MissingDataError,
    ChannelNotFoundError,
    
    # Processing
    ProcessingError,
    PreprocessingError,
    FeatureExtractionError,
    FilterError,
    
    # Classification
    ClassificationError,
    ModelNotFittedError,
    ModelNotFoundError,
    PredictionError,
    
    # Agent
    AgentError,
    AgentNotInitializedError,
    PolicyError,
    RewardError,
    InvalidStateError,
    
    # LLM
    LLMError,
    LLMNotLoadedError,
    GenerationError,
    PromptError,
    
    # Configuration
    ConfigurationError,
    ConfigNotFoundError,
    ConfigValidationError,
    MissingConfigError,
    
    # Storage
    StorageError,
    StorageReadError,
    StorageWriteError,
    CheckpointError,
    
    # Component
    ComponentError,
    ComponentNotFoundError,
    RegistrationError,
    InitializationError,
)

# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    # Interfaces
    'IDataLoader',
    'IPreprocessor',
    'IFeatureExtractor',
    'IClassifier',
    'IAgent',
    'IPolicy',
    'IReward',
    'ICompositeReward',
    'RewardType',
    'ILLMProvider',
    'LLMProviderType',
    'IStorageAdapter',
    'StorageBackendType',
    
    # Data Types
    'EEGData',
    'TrialData',
    'EventMarker',
    'DatasetInfo',
    
    # Configuration
    'ConfigManager',
    'get_config',
    'load_config',
    
    # Registry
    'ComponentRegistry',
    'get_registry',
    'register',
    'create',
    'registered',
    
    # All Exceptions
    'BCIFrameworkError',
    'DataError',
    'DataLoadError',
    'DataValidationError',
    'DataFormatError',
    'MissingDataError',
    'ChannelNotFoundError',
    'ProcessingError',
    'PreprocessingError',
    'FeatureExtractionError',
    'FilterError',
    'ClassificationError',
    'ModelNotFittedError',
    'ModelNotFoundError',
    'PredictionError',
    'AgentError',
    'AgentNotInitializedError',
    'PolicyError',
    'RewardError',
    'InvalidStateError',
    'LLMError',
    'LLMNotLoadedError',
    'GenerationError',
    'PromptError',
    'ConfigurationError',
    'ConfigNotFoundError',
    'ConfigValidationError',
    'MissingConfigError',
    'StorageError',
    'StorageReadError',
    'StorageWriteError',
    'CheckpointError',
    'ComponentError',
    'ComponentNotFoundError',
    'RegistrationError',
    'InitializationError',
]

# Version
__version__ = '1.0.0'
