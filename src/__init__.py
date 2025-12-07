"""
LLM-EEG Framework
=================

A modular Brain-Computer Interface framework with LLM integration and AI Agents
for motor imagery EEG classification.

Repository: https://github.com/erlika/llm-eeg

Features:
---------
- Modular plugin architecture
- Support for BCI Competition IV-2a dataset
- Adaptive Preprocessing Agent (APA) with RL-based policy
- Decision Validation Agent (DVA) with 0.8 confidence threshold
- LLM integration (Phi-3) for explanations
- Cross-trial learning
- Google Colab deployment ready

Quick Start:
-----------
```python
import src

# Setup logging
src.utils.setup_logging(level='INFO')

# Get configuration
config = src.core.get_config()

# Load data
registry = src.core.get_registry()
loader = registry.create('data_loader', 'mat')
eeg_data = loader.load('data/A01T.mat')

# Extract trials
X, y = eeg_data.get_trials_array(trial_length_sec=4.0)
```

Project Structure:
-----------------
src/
├── core/               # Core interfaces, types, config, registry
├── data/               # Data loading and management
├── preprocessing/      # Signal preprocessing
├── features/           # Feature extraction
├── classifiers/        # Classification models
├── agents/             # APA and DVA agents
├── llm/                # LLM providers
├── pipeline/           # Processing pipelines
├── evaluation/         # Metrics and evaluation
├── storage/            # Storage backends
└── utils/              # Utilities (logging, validation)

Author: EEG-BCI Framework
Date: 2024
"""

# Version
__version__ = '1.0.0'

# Core module
from src import core
from src import utils

# Convenience imports
from src.core import (
    # Configuration
    get_config,
    load_config,
    ConfigManager,
    
    # Registry
    get_registry,
    ComponentRegistry,
    
    # Types
    EEGData,
    TrialData,
    EventMarker,
    DatasetInfo,
)

from src.utils import (
    setup_logging,
    get_logger,
)

__all__ = [
    # Modules
    'core',
    'utils',
    
    # Configuration
    'get_config',
    'load_config',
    'ConfigManager',
    
    # Registry
    'get_registry',
    'ComponentRegistry',
    
    # Types
    'EEGData',
    'TrialData',
    'EventMarker',
    'DatasetInfo',
    
    # Logging
    'setup_logging',
    'get_logger',
    
    # Version
    '__version__',
]
