# LLM-EEG Framework

A modular Brain-Computer Interface framework with LLM integration and AI Agents for motor imagery EEG classification.

🔗 **Repository**: https://github.com/erlika/llm-eeg

---

## Features

- **Modular Plugin Architecture**: Easily swap components (loaders, preprocessors, classifiers)
- **AI Agent System**:
  - **Adaptive Preprocessing Agent (APA)**: RL-based (Q-learning) dynamic preprocessing optimization
  - **Decision Validation Agent (DVA)**: Multi-criteria classification validation (0.8 confidence threshold)
- **LLM Integration**: Phi-3-mini for human-readable explanations
- **Cross-Trial Learning**: Continuous improvement within and across sessions
- **BCI Competition IV-2a Support**: Pre-configured for the standard benchmark dataset
- **Google Colab Ready**: Designed for seamless deployment in Google Colab

---

## Dataset

📁 **Google Drive Dataset**: [BCI Competition IV-2a](https://drive.google.com/drive/folders/14tFFsegwr6oYF4wUuf_mjNOAgfuQ_Bwk)

| Property | Value |
|----------|-------|
| Subjects | 9 |
| Classes | 4 (left hand, right hand, feet, tongue) |
| Channels | 22 EEG + 3 EOG |
| Sampling Rate | 250 Hz |
| Trials per Session | 288 |
| Sessions per Subject | 2 (Training + Evaluation) |
| Trial Duration | 4 seconds |

---

## Project Structure

```
llm-eeg/
├── src/
│   ├── core/                   # Core framework components
│   │   ├── interfaces/         # Abstract interfaces (9 interfaces)
│   │   │   ├── i_data_loader.py
│   │   │   ├── i_preprocessor.py
│   │   │   ├── i_feature_extractor.py
│   │   │   ├── i_classifier.py
│   │   │   ├── i_agent.py
│   │   │   ├── i_policy.py
│   │   │   ├── i_reward.py
│   │   │   ├── i_llm_provider.py
│   │   │   └── i_storage_adapter.py
│   │   ├── types/              # Data types
│   │   │   └── eeg_data.py     # EEGData, TrialData, EventMarker, DatasetInfo
│   │   ├── exceptions/         # Custom exceptions (30+ exception types)
│   │   ├── config.py           # Configuration manager
│   │   └── registry.py         # Component registry
│   ├── data/                   # Data loading (Phase 2)
│   ├── preprocessing/          # Signal preprocessing (Phase 2)
│   ├── features/               # Feature extraction (Phase 3)
│   ├── classifiers/            # Classification models (Phase 3)
│   ├── agents/                 # AI agents (Phase 4)
│   │   ├── apa/                # Adaptive Preprocessing Agent
│   │   └── dva/                # Decision Validation Agent
│   ├── llm/                    # LLM providers (Phase 5)
│   ├── pipeline/               # Processing pipelines
│   ├── evaluation/             # Metrics and evaluation (Phase 6)
│   ├── storage/                # Storage backends
│   └── utils/                  # Utilities
│       ├── logging.py          # Logging configuration
│       └── validation.py       # Input validation
├── configs/                    # YAML configuration files
├── notebooks/                  # Jupyter/Colab notebooks
├── tests/                      # Unit and integration tests
├── data/                       # Data directory
│   ├── raw/                    # Raw EEG data
│   ├── processed/              # Processed data
│   ├── features/               # Extracted features
│   └── checkpoints/            # Model checkpoints
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Installation

### Google Colab (Recommended)

```python
# Cell 1: Clone and Setup
!git clone https://github.com/erlika/llm-eeg.git

import sys
import os
REPO_PATH = '/content/llm-eeg'
os.chdir(REPO_PATH)
sys.path.insert(0, REPO_PATH)

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Import and Initialize
from src.core import get_config, EEGData, EventMarker, DatasetInfo
from src.utils import setup_logging

setup_logging(level='INFO')
config = get_config()

print("✅ LLM-EEG Framework Ready!")
```

### Local Installation

```bash
git clone https://github.com/erlika/llm-eeg.git
cd llm-eeg
pip install -r requirements.txt
```

---

## Quick Start

### Basic Usage

```python
from src.core import get_config, EEGData, EventMarker, DatasetInfo
from src.utils import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO')
logger = get_logger(__name__)

# Get configuration
config = get_config()

# Display configuration
print(f"DVA Confidence Threshold: {config.get('agents.dva.confidence_threshold')}")
# Output: 0.8

print(f"APA Policy: {config.get('agents.apa.policy.type')}")
# Output: q_learning

print(f"Cross-Trial Learning: {config.get('agents.apa.cross_trial_learning')}")
# Output: True
```

### Create EEG Data

```python
import numpy as np
from src.core import EEGData, EventMarker

# Create sample data
signals = np.random.randn(22, 1000) * 50  # 22 channels, 4 seconds at 250 Hz

events = [
    EventMarker(sample=0, code=1, label='left_hand'),
    EventMarker(sample=250, code=2, label='right_hand'),
]

eeg_data = EEGData(
    signals=signals,
    sampling_rate=250,
    channel_names=config.get('data.channel_names'),
    events=events,
    subject_id='S01',
    session_id='T'
)

print(f"EEGData: {eeg_data}")
# Output: EEGData(shape=(22, 1000), sr=250Hz, duration=4.0s, events=2)
```

### Get Dataset Information

```python
from src.core import DatasetInfo

# Get BCI Competition IV-2a dataset info
dataset_info = DatasetInfo.for_bci_competition_iv_2a()

print(f"Dataset: {dataset_info.name}")
print(f"Subjects: {dataset_info.n_subjects}")
print(f"Classes: {dataset_info.class_names}")
print(f"Channels: {len(dataset_info.channel_names)}")
```

---

## Configuration

The framework uses a hierarchical configuration system with user-approved defaults.

### Access Configuration

```python
from src.core import get_config

config = get_config()

# Agent settings (User-Approved)
config.get('agents.dva.confidence_threshold')      # 0.8
config.get('agents.apa.policy.type')               # 'q_learning'
config.get('agents.apa.cross_trial_learning')      # True

# LLM settings
config.get('llm.provider')                         # 'phi3'
config.get('llm.model_path')                       # 'microsoft/phi-3-mini-4k-instruct'

# Data settings
config.get('data.sampling_rate')                   # 250
config.get('data.n_channels')                      # 22
config.get('data.n_classes')                       # 4

# Google Drive settings
config.get('data.google_drive.folder_url')         # Your dataset URL
config.get('data.google_drive.colab_mount_path')   # '/content/drive/MyDrive'
```

### Configuration Structure

| Category | Key Settings |
|----------|--------------|
| `data` | sampling_rate, n_channels, n_classes, channel_names, google_drive |
| `preprocessing` | bandpass (8-30 Hz), notch (50 Hz), artifact_threshold |
| `agents.apa` | policy (q_learning), state_bins, action_space, cross_trial_learning |
| `agents.dva` | confidence_threshold (0.8), validators, adaptive_threshold |
| `llm` | provider (phi3), model_path, quantization (4bit) |
| `classifiers` | default (eegnet), model configs |
| `training` | validation_split, early_stopping, cross_validation |

---

## Key Design Decisions (User-Approved)

| Decision | Value | Rationale |
|----------|-------|-----------|
| APA Policy | Q-learning (RL-based) | Learns optimal preprocessing per trial |
| DVA Threshold | 0.8 | Balance between acceptance and rejection |
| Cross-Trial Learning | Enabled | Continuous improvement across trials |
| LLM Provider | Phi-3-mini-4k | Efficient for Google Colab (4-bit quantization) |
| Dataset Format | .mat files | Compatible with BCI Competition IV-2a |

---

## Core Interfaces

| Interface | Description | Location |
|-----------|-------------|----------|
| `IDataLoader` | Load EEG data from files | `src/core/interfaces/i_data_loader.py` |
| `IPreprocessor` | Signal preprocessing steps | `src/core/interfaces/i_preprocessor.py` |
| `IFeatureExtractor` | Feature extraction methods | `src/core/interfaces/i_feature_extractor.py` |
| `IClassifier` | Classification models | `src/core/interfaces/i_classifier.py` |
| `IAgent` | AI agents (APA, DVA) | `src/core/interfaces/i_agent.py` |
| `IPolicy` | RL policies | `src/core/interfaces/i_policy.py` |
| `IReward` | Reward functions | `src/core/interfaces/i_reward.py` |
| `ILLMProvider` | LLM providers | `src/core/interfaces/i_llm_provider.py` |
| `IStorageAdapter` | Storage backends | `src/core/interfaces/i_storage_adapter.py` |

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Foundation & Setup | ✅ Complete |
| **Phase 2** | Data Loading & Processing | ⏳ Pending |
| **Phase 3** | Feature Extraction & Classification | ⏳ Pending |
| **Phase 4** | Agent System (APA, DVA) | ⏳ Pending |
| **Phase 5** | LLM Integration | ⏳ Pending |
| **Phase 6** | Evaluation & Documentation | ⏳ Pending |

### Phase 1 Deliverables (Complete)
- ✅ 9 Abstract interfaces
- ✅ Core data types (EEGData, TrialData, EventMarker, DatasetInfo)
- ✅ Configuration manager with user-approved defaults
- ✅ Component registry for plugin architecture
- ✅ 30+ Custom exceptions
- ✅ Logging and validation utilities
- ✅ Google Colab compatibility (relative imports)

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Subject-dependent accuracy | >85% |
| Subject-independent accuracy | >70% |
| Kappa coefficient | >0.80 |
| Information Transfer Rate | >100 bits/min |

---

## Troubleshooting

### ModuleNotFoundError in Google Colab

If you get `ModuleNotFoundError: No module named 'src'`:

```python
# Make sure to run these lines BEFORE importing
import sys
import os
REPO_PATH = '/content/llm-eeg'
os.chdir(REPO_PATH)
sys.path.insert(0, REPO_PATH)

# Now import works
from src.core import get_config, EEGData
```

### Update to Latest Version

```python
# Delete old clone
!rm -rf /content/llm-eeg

# Clone fresh
!git clone https://github.com/erlika/llm-eeg.git

# Restart runtime: Runtime → Restart runtime
```

---

## License

MIT License

---

## Authors

EEG-BCI Framework Team

---

## Changelog

### v1.0.0 (Phase 1) - Foundation & Setup
- Initial release with core framework architecture
- 9 abstract interfaces for all components
- Core data types for EEG handling
- Configuration manager with user-approved defaults
- Component registry for plugin architecture
- Google Colab compatibility
