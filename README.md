# LLM-EEG Framework

A modular Brain-Computer Interface framework with LLM integration and AI Agents for motor imagery EEG classification.

🔗 **Repository**: https://github.com/erlika/llm-eeg

## Features

- **Modular Plugin Architecture**: Easily swap components (loaders, preprocessors, classifiers)
- **AI Agent System**:
  - **Adaptive Preprocessing Agent (APA)**: RL-based (Q-learning) dynamic preprocessing optimization
  - **Decision Validation Agent (DVA)**: Multi-criteria classification validation (0.8 confidence threshold)
- **LLM Integration**: Phi-3-mini for human-readable explanations
- **Cross-Trial Learning**: Continuous improvement within and across sessions
- **BCI Competition IV-2a Support**: Pre-configured for the standard benchmark dataset

## Dataset

This framework is designed for the **BCI Competition IV-2a** dataset:
- 9 subjects
- 4 motor imagery classes (left hand, right hand, feet, tongue)
- 22 EEG channels
- 250 Hz sampling rate
- 288 trials per subject per session

## Dataset

📁 **Google Drive Dataset**: [BCI Competition IV-2a](https://drive.google.com/drive/folders/14tFFsegwr6oYF4wUuf_mjNOAgfuQ_Bwk)

The dataset will be automatically loaded from Google Drive when running in Google Colab.

## Project Structure

```
llm-eeg/
├── src/
│   ├── core/               # Interfaces, types, config, registry
│   │   ├── interfaces/     # Abstract interfaces
│   │   ├── types/          # Data types (EEGData, TrialData)
│   │   ├── exceptions/     # Custom exceptions
│   │   ├── config.py       # Configuration manager
│   │   └── registry.py     # Component registry
│   ├── data/               # Data loading and management
│   ├── preprocessing/      # Signal preprocessing
│   ├── features/           # Feature extraction (CSP, band power)
│   ├── classifiers/        # Classification models (EEGNet, SVM)
│   ├── agents/             # AI agents (APA, DVA)
│   │   ├── apa/            # Adaptive Preprocessing Agent
│   │   └── dva/            # Decision Validation Agent
│   ├── llm/                # LLM providers (Phi-3)
│   ├── pipeline/           # Processing pipelines
│   ├── evaluation/         # Metrics and evaluation
│   ├── storage/            # Storage backends
│   └── utils/              # Utilities (logging, validation)
├── configs/                # YAML configuration files
├── notebooks/              # Jupyter/Colab notebooks
├── tests/                  # Unit and integration tests
├── data/                   # Data directory
│   ├── raw/                # Raw EEG data
│   ├── processed/          # Processed data
│   └── checkpoints/        # Model checkpoints
└── docs/                   # Documentation
```

## Installation

### Google Colab (Recommended)

```python
# Clone the repository
!git clone https://github.com/erlika/llm-eeg.git
%cd llm-eeg

# Install dependencies
!pip install -r requirements.txt

# Mount Google Drive for dataset access
from google.colab import drive
drive.mount('/content/drive')
```

### Local Installation

```bash
git clone https://github.com/erlika/llm-eeg.git
cd llm-eeg
pip install -r requirements.txt
```

## Quick Start

```python
from src.core import get_config, get_registry, EEGData
from src.utils import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO')
logger = get_logger(__name__)

# Get configuration
config = get_config()
print(f"DVA Confidence Threshold: {config.get('agents.dva.confidence_threshold')}")
# Output: DVA Confidence Threshold: 0.8

print(f"APA Policy: {config.get('agents.apa.policy.type')}")
# Output: APA Policy: q_learning

# Get component registry
registry = get_registry()

# Create a data loader (when implemented)
# loader = registry.create('data_loader', 'mat')
# eeg_data = loader.load('data/A01T.mat')
```

## Configuration

The framework uses a hierarchical configuration system:

```python
from src.core import get_config

config = get_config()

# Access user-approved settings
config.get('agents.dva.confidence_threshold')  # 0.8
config.get('agents.apa.policy.type')           # 'q_learning'
config.get('agents.apa.cross_trial_learning')  # True
config.get('llm.provider')                     # 'phi3'

# Access Google Drive dataset settings
config.get('data.google_drive.folder_url')     # Your dataset URL
config.get('data.google_drive.colab_mount_path')  # '/content/drive/MyDrive'
```

## Key Design Decisions (User-Approved)

1. **APA Policy**: RL-based Q-learning for adaptive preprocessing
2. **DVA Confidence Threshold**: 0.8 for decision validation
3. **Cross-Trial Learning**: Enabled for continuous improvement
4. **LLM Provider**: Phi-3-mini-4k-instruct for Google Colab compatibility
5. **Dataset**: BCI Competition IV-2a with `.mat` format

## Development Phases

- [x] **Phase 1**: Foundation & Setup (Core interfaces, config, registry)
- [ ] **Phase 2**: Data Loading & Processing
- [ ] **Phase 3**: Feature Extraction & Classification
- [ ] **Phase 4**: Agent System (APA, DVA)
- [ ] **Phase 5**: LLM Integration
- [ ] **Phase 6**: Evaluation & Documentation

## Performance Targets

- Subject-dependent accuracy: >85%
- Subject-independent accuracy: >70%
- Kappa coefficient: >0.80
- Information Transfer Rate: >100 bits/min

## License

MIT License

## Authors

EEG-BCI Framework Team
