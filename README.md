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

### Google Colab Notebooks

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [Phase 2: Data Loading & Processing](notebooks/Phase2_Data_Loading_Processing.ipynb) | Complete tutorial for loading, preprocessing, and PyTorch integration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erlika/llm-eeg/blob/main/notebooks/Phase2_Data_Loading_Processing.ipynb) |
| [Phase 3: Feature Extraction & Classification](notebooks/Phase3_Feature_Extraction_Classification.ipynb) | CSP features, LDA/SVM/EEGNet classification, model comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erlika/llm-eeg/blob/main/notebooks/Phase3_Feature_Extraction_Classification.ipynb) |

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

### Load EEG Data (Phase 2)

```python
from src.data import load_eeg_file, DataLoaderFactory

# Method 1: Quick one-liner
eeg_data = load_eeg_file('/content/drive/MyDrive/BCI_IV_2a/A01T.mat')
print(f"Loaded: {eeg_data}")

# Method 2: Factory with config
loader = DataLoaderFactory.create('mat', config={'include_eog': False})
eeg_data = loader.load('/content/drive/MyDrive/BCI_IV_2a/A01T.mat')

# Extract trials
X, y = eeg_data.get_trials_array(trial_length_sec=4.0)
print(f"Trials: {X.shape}, Labels: {y.shape}")
```

### Preprocess Data (Phase 2)

```python
from src.preprocessing import create_standard_pipeline

# Create standard preprocessing pipeline
pipeline = create_standard_pipeline(
    sampling_rate=250,
    notch_freq=50,       # European power line
    bandpass_low=8,      # Mu rhythm
    bandpass_high=30     # Beta rhythm
)

# Apply preprocessing
processed_data = pipeline.process(eeg_data)
print(f"Preprocessed: {processed_data}")
```

### Create PyTorch Dataset (Phase 2)

```python
from src.datasets import BCICIV2aDataset, train_val_test_split
from torch.utils.data import DataLoader

# Load dataset for subject 1
dataset = BCICIV2aDataset.from_subject(
    subject_id=1,
    session='T',
    data_dir='/content/drive/MyDrive/BCI_IV_2a',
    trial_length_sec=4.0
)

# Split into train/validation
train_ds, val_ds = train_val_test_split(dataset, val_ratio=0.2)

# Create DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

for batch_x, batch_y in train_loader:
    print(f"Batch: {batch_x.shape}, Labels: {batch_y.shape}")
    break
```

### Create EEG Data Manually

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

### Feature Extraction (Phase 3)

```python
from src.features import (
    CSPExtractor, BandPowerExtractor, 
    FeatureExtractionPipeline, create_motor_imagery_pipeline
)

# Method 1: CSP Feature Extraction
csp = CSPExtractor(n_components=6)
csp.initialize({'sampling_rate': 250})
X_train_csp = csp.fit_extract(X_train, y_train)
X_test_csp = csp.extract(X_test)

# Method 2: Motor Imagery Pipeline (CSP + Band Power)
pipeline = create_motor_imagery_pipeline(n_csp_components=6, sampling_rate=250)
X_features = pipeline.fit_extract(X_train, y_train)

print(f"CSP Features: {X_train_csp.shape}")
print(f"Pipeline Features: {X_features.shape}")
```

### Classification (Phase 3)

```python
from src.classifiers import (
    create_lda_classifier, create_svm_classifier,
    create_eegnet_classifier, ClassifierFactory
)

# Traditional ML: CSP + LDA
lda = create_lda_classifier(n_classes=4)
lda.fit(X_train_csp, y_train)
predictions = lda.predict(X_test_csp)
probabilities = lda.predict_proba(X_test_csp)

# Traditional ML: CSP + SVM
svm = create_svm_classifier(kernel='rbf', C=1.0, n_classes=4)
svm.fit(X_train_csp, y_train)
predictions_svm = svm.predict(X_test_csp)

# Deep Learning: EEGNet (end-to-end)
eegnet = create_eegnet_classifier(
    n_classes=4, n_channels=22, n_samples=1000
)
eegnet.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
predictions_dl = eegnet.predict(X_test)

# Using Factory
clf = ClassifierFactory.create('eegnet', n_classes=4, n_channels=22, n_samples=1000)
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
| **Phase 2** | Data Loading & Processing | ✅ Complete |
| **Phase 3** | Feature Extraction & Classification | ✅ Complete |
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

### Phase 2 Deliverables (Complete)
- ✅ **Data Loaders**: MATLoader for BCI Competition IV-2a .mat files
- ✅ **Data Factory**: DataLoaderFactory with auto-detection
- ✅ **Preprocessing Pipeline**: Composable preprocessing steps
- ✅ **Filters**: BandpassFilter (8-30 Hz), NotchFilter (50/60 Hz)
- ✅ **Normalization**: Z-score, min-max, robust, L2 methods
- ✅ **Data Validation**: DataValidator for structure/format checks
- ✅ **Quality Assessment**: QualityChecker for SNR, artifacts, line noise
- ✅ **PyTorch Integration**: EEGDataset, BCICIV2aDataset classes
- ✅ **Data Splitting**: train_val_test_split, cross-validation folds
- ✅ **Unit Tests**: Comprehensive test coverage

### Phase 3 Deliverables (Complete)
- ✅ **CSP Extractor**: Common Spatial Pattern feature extraction (scipy-based)
- ✅ **Band Power Extractor**: Frequency band power (mu, beta) via Welch PSD
- ✅ **Time Domain Extractor**: Statistical features, Hjorth parameters
- ✅ **Feature Pipeline**: Modular multi-extractor pipeline
- ✅ **LDA Classifier**: Linear Discriminant Analysis with shrinkage
- ✅ **SVM Classifier**: Support Vector Machine (RBF, linear kernels)
- ✅ **EEGNet Classifier**: Compact CNN for end-to-end EEG classification
- ✅ **Classifier Factory**: Dynamic classifier creation
- ✅ **Unit Tests**: 71 tests (33 features + 38 classifiers)
- ✅ **Colab Notebook**: Phase3_Feature_Extraction_Classification.ipynb

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

### v2.0.0 (Phase 2) - Data Loading & Processing
- **Data Loaders**: MATLoader for BCI Competition IV-2a .mat files
- **Factory Pattern**: DataLoaderFactory with auto-detection by file extension
- **Preprocessing Pipeline**: Composable, sequential preprocessing
- **Filters**: BandpassFilter (Butterworth), NotchFilter (IIR)
- **Normalization**: Z-score, min-max, robust, L2 normalization
- **Validation**: DataValidator for structure/format validation
- **Quality Assessment**: QualityChecker (SNR, artifacts, line noise)
- **PyTorch Integration**: EEGDataset, BCICIV2aDataset classes
- **Data Splitting**: train_val_test_split, create_cv_folds
- **Unit Tests**: Comprehensive test coverage for all components

### v1.0.0 (Phase 1) - Foundation & Setup
- Initial release with core framework architecture
- 9 abstract interfaces for all components
- Core data types for EEG handling
- Configuration manager with user-approved defaults
- Component registry for plugin architecture
- Google Colab compatibility
