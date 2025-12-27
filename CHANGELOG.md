# Changelog

All notable changes to the LLM-EEG Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2024-12-07 - Phase 1: Foundation & Setup

### Added

#### Core Interfaces (`src/core/interfaces/`)
- `IDataLoader`: Abstract interface for EEG data loading from various formats (.mat, .gdf, .edf, MOABB)
- `IPreprocessor`: Interface for signal preprocessing steps (filtering, artifact removal)
- `IFeatureExtractor`: Interface for feature extraction methods (CSP, band power, wavelet)
- `IClassifier`: Interface for classification models (traditional ML and deep learning)
- `IAgent`: Interface for AI agents with observe-act-learn cycle
- `IPolicy`: Interface for RL policies (Q-learning, DQN) with state discretization helpers
- `IReward`: Interface for reward functions with composite reward support
- `ILLMProvider`: Interface for LLM providers with BCI-specific methods
- `IStorageAdapter`: Interface for storage backends with checkpointing

#### Core Data Types (`src/core/types/`)
- `EEGData`: Container for continuous EEG recordings with trial extraction
- `TrialData`: Single trial representation with signal, label, and metadata
- `EventMarker`: Event/stimulus marker with sample position and code
- `DatasetInfo`: Dataset-level metadata with BCI Competition IV-2a preset

#### Configuration System (`src/core/config.py`)
- `ConfigManager`: Singleton configuration manager with hierarchical settings
- User-approved default values:
  - DVA confidence threshold: 0.8
  - APA policy type: q_learning (RL-based)
  - Cross-trial learning: enabled
  - LLM provider: phi3 (Phi-3-mini-4k-instruct)
- Google Drive dataset configuration
- Dot-notation access (e.g., `config.get('agents.dva.confidence_threshold')`)

#### Component Registry (`src/core/registry.py`)
- `ComponentRegistry`: Singleton registry for plugin architecture
- Dynamic component instantiation
- Factory function support
- Component discovery and auto-registration
- `@registered` decorator for automatic registration

#### Custom Exceptions (`src/core/exceptions/`)
- `BCIFrameworkError`: Base exception class
- Data exceptions: `DataLoadError`, `DataValidationError`, `DataFormatError`, `MissingDataError`, `ChannelNotFoundError`
- Processing exceptions: `PreprocessingError`, `FeatureExtractionError`, `FilterError`
- Classification exceptions: `ModelNotFittedError`, `ModelNotFoundError`, `PredictionError`
- Agent exceptions: `AgentNotInitializedError`, `PolicyError`, `RewardError`, `InvalidStateError`
- LLM exceptions: `LLMNotLoadedError`, `GenerationError`, `PromptError`
- Configuration exceptions: `ConfigNotFoundError`, `ConfigValidationError`, `MissingConfigError`
- Storage exceptions: `StorageReadError`, `StorageWriteError`, `CheckpointError`
- Component exceptions: `ComponentNotFoundError`, `RegistrationError`, `InitializationError`

#### Utilities (`src/utils/`)
- `logging.py`: Centralized logging with colored console output, file logging, performance decorators
- `validation.py`: Input validation functions for arrays, configs, types, and ranges

#### Project Structure
- Complete directory structure for all phases
- Empty `__init__.py` files for all modules
- `.gitkeep` files for empty data directories
- `.gitignore` for Python projects

#### Documentation
- Comprehensive README with installation and usage instructions
- Google Colab setup guide
- Configuration reference
- Troubleshooting section

### Configuration Defaults

```yaml
agents:
  apa:
    policy:
      type: q_learning
      learning_rate: 0.1
      discount_factor: 0.99
      epsilon_start: 1.0
      epsilon_decay: 0.995
      epsilon_min: 0.01
    state_bins:
      snr: [0, 5, 10, 20, inf]
      artifact_ratio: [0, 0.1, 0.3, 0.5, 1.0]
      line_noise: [0, 0.5, 1.0, 2.0, inf]
    action_space: [conservative, moderate, aggressive]
    cross_trial_learning: true
  dva:
    confidence_threshold: 0.8
    adaptive_threshold: true
    validators: [confidence, margin, signal_quality, historical_consistency]
    cross_trial_learning: true

llm:
  provider: phi3
  model_path: microsoft/phi-3-mini-4k-instruct
  quantization: 4bit

data:
  sampling_rate: 250
  n_channels: 22
  n_classes: 4
  google_drive:
    folder_url: https://drive.google.com/drive/folders/14tFFsegwr6oYF4wUuf_mjNOAgfuQ_Bwk
```

### Fixed
- Changed absolute imports to relative imports for Google Colab compatibility

---

## [2.0.0] - 2024-12-07 - Phase 2: Data Loading & Processing

**Branch: `phase-2`**

### Added

#### Google Colab Notebook (`notebooks/`)
- `Phase2_Data_Loading_Processing.ipynb`: Complete Phase 2 tutorial notebook
  - Step-by-step environment setup for Google Colab
  - Custom `BCICIV2aLoader` for BCI Competition IV-2a MAT files
  - Data validation and quality assessment examples
  - Preprocessing pipeline demonstration with visualizations
  - PyTorch dataset integration and DataLoader setup
  - Multi-subject loading with LOSO cross-validation
  - Simple CNN training example with evaluation metrics

#### Data Loaders (`src/data/loaders/`)
- `BaseDataLoader`: Abstract base class with common loading functionality
- `MATLoader`: MATLAB .mat file loader optimized for BCI Competition IV-2a
  - Automatic signal/event extraction
  - Support for scipy, mat73, and pymatreader libraries
  - Channel selection and EOG exclusion
  - Event code mapping for motor imagery classes
- `DataLoaderFactory`: Factory pattern for loader creation
  - Auto-detection by file extension
  - Configurable initialization
  - Custom loader registration support
- Convenience functions: `load_eeg_file()`, `create_loader()`, `create_mat_loader()`

#### Preprocessing (`src/preprocessing/`)
- `PreprocessingPipeline`: Composable preprocessing pipeline
  - Sequential step execution
  - Step-wise debugging
  - Execution time tracking
  - Configuration from dict/YAML
- `BandpassFilter`: Butterworth bandpass filter
  - Zero-phase filtering (filtfilt)
  - Configurable passband (default: 8-30 Hz for motor imagery)
  - Support for 2D and 3D arrays
- `NotchFilter`: IIR notch filter for line noise removal
  - 50/60 Hz power line removal
  - Harmonic removal support
  - Configurable quality factor
- `Normalization`: Signal normalization
  - Methods: z-score, min-max, robust, L2
  - Axes: channel-wise, trial-wise, global, sample-wise
- Factory function: `create_standard_pipeline()`

#### Data Validators (`src/data/validators/`)
- `DataValidator`: Comprehensive data validation
  - Structure validation (shapes, types)
  - Value validation (NaN, Inf, amplitude)
  - Format validation (channels, sampling rate, events)
  - BCI Competition IV-2a specific validation
- `ValidationResult`: Container for validation results
- `ValidationError`: Custom exception for validation failures
- `QualityChecker`: Signal quality assessment
  - SNR computation (signal vs noise band power)
  - Line noise level detection
  - Artifact ratio calculation
  - Flatline channel detection
  - Per-channel quality scores
  - Trial-level quality assessment
  - Clean trial filtering

#### PyTorch Datasets (`src/datasets/`)
- `EEGDataset`: Base PyTorch Dataset for EEG trials
  - Support for numpy arrays and TrialData lists
  - Optional transforms
  - Return tensors or numpy arrays
- `BCICIV2aDataset`: Specialized for BCI Competition IV-2a
  - `from_subject()`: Load single subject
  - `from_subjects()`: Load multiple subjects
  - Subject/session metadata tracking
  - Class name mapping
- Utility functions:
  - `train_val_test_split()`: Split dataset
  - `create_cv_folds()`: Cross-validation folds

#### Unit Tests (`tests/unit/`)
- `test_data_loaders.py`: Tests for data loading
- `test_preprocessing.py`: Tests for preprocessing pipeline
- `test_validators.py`: Tests for validation and quality checking

#### Constants
- `BCI_IV_2A_EEG_CHANNELS`: 22 standard EEG channel names
- `BCI_IV_2A_EOG_CHANNELS`: 3 EOG channel names
- `BCI_IV_2A_ALL_CHANNELS`: All 25 channel names
- `BCI_IV_2A_EVENT_CODES`: Event code to label mapping
- `BCI_IV_2A_CLASS_MAPPING`: Event code to class index mapping
- `BCI_IV_2A_SAMPLING_RATE`: 250 Hz
- `BCI_IV_2A_TRIALS_PER_SESSION`: 288

### Changed
- Updated README with Phase 2 usage examples
- Updated project structure documentation

---

## [3.0.0] - 2024-12-27 - Phase 3: Feature Extraction & Classification

**Branch: `phase-3`**

### Added

#### Feature Extractors (`src/features/`)

##### Base Classes
- `BaseFeatureExtractor`: Abstract base class implementing IFeatureExtractor
  - Common initialization and validation
  - State management (save/load)
  - Feature name generation
  - Batch processing support

##### CSP Extractor (`extractors/csp.py`)
- `CSPExtractor`: Common Spatial Pattern feature extraction
  - SciPy-based implementation (lightweight, no MNE dependency)
  - Configurable n_components (default: 6)
  - Regularization support (Tikhonov, Ledoit-Wolf)
  - Multi-class support via One-vs-Rest
  - Spatial filter and pattern visualization
  - Log-variance feature computation
  - Save/load functionality

##### Band Power Extractor (`extractors/band_power.py`)
- `BandPowerExtractor`: Frequency band power extraction
  - Welch PSD method
  - Default bands: mu (8-12 Hz), beta (12-30 Hz)
  - Custom band configuration
  - Channel averaging option
  - Relative power support
  - Log transformation

##### Time Domain Extractor (`extractors/time_domain.py`)
- `TimeDomainExtractor`: Statistical features
  - Mean, variance, std, skewness, kurtosis
  - RMS, peak-to-peak, min, max
  - Zero crossings, line length
  - Hjorth parameters (activity, mobility, complexity)
  - Per-channel or average computation

##### Feature Factory (`factory.py`)
- `FeatureExtractorFactory`: Dynamic extractor creation
  - Registry-based creation
  - Configuration-driven instantiation
  - Custom extractor registration
  - `create()`, `create_from_config()`, `list_available()`

##### Feature Pipeline (`pipeline.py`)
- `FeatureExtractionPipeline`: Multi-extractor pipeline
  - Concatenation mode (combine features)
  - Sequential and parallel modes
  - Fit/extract workflow
  - Pipeline serialization
  - Configuration-based creation
  - `create_motor_imagery_pipeline()` convenience function

#### Classifiers (`src/classifiers/`)

##### Base Classes
- `BaseClassifier`: Abstract base class implementing IClassifier
  - Common initialization and validation
  - Training history tracking
  - State management (save/load)
  - Phase-4 integration hooks
- `BaseDeepClassifier`: Base for PyTorch classifiers
  - Device management (CPU/CUDA)
  - Training loop with early stopping
  - Learning rate scheduling
  - Gradient clipping
  - Model checkpointing

##### LDA Classifier (`models/traditional/lda.py`)
- `LDAClassifier`: Linear Discriminant Analysis
  - sklearn-based implementation
  - Solver options: svd, lsqr, eigen
  - Shrinkage regularization
  - Feature scaling option
  - Decision function for DVA integration
  - Dimensionality reduction (transform)

##### SVM Classifier (`models/traditional/svm.py`)
- `SVMClassifier`: Support Vector Machine
  - Kernel options: rbf, linear, poly, sigmoid
  - Hyperparameters: C, gamma, degree
  - Probability estimation (Platt scaling)
  - Support vector information
  - Margin information for DVA
  - Class balancing support

##### EEGNet Classifier (`models/deep_learning/eegnet.py`)
- `EEGNetClassifier`: Compact CNN for EEG
  - EEGNet architecture (Lawhern et al., 2018)
  - Temporal convolution (frequency filters)
  - Depthwise convolution (spatial filters)
  - Separable convolution (temporal patterns)
  - Configurable F1, D, F2 parameters
  - Feature map extraction for DVA
  - Spatial/temporal pattern visualization

##### Classifier Factory (`factory.py`)
- `ClassifierFactory`: Dynamic classifier creation
  - Traditional ML: 'lda', 'svm'
  - Deep Learning: 'eegnet'
  - Default configurations
  - Pipeline-specific configurations
  - `get_classifier_for_pipeline()` convenience function

#### Google Colab Notebook
- `Phase3_Feature_Extraction_Classification.ipynb`: Complete Phase 3 tutorial
  - CSP feature extraction with visualization
  - Band power and time domain features
  - Feature pipeline combination
  - LDA, SVM, EEGNet classification
  - Model comparison (accuracy, kappa)
  - LOSO cross-validation setup
  - Results export for AI assistant sharing

#### Unit Tests (`tests/unit/`)
- `test_feature_extractors.py`: 33 tests for feature extractors
  - CSP: creation, fit, extract, save/load, multiclass
  - BandPower: bands, averaging, PSD
  - TimeDomain: statistics, Hjorth parameters
  - Factory and Pipeline tests
- `test_classifiers.py`: 38 tests for classifiers
  - LDA: fit, predict, decision function, regularization
  - SVM: kernels, support vectors, margins
  - EEGNet: training, inference, feature maps
  - Factory and integration tests

### Performance Benchmarks (BCI Competition IV-2a)
- CSP + LDA: ~75-80% accuracy
- CSP + SVM: ~78-82% accuracy
- EEGNet: ~70-75% accuracy
- CSP + EEGNet: ~82-85% accuracy

### Changed
- Updated README with Phase 3 usage examples
- Updated project version to 3.0.0

---

## Upcoming

### [3.1.0] - Phase 4: Agent System (Planned)
- Adaptive Preprocessing Agent (APA) implementation
- Decision Validation Agent (DVA) implementation
- Q-learning policy
- Reward functions
- Cross-trial learning

### [4.0.0] - Phase 5: LLM Integration (Planned)
- Phi-3 provider implementation
- Agent explanation generation
- Semantic feature encoding

### [5.0.0] - Phase 6: Evaluation & Documentation (Planned)
- Comprehensive metrics
- Ablation studies
- Final documentation
- Google Colab notebooks
