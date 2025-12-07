"""
PyTorch EEG Dataset
===================

This module provides PyTorch Dataset implementations for EEG data,
enabling seamless integration with PyTorch DataLoader for training
deep learning models.

Dataset Classes:
---------------
- EEGDataset: Base dataset for EEG trials
- BCICIV2aDataset: Specialized for BCI Competition IV-2a
- SubjectDataset: Dataset for single subject data

Features:
---------
- Automatic trial extraction from continuous data
- Optional preprocessing on-the-fly
- Data augmentation support
- Train/val/test splitting
- Cross-validation fold generation

PyTorch Integration:
-------------------
The datasets are compatible with PyTorch DataLoader:
```python
from torch.utils.data import DataLoader
dataset = EEGDataset(trials, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Usage Example:
    ```python
    from src.datasets import BCICIV2aDataset
    
    # Create dataset from file
    dataset = BCICIV2aDataset.from_file(
        'data/A01T.mat',
        trial_length_sec=4.0,
        preprocessing_pipeline=pipeline
    )
    
    # Use with DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for batch_x, batch_y in loader:
        # batch_x: (batch, channels, samples)
        # batch_y: (batch,)
        outputs = model(batch_x)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import numpy as np
import logging
from pathlib import Path

# PyTorch imports (optional for Google Colab compatibility)
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy base class
    class Dataset:
        pass

from src.core.types.eeg_data import EEGData, TrialData


# Configure module logger
logger = logging.getLogger(__name__)


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG trial data.
    
    This dataset wraps EEG trials for use with PyTorch's DataLoader.
    It supports preprocessing transforms and data augmentation.
    
    Attributes:
        _trials (np.ndarray): Trial data with shape (n_trials, n_channels, n_samples)
        _labels (np.ndarray): Labels with shape (n_trials,)
        _transform (Callable): Optional transform to apply to each sample
        _label_transform (Callable): Optional transform for labels
        _return_numpy (bool): Return numpy arrays instead of tensors
    
    Data Format:
        - Input: numpy arrays or list of TrialData
        - Output: (trial_tensor, label_tensor) or (trial_array, label)
    
    Example:
        >>> dataset = EEGDataset(X_train, y_train)
        >>> x, y = dataset[0]  # Get first sample
        >>> print(x.shape)  # (n_channels, n_samples)
    """
    
    def __init__(
        self,
        trials: Union[np.ndarray, List[TrialData]],
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        return_numpy: bool = False
    ):
        """
        Initialize the EEG dataset.
        
        Args:
            trials: Trial data as:
                - numpy array: Shape (n_trials, n_channels, n_samples)
                - List of TrialData objects
            labels: Labels array (n_trials,). If None, extracted from TrialData
            transform: Optional transform function for trials
            label_transform: Optional transform for labels
            return_numpy: If True, return numpy arrays instead of tensors
        
        Raises:
            ValueError: If data format is invalid
            ImportError: If PyTorch is not available and return_numpy=False
        """
        if not TORCH_AVAILABLE and not return_numpy:
            raise ImportError(
                "PyTorch is not available. Install with 'pip install torch' "
                "or set return_numpy=True"
            )
        
        # Convert TrialData list to arrays
        if isinstance(trials, list) and len(trials) > 0:
            if isinstance(trials[0], TrialData):
                self._trials = np.array([t.signals for t in trials])
                if labels is None:
                    labels = np.array([t.label for t in trials])
            else:
                self._trials = np.array(trials)
        else:
            self._trials = np.array(trials)
        
        # Validate trials shape
        if self._trials.ndim != 3:
            raise ValueError(
                f"Trials must be 3D (n_trials, n_channels, n_samples), "
                f"got {self._trials.ndim}D with shape {self._trials.shape}"
            )
        
        # Handle labels
        if labels is None:
            raise ValueError("Labels must be provided")
        self._labels = np.array(labels).flatten()
        
        if len(self._labels) != len(self._trials):
            raise ValueError(
                f"Number of labels ({len(self._labels)}) doesn't match "
                f"number of trials ({len(self._trials)})"
            )
        
        # Store transforms
        self._transform = transform
        self._label_transform = label_transform
        self._return_numpy = return_numpy
        
        # Metadata
        self._n_trials, self._n_channels, self._n_samples = self._trials.shape
        self._n_classes = len(np.unique(self._labels))
        
        logger.info(
            f"EEGDataset created: {self._n_trials} trials, "
            f"{self._n_channels} channels, {self._n_samples} samples, "
            f"{self._n_classes} classes"
        )
    
    def __len__(self) -> int:
        """Return the number of trials in the dataset."""
        return self._n_trials
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Get a single trial and its label.
        
        Args:
            idx: Trial index
        
        Returns:
            Tuple of (trial_data, label):
                - trial_data: Tensor or array (n_channels, n_samples)
                - label: Tensor or int
        """
        # Get trial and label
        trial = self._trials[idx].astype(np.float32)
        label = self._labels[idx]
        
        # Apply transforms
        if self._transform is not None:
            trial = self._transform(trial)
        
        if self._label_transform is not None:
            label = self._label_transform(label)
        
        # Convert to tensors if needed
        if not self._return_numpy and TORCH_AVAILABLE:
            trial = torch.from_numpy(trial) if isinstance(trial, np.ndarray) else trial
            label = torch.tensor(label, dtype=torch.long)
        
        return trial, label
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def n_trials(self) -> int:
        """Number of trials in the dataset."""
        return self._n_trials
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self._n_channels
    
    @property
    def n_samples(self) -> int:
        """Number of samples per trial."""
        return self._n_samples
    
    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return self._n_classes
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the data (n_trials, n_channels, n_samples)."""
        return (self._n_trials, self._n_channels, self._n_samples)
    
    @property
    def class_distribution(self) -> Dict[int, int]:
        """Distribution of classes."""
        unique, counts = np.unique(self._labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    # =========================================================================
    # DATA ACCESS
    # =========================================================================
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all data as numpy arrays.
        
        Returns:
            Tuple of (trials, labels) numpy arrays
        """
        return self._trials.copy(), self._labels.copy()
    
    def get_subset(self, indices: List[int]) -> 'EEGDataset':
        """
        Create a new dataset with a subset of trials.
        
        Args:
            indices: List of trial indices to include
        
        Returns:
            New EEGDataset with selected trials
        """
        return EEGDataset(
            trials=self._trials[indices],
            labels=self._labels[indices],
            transform=self._transform,
            label_transform=self._label_transform,
            return_numpy=self._return_numpy
        )
    
    def get_class_subset(self, classes: List[int]) -> 'EEGDataset':
        """
        Create a new dataset with only specified classes.
        
        Args:
            classes: List of class labels to include
        
        Returns:
            New EEGDataset with selected classes
        """
        mask = np.isin(self._labels, classes)
        indices = np.where(mask)[0]
        return self.get_subset(indices.tolist())
    
    # =========================================================================
    # CLASS METHODS
    # =========================================================================
    
    @classmethod
    def from_eegdata(
        cls,
        eeg_data: EEGData,
        trial_length_sec: float = 4.0,
        pre_stimulus_sec: float = 0.0,
        class_mapping: Optional[Dict[int, int]] = None,
        **kwargs
    ) -> 'EEGDataset':
        """
        Create dataset from EEGData object.
        
        Extracts trials based on event markers and creates a dataset.
        
        Args:
            eeg_data: EEGData object with events
            trial_length_sec: Length of each trial in seconds
            pre_stimulus_sec: Time before event to include
            class_mapping: Mapping from event codes to class indices
            **kwargs: Additional arguments passed to __init__
        
        Returns:
            EEGDataset with extracted trials
        
        Example:
            >>> dataset = EEGDataset.from_eegdata(eeg_data, trial_length_sec=4.0)
        """
        # Default class mapping for BCI Competition IV-2a
        if class_mapping is None:
            class_mapping = {
                769: 0,  # left_hand
                770: 1,  # right_hand
                771: 2,  # feet
                772: 3   # tongue
            }
        
        # Extract trials
        X, y = eeg_data.get_trials_array(
            trial_length_sec=trial_length_sec,
            pre_stimulus_sec=pre_stimulus_sec,
            class_mapping={k: f'class_{v}' for k, v in class_mapping.items()}
        )
        
        return cls(trials=X, labels=y, **kwargs)
    
    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        trial_length_sec: float = 4.0,
        include_eog: bool = False,
        preprocessing_pipeline=None,
        **kwargs
    ) -> 'EEGDataset':
        """
        Create dataset directly from a file.
        
        Loads the file, optionally applies preprocessing, and
        extracts trials.
        
        Args:
            file_path: Path to the EEG data file
            trial_length_sec: Length of each trial
            include_eog: Whether to include EOG channels
            preprocessing_pipeline: Optional preprocessing to apply
            **kwargs: Additional arguments
        
        Returns:
            EEGDataset with trials from the file
        
        Example:
            >>> dataset = EEGDataset.from_file('A01T.mat', trial_length_sec=4.0)
        """
        from src.data.loaders import DataLoaderFactory
        
        # Load data
        loader = DataLoaderFactory.create_for_file(
            file_path,
            config={'include_eog': include_eog}
        )
        eeg_data = loader.load(file_path)
        
        # Apply preprocessing if provided
        if preprocessing_pipeline is not None:
            eeg_data = preprocessing_pipeline.process(eeg_data)
        
        return cls.from_eegdata(
            eeg_data,
            trial_length_sec=trial_length_sec,
            **kwargs
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EEGDataset(trials={self._n_trials}, "
            f"channels={self._n_channels}, "
            f"samples={self._n_samples}, "
            f"classes={self._n_classes})"
        )


class BCICIV2aDataset(EEGDataset):
    """
    Specialized dataset for BCI Competition IV-2a.
    
    This class extends EEGDataset with specific functionality for
    the BCI Competition IV-2a motor imagery dataset.
    
    Features:
        - Standard 22 EEG channel support
        - 4-class motor imagery (left, right, feet, tongue)
        - Subject and session metadata
        - Cross-validation split generation
    
    Example:
        >>> # Load single subject
        >>> dataset = BCICIV2aDataset.from_subject(
        ...     subject_id=1,
        ...     session='T',
        ...     data_dir='/content/drive/MyDrive/BCI_IV_2a'
        ... )
        
        >>> # Load multiple subjects
        >>> dataset = BCICIV2aDataset.from_subjects(
        ...     subjects=[1, 2, 3],
        ...     session='T',
        ...     data_dir=data_dir
        ... )
    """
    
    # Class labels
    CLASS_NAMES: List[str] = ['left_hand', 'right_hand', 'feet', 'tongue']
    N_SUBJECTS: int = 9
    N_CLASSES: int = 4
    SAMPLING_RATE: float = 250.0
    
    def __init__(
        self,
        trials: Union[np.ndarray, List[TrialData]],
        labels: np.ndarray,
        subject_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize BCI Competition IV-2a dataset.
        
        Args:
            trials: Trial data
            labels: Labels
            subject_ids: Subject ID for each trial
            session_ids: Session ID for each trial
            **kwargs: Additional arguments for EEGDataset
        """
        super().__init__(trials, labels, **kwargs)
        
        # Store metadata
        self._subject_ids = subject_ids
        self._session_ids = session_ids
    
    @classmethod
    def from_subject(
        cls,
        subject_id: int,
        session: str,
        data_dir: Union[str, Path],
        trial_length_sec: float = 4.0,
        preprocessing_pipeline=None,
        **kwargs
    ) -> 'BCICIV2aDataset':
        """
        Load dataset for a single subject.
        
        Args:
            subject_id: Subject number (1-9)
            session: Session type ('T' for training, 'E' for evaluation)
            data_dir: Directory containing the MAT files
            trial_length_sec: Trial length in seconds
            preprocessing_pipeline: Optional preprocessing
            **kwargs: Additional arguments
        
        Returns:
            BCICIV2aDataset for the specified subject/session
        
        Example:
            >>> dataset = BCICIV2aDataset.from_subject(
            ...     subject_id=1,
            ...     session='T',
            ...     data_dir='data/raw'
            ... )
        """
        # Construct file path
        data_dir = Path(data_dir)
        file_name = f"A0{subject_id}{session}.mat"
        file_path = data_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading subject {subject_id}, session {session}")
        
        # Load using parent class method
        from src.data.loaders import MATLoader
        
        loader = MATLoader()
        loader.initialize({'include_eog': False})
        eeg_data = loader.load(file_path)
        
        # Apply preprocessing
        if preprocessing_pipeline is not None:
            eeg_data = preprocessing_pipeline.process(eeg_data)
        
        # Extract trials
        X, y = eeg_data.get_trials_array(trial_length_sec=trial_length_sec)
        
        # Create subject/session arrays
        n_trials = len(y)
        subject_ids = [f"S{subject_id:02d}"] * n_trials
        session_ids = [session] * n_trials
        
        return cls(
            trials=X,
            labels=y,
            subject_ids=subject_ids,
            session_ids=session_ids,
            **kwargs
        )
    
    @classmethod
    def from_subjects(
        cls,
        subjects: List[int],
        session: str,
        data_dir: Union[str, Path],
        **kwargs
    ) -> 'BCICIV2aDataset':
        """
        Load and combine data from multiple subjects.
        
        Args:
            subjects: List of subject numbers (1-9)
            session: Session type ('T' or 'E')
            data_dir: Directory containing files
            **kwargs: Additional arguments
        
        Returns:
            Combined BCICIV2aDataset
        """
        all_trials = []
        all_labels = []
        all_subject_ids = []
        all_session_ids = []
        
        for subject_id in subjects:
            ds = cls.from_subject(
                subject_id=subject_id,
                session=session,
                data_dir=data_dir,
                **kwargs
            )
            
            X, y = ds.get_data()
            all_trials.append(X)
            all_labels.append(y)
            all_subject_ids.extend(ds._subject_ids or [f"S{subject_id:02d}"] * len(y))
            all_session_ids.extend(ds._session_ids or [session] * len(y))
        
        # Concatenate
        combined_trials = np.concatenate(all_trials, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        return cls(
            trials=combined_trials,
            labels=combined_labels,
            subject_ids=all_subject_ids,
            session_ids=all_session_ids,
            **{k: v for k, v in kwargs.items() if k not in ['trial_length_sec', 'preprocessing_pipeline']}
        )
    
    def get_subject_subset(self, subject_id: str) -> 'BCICIV2aDataset':
        """
        Get subset of trials for a specific subject.
        
        Args:
            subject_id: Subject ID (e.g., 'S01')
        
        Returns:
            BCICIV2aDataset with only that subject's trials
        """
        if self._subject_ids is None:
            raise ValueError("Subject IDs not available")
        
        indices = [
            i for i, sid in enumerate(self._subject_ids)
            if sid == subject_id
        ]
        
        return BCICIV2aDataset(
            trials=self._trials[indices],
            labels=self._labels[indices],
            subject_ids=[self._subject_ids[i] for i in indices],
            session_ids=[self._session_ids[i] for i in indices] if self._session_ids else None,
            transform=self._transform,
            label_transform=self._label_transform,
            return_numpy=self._return_numpy
        )
    
    def get_label_name(self, label: int) -> str:
        """
        Get the human-readable name for a label.
        
        Args:
            label: Class label (0-3)
        
        Returns:
            Class name string
        """
        if 0 <= label < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[label]
        return f"unknown_{label}"
    
    def __repr__(self) -> str:
        """String representation."""
        n_subjects = len(set(self._subject_ids)) if self._subject_ids else "?"
        return (
            f"BCICIV2aDataset(trials={self._n_trials}, "
            f"subjects={n_subjects}, "
            f"classes={self._n_classes})"
        )


# =============================================================================
# DATA SPLITTING UTILITIES
# =============================================================================

def train_val_test_split(
    dataset: EEGDataset,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[EEGDataset, ...]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset: Dataset to split
        val_ratio: Proportion for validation
        test_ratio: Proportion for test (0 for no test set)
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset) or
        (train_dataset, val_dataset, test_dataset) if test_ratio > 0
    
    Example:
        >>> train_ds, val_ds = train_val_test_split(dataset, val_ratio=0.2)
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    # Calculate split points
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val - n_test
    
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    
    train_ds = dataset.get_subset(train_indices)
    val_ds = dataset.get_subset(val_indices)
    
    if test_ratio > 0:
        test_indices = indices[n_train + n_val:].tolist()
        test_ds = dataset.get_subset(test_indices)
        return train_ds, val_ds, test_ds
    
    return train_ds, val_ds


def create_cv_folds(
    dataset: EEGDataset,
    n_folds: int = 5,
    shuffle: bool = True,
    random_seed: int = 42
) -> List[Tuple[EEGDataset, EEGDataset]]:
    """
    Create cross-validation folds.
    
    Args:
        dataset: Dataset to split
        n_folds: Number of folds
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed
    
    Returns:
        List of (train_dataset, val_dataset) tuples
    
    Example:
        >>> folds = create_cv_folds(dataset, n_folds=5)
        >>> for train_ds, val_ds in folds:
        ...     # Train and evaluate
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    # Create folds
    fold_size = n_samples // n_folds
    folds = []
    
    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        val_indices = indices[start:end].tolist()
        train_indices = np.concatenate([
            indices[:start],
            indices[end:]
        ]).tolist()
        
        train_ds = dataset.get_subset(train_indices)
        val_ds = dataset.get_subset(val_indices)
        
        folds.append((train_ds, val_ds))
    
    return folds
