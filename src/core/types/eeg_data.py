"""
EEG Data Types
==============

This module defines the core data types for representing EEG data in the BCI framework.

Data Types:
----------
1. EEGData: Container for continuous EEG recordings
2. TrialData: Single trial with signal, label, and metadata
3. DatasetInfo: Dataset-level metadata
4. EventMarker: Event/stimulus marker

Design Principles:
-----------------
- Immutable after creation (for thread safety)
- Rich metadata for reproducibility
- Numpy-backed for performance
- Serializable for persistence

BCI Competition IV-2a Specifications:
------------------------------------
- Sampling Rate: 250 Hz
- Channels: 22 EEG + 3 EOG
- Classes: 4 (left hand, right hand, feet, tongue)
- Trials: 288 per subject per session
- Duration: 4 seconds per trial

Example Usage:
    ```python
    from src.core.types.eeg_data import EEGData, TrialData
    
    # Create from numpy arrays
    eeg_data = EEGData(
        signals=signals,  # Shape: (n_channels, n_samples)
        sampling_rate=250,
        channel_names=['C3', 'C4', 'Cz', ...],
        events=events  # List of EventMarker
    )
    
    # Access properties
    print(f"Duration: {eeg_data.duration_seconds:.1f}s")
    print(f"Channels: {eeg_data.n_channels}")
    
    # Extract trials
    trials = eeg_data.extract_trials(trial_length_sec=4.0)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from datetime import datetime
from pathlib import Path


@dataclass(frozen=False)
class EventMarker:
    """
    Represents a single event/stimulus marker in EEG data.
    
    Attributes:
        sample: Sample index where event occurred
        code: Event code/type (e.g., 1=left_hand, 2=right_hand)
        label: Human-readable label
        duration_samples: Event duration in samples (optional)
        metadata: Additional event metadata
    """
    sample: int
    code: int
    label: str = ''
    duration_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def time_seconds(self) -> float:
        """Event time in seconds (requires sampling_rate in metadata)."""
        sr = self.metadata.get('sampling_rate', 250)
        return self.sample / sr
    
    def __repr__(self) -> str:
        return f"EventMarker(sample={self.sample}, code={self.code}, label='{self.label}')"


@dataclass(frozen=False)
class DatasetInfo:
    """
    Metadata about a dataset.
    
    Attributes:
        name: Dataset name
        description: Dataset description
        n_subjects: Number of subjects
        n_sessions: Sessions per subject
        n_trials: Trials per session
        n_classes: Number of classes
        class_names: Names of classes
        sampling_rate: Sampling frequency in Hz
        channel_names: List of channel names
        source: Data source (file path, URL, etc.)
        created_at: Creation timestamp
    """
    name: str
    description: str = ''
    n_subjects: int = 1
    n_sessions: int = 1
    n_trials: int = 0
    n_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    sampling_rate: float = 250.0
    channel_names: List[str] = field(default_factory=list)
    source: str = ''
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def for_bci_competition_iv_2a(cls) -> 'DatasetInfo':
        """Create DatasetInfo for BCI Competition IV-2a dataset."""
        return cls(
            name='BCI Competition IV-2a',
            description='Motor imagery EEG dataset with 4 classes',
            n_subjects=9,
            n_sessions=2,
            n_trials=288,
            n_classes=4,
            class_names=['left_hand', 'right_hand', 'feet', 'tongue'],
            sampling_rate=250.0,
            channel_names=[
                'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2', 'POz'
            ],
            source='http://www.bbci.de/competition/iv/',
            metadata={
                'eog_channels': ['EOG-left', 'EOG-central', 'EOG-right'],
                'trial_duration_sec': 4.0,
                'inter_trial_interval_sec': [1.5, 2.5],
                'paradigm': 'cue-based motor imagery'
            }
        )


@dataclass
class TrialData:
    """
    Represents a single trial of EEG data.
    
    Attributes:
        signals: EEG signal data, shape (n_channels, n_samples)
        label: Class label (0-indexed)
        label_name: Human-readable label name
        trial_id: Unique trial identifier
        subject_id: Subject identifier
        session_id: Session identifier
        sampling_rate: Sampling frequency in Hz
        channel_names: List of channel names
        start_sample: Starting sample in original recording
        metadata: Additional trial metadata
    """
    signals: np.ndarray  # Shape: (n_channels, n_samples)
    label: int
    label_name: str = ''
    trial_id: int = 0
    subject_id: str = ''
    session_id: str = ''
    sampling_rate: float = 250.0
    channel_names: List[str] = field(default_factory=list)
    start_sample: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.signals.ndim != 2:
            raise ValueError(f"signals must be 2D (channels, samples), got {self.signals.ndim}D")
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.signals.shape[0]
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.signals.shape[1]
    
    @property
    def duration_seconds(self) -> float:
        """Trial duration in seconds."""
        return self.n_samples / self.sampling_rate
    
    def get_channel(self, channel: Union[int, str]) -> np.ndarray:
        """
        Get data for a specific channel.
        
        Args:
            channel: Channel index or name
        
        Returns:
            np.ndarray: Channel data, shape (n_samples,)
        """
        if isinstance(channel, str):
            if channel not in self.channel_names:
                raise ValueError(f"Channel '{channel}' not found")
            idx = self.channel_names.index(channel)
        else:
            idx = channel
        return self.signals[idx]
    
    def get_time_axis(self) -> np.ndarray:
        """Get time axis in seconds."""
        return np.arange(self.n_samples) / self.sampling_rate
    
    def copy(self) -> 'TrialData':
        """Create a copy of this trial."""
        return TrialData(
            signals=self.signals.copy(),
            label=self.label,
            label_name=self.label_name,
            trial_id=self.trial_id,
            subject_id=self.subject_id,
            session_id=self.session_id,
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names.copy(),
            start_sample=self.start_sample,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signals': self.signals.tolist(),
            'label': self.label,
            'label_name': self.label_name,
            'trial_id': self.trial_id,
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'sampling_rate': self.sampling_rate,
            'channel_names': self.channel_names,
            'start_sample': self.start_sample,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrialData':
        """Create from dictionary."""
        return cls(
            signals=np.array(data['signals']),
            label=data['label'],
            label_name=data.get('label_name', ''),
            trial_id=data.get('trial_id', 0),
            subject_id=data.get('subject_id', ''),
            session_id=data.get('session_id', ''),
            sampling_rate=data.get('sampling_rate', 250.0),
            channel_names=data.get('channel_names', []),
            start_sample=data.get('start_sample', 0),
            metadata=data.get('metadata', {})
        )
    
    def __repr__(self) -> str:
        return (
            f"TrialData(shape={self.signals.shape}, "
            f"label={self.label} ({self.label_name}), "
            f"duration={self.duration_seconds:.2f}s)"
        )


@dataclass
class EEGData:
    """
    Container for continuous EEG data with metadata.
    
    Attributes:
        signals: EEG signal data, shape (n_channels, n_samples)
        sampling_rate: Sampling frequency in Hz
        channel_names: List of channel names
        events: List of event markers
        subject_id: Subject identifier
        session_id: Session identifier
        recording_date: Recording timestamp
        metadata: Additional metadata
    """
    signals: np.ndarray  # Shape: (n_channels, n_samples)
    sampling_rate: float
    channel_names: List[str] = field(default_factory=list)
    events: List[EventMarker] = field(default_factory=list)
    subject_id: str = ''
    session_id: str = ''
    recording_date: Optional[datetime] = None
    source_file: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.signals.ndim != 2:
            raise ValueError(
                f"signals must be 2D (channels, samples), got {self.signals.ndim}D"
            )
        
        if self.channel_names and len(self.channel_names) != self.n_channels:
            raise ValueError(
                f"Number of channel names ({len(self.channel_names)}) "
                f"doesn't match number of channels ({self.n_channels})"
            )
        
        # Set default channel names if not provided
        if not self.channel_names:
            self.channel_names = [f"Ch{i+1}" for i in range(self.n_channels)]
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.signals.shape[0]
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.signals.shape[1]
    
    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        return self.n_samples / self.sampling_rate
    
    @property
    def n_events(self) -> int:
        """Number of events."""
        return len(self.events)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Signal shape (n_channels, n_samples)."""
        return self.signals.shape
    
    # =========================================================================
    # DATA ACCESS METHODS
    # =========================================================================
    
    def get_channel(self, channel: Union[int, str]) -> np.ndarray:
        """
        Get data for a specific channel.
        
        Args:
            channel: Channel index or name
        
        Returns:
            np.ndarray: Channel data, shape (n_samples,)
        """
        if isinstance(channel, str):
            if channel not in self.channel_names:
                raise ValueError(f"Channel '{channel}' not found in {self.channel_names}")
            idx = self.channel_names.index(channel)
        else:
            idx = channel
        
        return self.signals[idx]
    
    def get_channels(self, channels: List[Union[int, str]]) -> np.ndarray:
        """
        Get data for multiple channels.
        
        Args:
            channels: List of channel indices or names
        
        Returns:
            np.ndarray: Shape (n_selected_channels, n_samples)
        """
        indices = []
        for ch in channels:
            if isinstance(ch, str):
                indices.append(self.channel_names.index(ch))
            else:
                indices.append(ch)
        
        return self.signals[indices]
    
    def get_time_segment(self, 
                         start_sec: float, 
                         end_sec: float) -> np.ndarray:
        """
        Get data for a time segment.
        
        Args:
            start_sec: Start time in seconds
            end_sec: End time in seconds
        
        Returns:
            np.ndarray: Shape (n_channels, segment_samples)
        """
        start_sample = int(start_sec * self.sampling_rate)
        end_sample = int(end_sec * self.sampling_rate)
        
        return self.signals[:, start_sample:end_sample]
    
    def get_time_axis(self) -> np.ndarray:
        """Get time axis in seconds."""
        return np.arange(self.n_samples) / self.sampling_rate
    
    # =========================================================================
    # TRIAL EXTRACTION
    # =========================================================================
    
    def extract_trials(self,
                       trial_length_sec: float = 4.0,
                       pre_stimulus_sec: float = 0.0,
                       class_mapping: Optional[Dict[int, str]] = None
                       ) -> List[TrialData]:
        """
        Extract trials based on event markers.
        
        Args:
            trial_length_sec: Length of each trial in seconds
            pre_stimulus_sec: Time before event to include
            class_mapping: Mapping from event codes to class names
        
        Returns:
            List[TrialData]: Extracted trials
        """
        trials = []
        trial_samples = int(trial_length_sec * self.sampling_rate)
        pre_samples = int(pre_stimulus_sec * self.sampling_rate)
        
        # Default class mapping for BCI Competition IV-2a
        if class_mapping is None:
            class_mapping = {
                1: 'left_hand',
                2: 'right_hand',
                3: 'feet',
                4: 'tongue'
            }
        
        for event_idx, event in enumerate(self.events):
            # Only extract trials for mapped classes
            if event.code not in class_mapping:
                continue
            
            start = event.sample - pre_samples
            end = start + trial_samples
            
            # Skip if outside data bounds
            if start < 0 or end > self.n_samples:
                continue
            
            trial_signals = self.signals[:, start:end]
            
            trial = TrialData(
                signals=trial_signals.copy(),
                label=event.code - 1,  # Convert to 0-indexed
                label_name=class_mapping.get(event.code, ''),
                trial_id=event_idx,
                subject_id=self.subject_id,
                session_id=self.session_id,
                sampling_rate=self.sampling_rate,
                channel_names=self.channel_names.copy(),
                start_sample=start,
                metadata={
                    'event_code': event.code,
                    'event_sample': event.sample,
                    'pre_stimulus_sec': pre_stimulus_sec,
                    'trial_length_sec': trial_length_sec
                }
            )
            trials.append(trial)
        
        return trials
    
    def get_trials_array(self,
                         trial_length_sec: float = 4.0,
                         pre_stimulus_sec: float = 0.0,
                         class_mapping: Optional[Dict[int, str]] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract trials as numpy arrays.
        
        Args:
            trial_length_sec: Length of each trial
            pre_stimulus_sec: Pre-stimulus time
            class_mapping: Event code to class name mapping
        
        Returns:
            Tuple of:
                - X: Trial data, shape (n_trials, n_channels, n_samples)
                - y: Labels, shape (n_trials,)
        """
        trials = self.extract_trials(
            trial_length_sec=trial_length_sec,
            pre_stimulus_sec=pre_stimulus_sec,
            class_mapping=class_mapping
        )
        
        X = np.array([t.signals for t in trials])
        y = np.array([t.label for t in trials])
        
        return X, y
    
    # =========================================================================
    # EVENT METHODS
    # =========================================================================
    
    def get_events_by_code(self, code: int) -> List[EventMarker]:
        """Get all events with a specific code."""
        return [e for e in self.events if e.code == code]
    
    def get_event_counts(self) -> Dict[int, int]:
        """Get count of each event type."""
        counts: Dict[int, int] = {}
        for event in self.events:
            counts[event.code] = counts.get(event.code, 0) + 1
        return counts
    
    def add_event(self, event: EventMarker) -> None:
        """Add an event marker."""
        self.events.append(event)
        # Keep events sorted by sample
        self.events.sort(key=lambda e: e.sample)
    
    # =========================================================================
    # TRANSFORMATION METHODS
    # =========================================================================
    
    def select_channels(self, channels: List[Union[int, str]]) -> 'EEGData':
        """
        Create new EEGData with selected channels.
        
        Args:
            channels: List of channel indices or names
        
        Returns:
            New EEGData with selected channels
        """
        indices = []
        new_names = []
        
        for ch in channels:
            if isinstance(ch, str):
                idx = self.channel_names.index(ch)
                indices.append(idx)
                new_names.append(ch)
            else:
                indices.append(ch)
                new_names.append(self.channel_names[ch])
        
        return EEGData(
            signals=self.signals[indices].copy(),
            sampling_rate=self.sampling_rate,
            channel_names=new_names,
            events=self.events.copy(),
            subject_id=self.subject_id,
            session_id=self.session_id,
            recording_date=self.recording_date,
            source_file=self.source_file,
            metadata={**self.metadata, 'selected_channels': channels}
        )
    
    def crop(self, 
             start_sec: float, 
             end_sec: float) -> 'EEGData':
        """
        Crop data to a time range.
        
        Args:
            start_sec: Start time in seconds
            end_sec: End time in seconds
        
        Returns:
            New EEGData with cropped data
        """
        start_sample = int(start_sec * self.sampling_rate)
        end_sample = int(end_sec * self.sampling_rate)
        
        # Adjust events
        new_events = []
        for event in self.events:
            if start_sample <= event.sample < end_sample:
                new_event = EventMarker(
                    sample=event.sample - start_sample,
                    code=event.code,
                    label=event.label,
                    duration_samples=event.duration_samples,
                    metadata=event.metadata.copy()
                )
                new_events.append(new_event)
        
        return EEGData(
            signals=self.signals[:, start_sample:end_sample].copy(),
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names.copy(),
            events=new_events,
            subject_id=self.subject_id,
            session_id=self.session_id,
            recording_date=self.recording_date,
            source_file=self.source_file,
            metadata={**self.metadata, 'cropped': [start_sec, end_sec]}
        )
    
    def copy(self) -> 'EEGData':
        """Create a deep copy."""
        return EEGData(
            signals=self.signals.copy(),
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names.copy(),
            events=[EventMarker(e.sample, e.code, e.label, e.duration_samples, 
                               e.metadata.copy()) for e in self.events],
            subject_id=self.subject_id,
            session_id=self.session_id,
            recording_date=self.recording_date,
            source_file=self.source_file,
            metadata=self.metadata.copy()
        )
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signals': self.signals.tolist(),
            'sampling_rate': self.sampling_rate,
            'channel_names': self.channel_names,
            'events': [
                {
                    'sample': e.sample,
                    'code': e.code,
                    'label': e.label,
                    'duration_samples': e.duration_samples,
                    'metadata': e.metadata
                }
                for e in self.events
            ],
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'recording_date': self.recording_date.isoformat() if self.recording_date else None,
            'source_file': self.source_file,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EEGData':
        """Create from dictionary."""
        events = [
            EventMarker(
                sample=e['sample'],
                code=e['code'],
                label=e.get('label', ''),
                duration_samples=e.get('duration_samples', 0),
                metadata=e.get('metadata', {})
            )
            for e in data.get('events', [])
        ]
        
        return cls(
            signals=np.array(data['signals']),
            sampling_rate=data['sampling_rate'],
            channel_names=data.get('channel_names', []),
            events=events,
            subject_id=data.get('subject_id', ''),
            session_id=data.get('session_id', ''),
            recording_date=datetime.fromisoformat(data['recording_date']) 
                          if data.get('recording_date') else None,
            source_file=data.get('source_file', ''),
            metadata=data.get('metadata', {})
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get summary information."""
        return {
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'sampling_rate': self.sampling_rate,
            'duration_seconds': self.duration_seconds,
            'n_events': self.n_events,
            'event_counts': self.get_event_counts(),
            'channel_names': self.channel_names,
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'source_file': self.source_file
        }
    
    def __repr__(self) -> str:
        return (
            f"EEGData("
            f"shape={self.shape}, "
            f"sr={self.sampling_rate}Hz, "
            f"duration={self.duration_seconds:.1f}s, "
            f"events={self.n_events})"
        )
