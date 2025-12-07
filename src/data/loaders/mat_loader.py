"""
MAT File Data Loader
====================

This module implements the data loader for MAT files, specifically designed
for the BCI Competition IV-2a dataset.

BCI Competition IV-2a Dataset Specifications:
--------------------------------------------
- 9 subjects (A01-A09)
- 2 sessions per subject (Training 'T', Evaluation 'E')
- 4 motor imagery classes:
    - Class 1 (769): Left hand
    - Class 2 (770): Right hand
    - Class 3 (771): Both feet
    - Class 4 (772): Tongue
- 25 channels: 22 EEG + 3 EOG
- Sampling rate: 250 Hz
- 288 trials per session (6 runs × 48 trials)
- Trial timing:
    - t=0s: Fixation cross + beep
    - t=2s: Cue appears (arrow)
    - t=2s-6s: Motor imagery period (4 seconds)

File Structure (MAT format):
---------------------------
The converted MAT files typically contain:
- 'data': EEG signals (samples × channels) or (channels × samples)
- 'labels': Class labels for each trial
- 'fs' or 'srate': Sampling rate
- Event information in various formats

Google Drive Integration:
------------------------
This loader is designed to work with files stored in:
- Google Drive (Google Colab): /content/drive/MyDrive/BCI_Competition_IV_2a/
- Local storage: ./data/raw/

Usage Example:
    ```python
    from src.data.loaders import MATLoader
    
    # Create and initialize loader
    loader = MATLoader()
    loader.initialize({
        'channels': ['C3', 'C4', 'Cz'],  # Optional channel selection
        'include_eog': False              # Exclude EOG channels
    })
    
    # Load single file
    eeg_data = loader.load('data/raw/A01T.mat')
    print(f"Shape: {eeg_data.shape}")
    print(f"Events: {eeg_data.n_events}")
    
    # Extract trials
    trials = eeg_data.extract_trials(trial_length_sec=4.0)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import numpy as np

from src.data.loaders.base_loader import BaseDataLoader
from src.core.types.eeg_data import EventMarker


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - BCI Competition IV-2a Dataset
# =============================================================================

# Standard channel names for BCI Competition IV-2a (22 EEG channels)
BCI_IV_2A_EEG_CHANNELS: List[str] = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

# EOG channel names (3 EOG channels)
BCI_IV_2A_EOG_CHANNELS: List[str] = [
    'EOG-left', 'EOG-central', 'EOG-right'
]

# All channels (22 EEG + 3 EOG = 25 total)
BCI_IV_2A_ALL_CHANNELS: List[str] = BCI_IV_2A_EEG_CHANNELS + BCI_IV_2A_EOG_CHANNELS

# Event codes for BCI Competition IV-2a
BCI_IV_2A_EVENT_CODES: Dict[int, str] = {
    276: 'eyes_open',       # Idling EEG (eyes open)
    277: 'eyes_closed',     # Idling EEG (eyes closed)
    768: 'trial_start',     # Start of a trial
    769: 'left_hand',       # Class 1: Left hand MI
    770: 'right_hand',      # Class 2: Right hand MI
    771: 'feet',            # Class 3: Both feet MI
    772: 'tongue',          # Class 4: Tongue MI
    783: 'cue_unknown',     # Cue unknown (for test data)
    1023: 'artifact',       # Rejected trial (artifact)
    1072: 'eye_movement',   # Eye movements
    32766: 'new_run'        # Start of a new run
}

# Class mapping (event code to 0-indexed class)
BCI_IV_2A_CLASS_MAPPING: Dict[int, int] = {
    769: 0,  # left_hand -> class 0
    770: 1,  # right_hand -> class 1
    771: 2,  # feet -> class 2
    772: 3   # tongue -> class 3
}

# Sampling rate
BCI_IV_2A_SAMPLING_RATE: float = 250.0

# Number of trials per session
BCI_IV_2A_TRIALS_PER_SESSION: int = 288


class MATLoader(BaseDataLoader):
    """
    Data loader for MAT files from BCI Competition IV-2a dataset.
    
    This loader handles the MAT file format commonly used for distributing
    the BCI Competition IV-2a motor imagery dataset. It supports various
    MAT file structures and automatically detects the data layout.
    
    Supported MAT File Structures:
    1. Standard format: {'data': signals, 'labels': labels, 'fs': sampling_rate}
    2. MNE-exported: {'data': signals, 'events': events_array}
    3. MOABB format: Various structures depending on export settings
    
    Attributes:
        _include_eog (bool): Whether to include EOG channels
        _trial_codes (List[int]): Event codes for motor imagery classes
        _mat_library (str): Library to use for loading ('scipy', 'mat73', 'pymatreader')
    
    Example:
        >>> loader = MATLoader()
        >>> loader.initialize({'include_eog': False})
        >>> eeg_data = loader.load('/content/drive/MyDrive/BCI_IV_2a/A01T.mat')
        >>> X, y = eeg_data.get_trials_array()
        >>> print(f"Trials: {X.shape}, Labels: {y.shape}")
    """
    
    def __init__(self):
        """Initialize the MAT loader."""
        super().__init__()
        
        # MAT-specific options
        self._include_eog: bool = False
        self._trial_codes: List[int] = [769, 770, 771, 772]  # MI classes
        self._mat_library: str = 'auto'  # auto-detect best library
        
        # Cached channel list based on options
        self._target_channels: List[str] = BCI_IV_2A_EEG_CHANNELS.copy()
    
    # =========================================================================
    # ABSTRACT PROPERTY IMPLEMENTATIONS
    # =========================================================================
    
    @property
    def name(self) -> str:
        """
        Unique identifier for this data loader.
        
        Returns:
            str: 'mat' - identifies this as a MAT file loader
        """
        return "mat"
    
    @property
    def supported_extensions(self) -> List[str]:
        """
        List of supported file extensions.
        
        Returns:
            List[str]: ['.mat', '.MAT'] - both lowercase and uppercase
        """
        return ['.mat', '.MAT']
    
    # =========================================================================
    # TEMPLATE METHOD IMPLEMENTATIONS
    # =========================================================================
    
    def _initialize_specific(self, config: Dict[str, Any]) -> None:
        """
        Perform MAT-specific initialization.
        
        Handles MAT-specific configuration options:
        - include_eog: Whether to load EOG channels
        - mat_library: Which library to use for loading
        - trial_codes: Custom event codes for trials
        
        Args:
            config: Configuration dictionary
        """
        # EOG channel inclusion
        self._include_eog = config.get('include_eog', False)
        
        # Update target channels based on EOG setting
        if self._include_eog:
            self._target_channels = BCI_IV_2A_ALL_CHANNELS.copy()
            logger.info("Including EOG channels (25 total)")
        else:
            self._target_channels = BCI_IV_2A_EEG_CHANNELS.copy()
            logger.info("Excluding EOG channels (22 EEG only)")
        
        # MAT library preference
        self._mat_library = config.get('mat_library', 'auto')
        
        # Custom trial codes
        if 'trial_codes' in config:
            self._trial_codes = config['trial_codes']
            logger.debug(f"Using custom trial codes: {self._trial_codes}")
    
    def _parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a MAT file and return its contents.
        
        Attempts to load the MAT file using multiple libraries in order:
        1. scipy.io.loadmat (for MATLAB v5 files)
        2. mat73 (for MATLAB v7.3 / HDF5 files)
        3. pymatreader (alternative reader)
        
        Args:
            file_path: Path to the MAT file
        
        Returns:
            Dict containing the parsed MAT file data
        
        Raises:
            ValueError: If file cannot be parsed by any library
            ImportError: If required libraries are not installed
        """
        logger.debug(f"Parsing MAT file: {file_path}")
        
        mat_data = None
        error_messages = []
        
        # Try scipy first (most common for v5 MAT files)
        if self._mat_library in ['auto', 'scipy']:
            try:
                import scipy.io as sio
                mat_data = sio.loadmat(
                    str(file_path),
                    squeeze_me=True,
                    struct_as_record=False
                )
                logger.debug("Successfully loaded with scipy.io")
                return self._normalize_mat_structure(mat_data, 'scipy')
            except Exception as e:
                error_messages.append(f"scipy.io: {e}")
                logger.debug(f"scipy.io failed: {e}")
        
        # Try mat73 for HDF5-based MAT files
        if self._mat_library in ['auto', 'mat73']:
            try:
                import mat73
                mat_data = mat73.loadmat(str(file_path))
                logger.debug("Successfully loaded with mat73")
                return self._normalize_mat_structure(mat_data, 'mat73')
            except Exception as e:
                error_messages.append(f"mat73: {e}")
                logger.debug(f"mat73 failed: {e}")
        
        # Try pymatreader as fallback
        if self._mat_library in ['auto', 'pymatreader']:
            try:
                from pymatreader import read_mat
                mat_data = read_mat(str(file_path))
                logger.debug("Successfully loaded with pymatreader")
                return self._normalize_mat_structure(mat_data, 'pymatreader')
            except Exception as e:
                error_messages.append(f"pymatreader: {e}")
                logger.debug(f"pymatreader failed: {e}")
        
        # All methods failed
        raise ValueError(
            f"Failed to parse MAT file '{file_path}'. "
            f"Errors: {'; '.join(error_messages)}"
        )
    
    def _normalize_mat_structure(
        self,
        mat_data: Dict[str, Any],
        source_library: str
    ) -> Dict[str, Any]:
        """
        Normalize MAT file structure to a common format.
        
        Different MAT file sources may have different structures.
        This method normalizes them to a consistent format.
        
        Expected output structure:
        {
            'signals': np.ndarray (channels, samples),
            'sampling_rate': float,
            'channel_names': List[str],
            'events': List[Dict] with 'position', 'type', 'duration',
            'labels': np.ndarray (optional, for trial labels),
            'metadata': Dict with additional info
        }
        
        Args:
            mat_data: Raw data from MAT file
            source_library: Which library loaded the data
        
        Returns:
            Dict: Normalized data structure
        """
        logger.debug(f"Normalizing MAT structure from {source_library}")
        logger.debug(f"Available keys: {list(mat_data.keys())}")
        
        normalized = {
            'signals': None,
            'sampling_rate': BCI_IV_2A_SAMPLING_RATE,
            'channel_names': [],
            'events': [],
            'labels': None,
            'metadata': {'source_library': source_library}
        }
        
        # Remove MATLAB metadata keys
        data_keys = [k for k in mat_data.keys() 
                     if not k.startswith('__')]
        
        # =====================================================================
        # EXTRACT SIGNALS
        # =====================================================================
        
        # Try various common key names for signal data
        signal_keys = ['data', 'X', 'eeg', 'EEG', 'signals', 's', 'raw']
        
        for key in signal_keys:
            if key in mat_data:
                signals = np.array(mat_data[key])
                
                # Handle different array orientations
                # BCI IV-2a: typically (samples, channels) or (channels, samples)
                if signals.ndim == 2:
                    # Determine orientation: channels should be smaller dimension
                    # For BCI IV-2a: 25 channels, many samples
                    if signals.shape[0] > signals.shape[1]:
                        # (samples, channels) -> transpose to (channels, samples)
                        signals = signals.T
                        logger.debug(f"Transposed signals from {mat_data[key].shape}")
                
                normalized['signals'] = signals.astype(np.float64)
                logger.debug(f"Extracted signals with shape {signals.shape}")
                break
        
        # If signals still not found, try nested structures
        if normalized['signals'] is None:
            normalized['signals'] = self._find_signals_recursive(mat_data)
        
        if normalized['signals'] is None:
            raise ValueError(
                f"Could not find signal data in MAT file. "
                f"Available keys: {data_keys}"
            )
        
        # =====================================================================
        # EXTRACT SAMPLING RATE
        # =====================================================================
        
        sr_keys = ['fs', 'Fs', 'srate', 'sampling_rate', 'SampleRate', 'sample_rate']
        for key in sr_keys:
            if key in mat_data:
                sr = mat_data[key]
                # Handle array vs scalar
                if isinstance(sr, np.ndarray):
                    sr = float(sr.flat[0])
                normalized['sampling_rate'] = float(sr)
                logger.debug(f"Found sampling rate: {normalized['sampling_rate']} Hz")
                break
        
        # =====================================================================
        # EXTRACT CHANNEL NAMES
        # =====================================================================
        
        ch_keys = ['ch_names', 'channel_names', 'channels', 'chanlabels', 'labels']
        for key in ch_keys:
            if key in mat_data and key != 'labels':  # Avoid confusion with class labels
                ch_names = mat_data[key]
                if isinstance(ch_names, np.ndarray):
                    ch_names = [str(c).strip() for c in ch_names.flat]
                elif isinstance(ch_names, list):
                    ch_names = [str(c).strip() for c in ch_names]
                normalized['channel_names'] = ch_names
                logger.debug(f"Found {len(ch_names)} channel names")
                break
        
        # Use default channel names if not found
        if not normalized['channel_names']:
            n_channels = normalized['signals'].shape[0]
            if n_channels == 25:
                normalized['channel_names'] = BCI_IV_2A_ALL_CHANNELS.copy()
            elif n_channels == 22:
                normalized['channel_names'] = BCI_IV_2A_EEG_CHANNELS.copy()
            else:
                normalized['channel_names'] = [f'Ch{i+1}' for i in range(n_channels)]
            logger.debug(f"Using default channel names for {n_channels} channels")
        
        # =====================================================================
        # EXTRACT EVENTS
        # =====================================================================
        
        normalized['events'] = self._extract_events_from_mat(mat_data)
        
        # =====================================================================
        # EXTRACT CLASS LABELS (for direct trial-based formats)
        # =====================================================================
        
        label_keys = ['labels', 'y', 'Y', 'classlabels', 'class_labels']
        for key in label_keys:
            if key in mat_data:
                labels = np.array(mat_data[key]).flatten()
                if len(labels) > 0 and max(labels) <= 4:  # Likely class labels
                    normalized['labels'] = labels
                    logger.debug(f"Found {len(labels)} class labels")
                    break
        
        # =====================================================================
        # EXTRACT ADDITIONAL METADATA
        # =====================================================================
        
        # Look for artifact markers
        artifact_keys = ['artifacts', 'artifact_trial', 'ArtifactSelection']
        for key in artifact_keys:
            if key in mat_data:
                normalized['metadata']['artifacts'] = np.array(mat_data[key])
                break
        
        # Store remaining keys in metadata
        processed_keys = set(signal_keys + sr_keys + ch_keys + label_keys + 
                            artifact_keys + ['events', 'EVENT'])
        for key in data_keys:
            if key not in processed_keys:
                try:
                    normalized['metadata'][key] = mat_data[key]
                except:
                    pass
        
        return normalized
    
    def _find_signals_recursive(
        self,
        data: Any,
        depth: int = 0,
        max_depth: int = 3
    ) -> Optional[np.ndarray]:
        """
        Recursively search for signal data in nested structures.
        
        Some MAT files have deeply nested structures. This method
        searches for the largest 2D array that looks like EEG data.
        
        Args:
            data: Current data to search
            depth: Current recursion depth
            max_depth: Maximum recursion depth
        
        Returns:
            np.ndarray or None: Found signal data or None
        """
        if depth > max_depth:
            return None
        
        # If it's already an ndarray, check if it looks like EEG data
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # EEG data typically has shape (channels, samples) or (samples, channels)
                # Check if dimensions are reasonable for EEG
                min_dim = min(data.shape)
                max_dim = max(data.shape)
                # Typical: 22-25 channels, thousands+ samples
                if 10 <= min_dim <= 100 and max_dim > 1000:
                    return data
        
        # If it's a dict-like structure, search its values
        if isinstance(data, dict):
            for key, value in data.items():
                if not key.startswith('__'):
                    result = self._find_signals_recursive(value, depth + 1)
                    if result is not None:
                        return result
        
        # If it has attributes (like MATLAB struct), search those
        if hasattr(data, '__dict__'):
            for key, value in vars(data).items():
                result = self._find_signals_recursive(value, depth + 1)
                if result is not None:
                    return result
        
        return None
    
    def _extract_events_from_mat(self, mat_data: Dict[str, Any]) -> List[Dict]:
        """
        Extract events from MAT file in various formats.
        
        Handles multiple event storage formats:
        1. EVENT structure with TYP, POS, DUR fields
        2. events array with [position, type, duration] rows
        3. Separate event_type, event_pos arrays
        
        Args:
            mat_data: Parsed MAT file data
        
        Returns:
            List of event dicts with 'position', 'type', 'duration' keys
        """
        events = []
        
        # Try EVENT structure (common in GDF-converted files)
        if 'EVENT' in mat_data:
            event_struct = mat_data['EVENT']
            events = self._parse_event_structure(event_struct)
        
        # Try events array
        elif 'events' in mat_data:
            events_array = np.array(mat_data['events'])
            if events_array.ndim == 2:
                for row in events_array:
                    if len(row) >= 2:
                        events.append({
                            'position': int(row[0]),
                            'type': int(row[1]),
                            'duration': int(row[2]) if len(row) > 2 else 0
                        })
        
        # Try separate arrays
        elif 'event_type' in mat_data and 'event_pos' in mat_data:
            types = np.array(mat_data['event_type']).flatten()
            positions = np.array(mat_data['event_pos']).flatten()
            durations = np.array(mat_data.get('event_dur', np.zeros_like(types))).flatten()
            
            for pos, typ, dur in zip(positions, types, durations):
                events.append({
                    'position': int(pos),
                    'type': int(typ),
                    'duration': int(dur)
                })
        
        # Try trial_start and labels combination
        elif 'trial_start' in mat_data and 'labels' in mat_data:
            starts = np.array(mat_data['trial_start']).flatten()
            labels = np.array(mat_data['labels']).flatten()
            
            # Convert labels to event codes (1-4 -> 769-772)
            for start, label in zip(starts, labels):
                # Add trial start event
                events.append({
                    'position': int(start),
                    'type': 768,  # trial_start
                    'duration': 0
                })
                # Add class cue event (label 1-4 maps to code 769-772)
                if 1 <= label <= 4:
                    events.append({
                        'position': int(start),
                        'type': int(768 + label),  # 769, 770, 771, 772
                        'duration': 0
                    })
        
        logger.debug(f"Extracted {len(events)} events")
        return events
    
    def _parse_event_structure(self, event_struct: Any) -> List[Dict]:
        """
        Parse MATLAB EVENT structure.
        
        The EVENT structure typically has:
        - TYP: Event types (array)
        - POS: Event positions in samples (array)
        - DUR: Event durations in samples (array)
        
        Args:
            event_struct: MATLAB EVENT structure
        
        Returns:
            List of event dictionaries
        """
        events = []
        
        # Handle different structure formats
        if isinstance(event_struct, dict):
            typ = event_struct.get('TYP', event_struct.get('typ', []))
            pos = event_struct.get('POS', event_struct.get('pos', []))
            dur = event_struct.get('DUR', event_struct.get('dur', []))
        elif hasattr(event_struct, 'TYP'):
            typ = getattr(event_struct, 'TYP', [])
            pos = getattr(event_struct, 'POS', [])
            dur = getattr(event_struct, 'DUR', [])
        else:
            return events
        
        # Convert to arrays
        typ = np.array(typ).flatten()
        pos = np.array(pos).flatten()
        dur = np.array(dur).flatten() if len(dur) > 0 else np.zeros_like(typ)
        
        # Create event list
        for i in range(len(typ)):
            events.append({
                'position': int(pos[i]) if i < len(pos) else 0,
                'type': int(typ[i]),
                'duration': int(dur[i]) if i < len(dur) else 0
            })
        
        return events
    
    def _extract_signals(self, parsed_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract signal data from normalized MAT data.
        
        Applies channel selection if EOG exclusion is enabled.
        
        Args:
            parsed_data: Normalized data from _parse_file()
        
        Returns:
            np.ndarray: Signal data (n_channels, n_samples)
        """
        signals = parsed_data['signals']
        channel_names = parsed_data['channel_names']
        
        # Apply EOG filtering if needed
        if not self._include_eog and len(channel_names) == 25:
            # Keep only first 22 channels (EEG)
            signals = signals[:22, :]
            logger.debug("Excluded EOG channels (kept 22 EEG channels)")
        
        return signals
    
    def _extract_sampling_rate(self, parsed_data: Dict[str, Any]) -> float:
        """
        Extract sampling rate from parsed data.
        
        Args:
            parsed_data: Normalized data from _parse_file()
        
        Returns:
            float: Sampling rate in Hz
        """
        return parsed_data['sampling_rate']
    
    def _extract_channel_names(self, parsed_data: Dict[str, Any]) -> List[str]:
        """
        Extract channel names from parsed data.
        
        Applies EOG filtering if enabled.
        
        Args:
            parsed_data: Normalized data from _parse_file()
        
        Returns:
            List[str]: Channel names
        """
        channel_names = parsed_data['channel_names']
        
        # Apply EOG filtering if needed
        if not self._include_eog and len(channel_names) == 25:
            channel_names = channel_names[:22]
        
        return channel_names
    
    def _extract_events(self, parsed_data: Dict[str, Any]) -> List[EventMarker]:
        """
        Extract and convert events to EventMarker objects.
        
        Converts the raw event list to EventMarker objects,
        applying the event code mapping for labels.
        
        Args:
            parsed_data: Normalized data from _parse_file()
        
        Returns:
            List[EventMarker]: Event markers
        """
        events = []
        raw_events = parsed_data.get('events', [])
        
        for event in raw_events:
            position = event['position']
            event_type = event['type']
            duration = event.get('duration', 0)
            
            # Get label from event code mapping
            label = self._event_mapping.get(
                event_type,
                BCI_IV_2A_EVENT_CODES.get(event_type, f'event_{event_type}')
            )
            
            marker = EventMarker(
                sample=position,
                code=event_type,
                label=label,
                duration_samples=duration,
                metadata={'sampling_rate': parsed_data['sampling_rate']}
            )
            events.append(marker)
        
        # Sort events by sample position
        events.sort(key=lambda e: e.sample)
        
        return events
    
    def _extract_metadata(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional metadata.
        
        Args:
            parsed_data: Normalized data from _parse_file()
        
        Returns:
            Dict: Metadata including dataset info
        """
        metadata = parsed_data.get('metadata', {})
        
        # Add BCI Competition IV-2a specific info
        metadata['dataset'] = 'BCI_Competition_IV_2a'
        metadata['n_classes'] = 4
        metadata['class_names'] = ['left_hand', 'right_hand', 'feet', 'tongue']
        metadata['trial_duration_sec'] = 4.0
        metadata['paradigm'] = 'motor_imagery'
        
        # Add trial labels if available
        if parsed_data.get('labels') is not None:
            metadata['trial_labels'] = parsed_data['labels']
        
        return metadata
    
    def _validate_file_format(self, file_path: Path) -> bool:
        """
        Validate MAT file format.
        
        Checks if the file is a valid MAT file by attempting
        to read its header.
        
        Args:
            file_path: Path to validate
        
        Returns:
            bool: True if valid MAT file
        """
        try:
            # Check file header for MAT signature
            with open(file_path, 'rb') as f:
                header = f.read(128)
            
            # MATLAB v5 files start with specific header
            # HDF5 (v7.3) files start with '\x89HDF'
            if header[:4] == b'MATL' or header[:4] == b'\x89HDF':
                return True
            
            # Some MAT files may have different headers, try loading
            import scipy.io as sio
            sio.whosmat(str(file_path))
            return True
            
        except Exception as e:
            logger.debug(f"MAT validation failed: {e}")
            return False
    
    def _get_file_info_specific(self, file_path: Path) -> Dict[str, Any]:
        """
        Get MAT-specific file information.
        
        Extracts metadata without loading all signal data.
        
        Args:
            file_path: Path to the MAT file
        
        Returns:
            Dict: File information
        """
        try:
            # Try to get info without full loading
            import scipy.io as sio
            variables = sio.whosmat(str(file_path))
            
            info = {
                'variables': [(name, shape, dtype) for name, shape, dtype in variables],
                'n_variables': len(variables)
            }
            
            # Find the main data variable
            for name, shape, dtype in variables:
                if name in ['data', 'X', 'eeg', 'signals']:
                    info['data_shape'] = shape
                    info['data_dtype'] = dtype
                    # Estimate channel/sample count
                    if len(shape) == 2:
                        if shape[0] > shape[1]:
                            info['n_samples'] = shape[0]
                            info['n_channels'] = shape[1]
                        else:
                            info['n_samples'] = shape[1]
                            info['n_channels'] = shape[0]
                    break
            
            # Add default values if not found
            info.setdefault('sampling_rate', BCI_IV_2A_SAMPLING_RATE)
            if 'n_samples' in info:
                info['duration_seconds'] = info['n_samples'] / info['sampling_rate']
            
            return info
            
        except Exception as e:
            logger.warning(f"Could not get detailed info: {e}")
            return {
                'sampling_rate': BCI_IV_2A_SAMPLING_RATE,
                'n_channels': 22,
                'error': str(e)
            }
    
    # =========================================================================
    # ADDITIONAL UTILITY METHODS
    # =========================================================================
    
    def get_class_mapping(self) -> Dict[int, str]:
        """
        Get the event code to class name mapping.
        
        Returns:
            Dict[int, str]: Mapping of event codes to class names
        """
        return {
            769: 'left_hand',
            770: 'right_hand',
            771: 'feet',
            772: 'tongue'
        }
    
    def get_default_channels(self, include_eog: bool = False) -> List[str]:
        """
        Get default channel names for BCI Competition IV-2a.
        
        Args:
            include_eog: Whether to include EOG channels
        
        Returns:
            List[str]: Channel names
        """
        if include_eog:
            return BCI_IV_2A_ALL_CHANNELS.copy()
        return BCI_IV_2A_EEG_CHANNELS.copy()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_mat_loader(
    include_eog: bool = False,
    channels: Optional[List[str]] = None,
    **kwargs
) -> MATLoader:
    """
    Create and initialize a MATLoader with common settings.
    
    This is a convenience function for quickly creating a configured loader.
    
    Args:
        include_eog: Whether to include EOG channels
        channels: Optional list of channels to load
        **kwargs: Additional configuration options
    
    Returns:
        MATLoader: Initialized loader
    
    Example:
        >>> loader = create_mat_loader(include_eog=False)
        >>> eeg_data = loader.load('A01T.mat')
    """
    loader = MATLoader()
    config = {
        'include_eog': include_eog,
        **kwargs
    }
    if channels:
        config['channels'] = channels
    
    loader.initialize(config)
    return loader
