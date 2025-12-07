"""
Base Data Loader Implementation
===============================

This module provides the base implementation of the IDataLoader interface.
It provides common functionality that all data loaders can inherit from.

The BaseDataLoader implements:
- Common initialization logic
- File validation utilities
- Logging setup
- Error handling patterns

Design Pattern:
- Template Method Pattern: Defines skeleton of loading algorithm
- Subclasses override specific steps (e.g., _parse_file, _extract_events)

Usage:
    Subclass BaseDataLoader and implement the abstract methods:
    
    ```python
    class MATLoader(BaseDataLoader):
        @property
        def name(self) -> str:
            return "mat"
        
        @property
        def supported_extensions(self) -> List[str]:
            return [".mat", ".MAT"]
        
        def _parse_file(self, file_path: Path) -> Dict[str, Any]:
            # Implementation specific to MAT files
            pass
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import os

from src.core.interfaces.i_data_loader import IDataLoader
from src.core.types.eeg_data import EEGData, EventMarker


# Configure module logger
logger = logging.getLogger(__name__)


class BaseDataLoader(IDataLoader):
    """
    Base implementation of the IDataLoader interface.
    
    This class provides common functionality for all data loaders,
    including file validation, logging, and error handling.
    
    Attributes:
        _config (Dict): Loader configuration
        _initialized (bool): Whether initialize() has been called
        _verbose (bool): Enable verbose logging
        _channel_selection (List[str]): Channels to load (None = all)
        _event_mapping (Dict[int, str]): Event code to label mapping
        _preload (bool): Whether to load data into memory immediately
    
    Template Methods (must be overridden):
        _parse_file: Parse file format and extract raw data
        _extract_signals: Extract signal data from parsed file
        _extract_events: Extract event markers from parsed file
        _extract_metadata: Extract metadata from parsed file
    
    Example:
        >>> loader = MATLoader()
        >>> loader.initialize({'channels': ['C3', 'C4', 'Cz']})
        >>> eeg_data = loader.load('data/A01T.mat')
    """
    
    def __init__(self):
        """Initialize the base data loader."""
        # Configuration storage
        self._config: Dict[str, Any] = {}
        
        # State flags
        self._initialized: bool = False
        self._verbose: bool = False
        
        # Loading options
        self._channel_selection: Optional[List[str]] = None
        self._event_mapping: Dict[int, str] = {}
        self._preload: bool = True
        
        # Default event mapping for BCI Competition IV-2a
        self._default_event_mapping: Dict[int, str] = {
            769: 'left_hand',   # Class 1
            770: 'right_hand',  # Class 2
            771: 'feet',        # Class 3
            772: 'tongue',      # Class 4
            768: 'trial_start',
            1023: 'artifact',
            32766: 'new_run'
        }
        
        logger.debug(f"{self.__class__.__name__} instantiated")
    
    # =========================================================================
    # INTERFACE IMPLEMENTATION
    # =========================================================================
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the data loader with configuration settings.
        
        This method configures the loader with the provided settings.
        It must be called before loading any data.
        
        Args:
            config: Dictionary containing loader configuration
                - 'channels' (List[str], optional): Channel names to load
                - 'event_mapping' (Dict[int, str], optional): Event code mappings
                - 'preload' (bool, optional): Load data into memory (default: True)
                - 'verbose' (bool, optional): Enable verbose logging
        
        Raises:
            ValueError: If configuration is invalid
        
        Example:
            >>> loader.initialize({
            ...     'channels': ['C3', 'C4', 'Cz'],
            ...     'event_mapping': {769: 'left', 770: 'right'},
            ...     'preload': True,
            ...     'verbose': True
            ... })
        """
        logger.info(f"Initializing {self.name} data loader")
        
        # Store full configuration
        self._config = config.copy()
        
        # Extract channel selection
        self._channel_selection = config.get('channels', None)
        if self._channel_selection:
            logger.debug(f"Channel selection: {self._channel_selection}")
        
        # Extract event mapping (merge with defaults)
        custom_mapping = config.get('event_mapping', {})
        self._event_mapping = {**self._default_event_mapping, **custom_mapping}
        
        # Extract other options
        self._preload = config.get('preload', True)
        self._verbose = config.get('verbose', False)
        
        # Set logging level based on verbose flag
        if self._verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
        
        # Call subclass-specific initialization
        self._initialize_specific(config)
        
        self._initialized = True
        logger.info(f"{self.name} loader initialized successfully")
    
    def load(self, file_path: Union[str, Path]) -> EEGData:
        """
        Load EEG data from a single file.
        
        This is the main loading method. It orchestrates the loading process:
        1. Validates the file path and format
        2. Parses the file using format-specific logic
        3. Extracts signals, events, and metadata
        4. Constructs and returns an EEGData object
        
        Args:
            file_path: Path to the EEG data file
        
        Returns:
            EEGData: Loaded and validated EEG data
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or unsupported
            RuntimeError: If loader hasn't been initialized
        
        Example:
            >>> eeg_data = loader.load('data/raw/A01T.mat')
            >>> print(f"Loaded {eeg_data.n_channels} channels")
        """
        # Ensure loader is initialized
        if not self._initialized:
            logger.warning("Loader not initialized, using default configuration")
            self.initialize({})
        
        # Convert to Path object
        file_path = Path(file_path)
        logger.info(f"Loading EEG data from: {file_path}")
        
        # Validate file
        self._validate_file_path(file_path)
        
        # Parse file (format-specific implementation)
        parsed_data = self._parse_file(file_path)
        
        # Extract components
        signals = self._extract_signals(parsed_data)
        sampling_rate = self._extract_sampling_rate(parsed_data)
        channel_names = self._extract_channel_names(parsed_data)
        events = self._extract_events(parsed_data)
        metadata = self._extract_metadata(parsed_data)
        
        # Apply channel selection if specified
        if self._channel_selection:
            signals, channel_names = self._apply_channel_selection(
                signals, channel_names
            )
        
        # Extract subject and session info from filename
        subject_id, session_id = self._parse_filename(file_path)
        
        # Construct EEGData object
        eeg_data = EEGData(
            signals=signals,
            sampling_rate=sampling_rate,
            channel_names=channel_names,
            events=events,
            subject_id=subject_id,
            session_id=session_id,
            source_file=str(file_path),
            metadata=metadata
        )
        
        logger.info(
            f"Loaded: {eeg_data.n_channels} channels, "
            f"{eeg_data.duration_seconds:.1f}s, "
            f"{eeg_data.n_events} events"
        )
        
        return eeg_data
    
    def load_multiple(
        self,
        file_paths: List[Union[str, Path]],
        concatenate: bool = False
    ) -> Union[List[EEGData], EEGData]:
        """
        Load EEG data from multiple files.
        
        Loads multiple files and optionally concatenates them into
        a single EEGData object.
        
        Args:
            file_paths: List of file paths to load
            concatenate: If True, concatenate all data into one EEGData
        
        Returns:
            List[EEGData] if concatenate=False
            EEGData if concatenate=True
        
        Raises:
            FileNotFoundError: If any file doesn't exist
            ValueError: If files have incompatible properties (when concatenating)
        
        Example:
            >>> files = ['A01T.mat', 'A01E.mat']
            >>> data_list = loader.load_multiple(files)
            >>> print(f"Loaded {len(data_list)} files")
        """
        logger.info(f"Loading {len(file_paths)} files (concatenate={concatenate})")
        
        # Load all files
        loaded_data: List[EEGData] = []
        for file_path in file_paths:
            try:
                eeg_data = self.load(file_path)
                loaded_data.append(eeg_data)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                raise
        
        if not concatenate:
            return loaded_data
        
        # Concatenate data
        return self._concatenate_data(loaded_data)
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file can be loaded by this loader.
        
        Performs comprehensive validation:
        1. Checks file existence
        2. Validates file extension
        3. Optionally validates file header/structure
        
        Args:
            file_path: Path to the file to validate
        
        Returns:
            bool: True if file is valid and loadable
        
        Example:
            >>> if loader.validate_file('data.mat'):
            ...     eeg_data = loader.load('data.mat')
        """
        file_path = Path(file_path)
        
        # Check existence
        if not file_path.exists():
            logger.debug(f"File does not exist: {file_path}")
            return False
        
        # Check extension
        if file_path.suffix.lower() not in [
            ext.lower() for ext in self.supported_extensions
        ]:
            logger.debug(f"Unsupported extension: {file_path.suffix}")
            return False
        
        # Check file is readable
        if not os.access(file_path, os.R_OK):
            logger.debug(f"File not readable: {file_path}")
            return False
        
        # Perform format-specific validation
        try:
            return self._validate_file_format(file_path)
        except Exception as e:
            logger.debug(f"Format validation failed: {e}")
            return False
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file metadata without loading the full data.
        
        Extracts metadata and summary information from a file
        without loading all signal data into memory.
        
        Args:
            file_path: Path to the EEG data file
        
        Returns:
            Dict containing:
                - 'sampling_rate': Sampling frequency in Hz
                - 'n_channels': Number of channels
                - 'channel_names': List of channel names
                - 'duration_seconds': Recording duration
                - 'n_samples': Total number of samples
                - 'n_events': Number of event markers
                - 'file_format': Format identifier
                - 'file_size_mb': File size in megabytes
        
        Example:
            >>> info = loader.get_file_info('A01T.mat')
            >>> print(f"Recording: {info['duration_seconds']:.1f} seconds")
        """
        file_path = Path(file_path)
        self._validate_file_path(file_path)
        
        # Get file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Get format-specific info
        info = self._get_file_info_specific(file_path)
        
        # Add common info
        info['file_format'] = self.name
        info['file_size_mb'] = round(file_size_mb, 2)
        info['file_path'] = str(file_path)
        
        return info
    
    # =========================================================================
    # TEMPLATE METHODS - Must be overridden by subclasses
    # =========================================================================
    
    @abstractmethod
    def _parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse the file and return raw data structure.
        
        This method contains format-specific parsing logic.
        Subclasses must implement this method.
        
        Args:
            file_path: Path to the file to parse
        
        Returns:
            Dict containing parsed file data (format-specific)
        
        Raises:
            ValueError: If file cannot be parsed
        """
        pass
    
    @abstractmethod
    def _extract_signals(self, parsed_data: Dict[str, Any]) -> 'np.ndarray':
        """
        Extract signal data from parsed file.
        
        Args:
            parsed_data: Data returned from _parse_file()
        
        Returns:
            np.ndarray: Signal data with shape (n_channels, n_samples)
        """
        pass
    
    @abstractmethod
    def _extract_sampling_rate(self, parsed_data: Dict[str, Any]) -> float:
        """
        Extract sampling rate from parsed file.
        
        Args:
            parsed_data: Data returned from _parse_file()
        
        Returns:
            float: Sampling rate in Hz
        """
        pass
    
    @abstractmethod
    def _extract_channel_names(self, parsed_data: Dict[str, Any]) -> List[str]:
        """
        Extract channel names from parsed file.
        
        Args:
            parsed_data: Data returned from _parse_file()
        
        Returns:
            List[str]: Channel names
        """
        pass
    
    @abstractmethod
    def _extract_events(self, parsed_data: Dict[str, Any]) -> List[EventMarker]:
        """
        Extract event markers from parsed file.
        
        Args:
            parsed_data: Data returned from _parse_file()
        
        Returns:
            List[EventMarker]: Event markers
        """
        pass
    
    @abstractmethod
    def _extract_metadata(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional metadata from parsed file.
        
        Args:
            parsed_data: Data returned from _parse_file()
        
        Returns:
            Dict: Additional metadata
        """
        pass
    
    @abstractmethod
    def _validate_file_format(self, file_path: Path) -> bool:
        """
        Validate file format (format-specific check).
        
        Args:
            file_path: Path to validate
        
        Returns:
            bool: True if format is valid
        """
        pass
    
    @abstractmethod
    def _get_file_info_specific(self, file_path: Path) -> Dict[str, Any]:
        """
        Get format-specific file information.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dict: File information
        """
        pass
    
    # =========================================================================
    # OPTIONAL TEMPLATE METHODS - Can be overridden
    # =========================================================================
    
    def _initialize_specific(self, config: Dict[str, Any]) -> None:
        """
        Perform format-specific initialization.
        
        Override this method to add format-specific initialization logic.
        
        Args:
            config: Configuration dictionary
        """
        pass
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _validate_file_path(self, file_path: Path) -> None:
        """
        Validate that a file path is valid and readable.
        
        Args:
            file_path: Path to validate
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If extension is unsupported
            PermissionError: If file is not readable
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in [
            ext.lower() for ext in self.supported_extensions
        ]:
            raise ValueError(
                f"Unsupported file extension: {file_path.suffix}. "
                f"Supported: {self.supported_extensions}"
            )
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read file: {file_path}")
    
    def _parse_filename(self, file_path: Path) -> tuple:
        """
        Extract subject ID and session ID from filename.
        
        Expected format: A01T.mat (Subject 01, Training session)
                        A01E.mat (Subject 01, Evaluation session)
        
        Args:
            file_path: Path to parse
        
        Returns:
            Tuple of (subject_id, session_id)
        """
        filename = file_path.stem  # e.g., "A01T"
        
        # Default values
        subject_id = ""
        session_id = ""
        
        # Try to parse BCI Competition IV-2a format
        if len(filename) >= 4:
            # Extract subject (e.g., "A01" -> "01")
            subject_id = filename[1:3] if filename[0].upper() == 'A' else filename[:2]
            
            # Extract session (e.g., "T" for Training, "E" for Evaluation)
            session_id = filename[3] if len(filename) > 3 else ""
        
        logger.debug(f"Parsed filename: subject={subject_id}, session={session_id}")
        return subject_id, session_id
    
    def _apply_channel_selection(
        self,
        signals: 'np.ndarray',
        channel_names: List[str]
    ) -> tuple:
        """
        Apply channel selection to signals and names.
        
        Args:
            signals: Signal array (n_channels, n_samples)
            channel_names: List of channel names
        
        Returns:
            Tuple of (selected_signals, selected_names)
        
        Raises:
            ValueError: If requested channel not found
        """
        import numpy as np
        
        if not self._channel_selection:
            return signals, channel_names
        
        indices = []
        selected_names = []
        
        for ch in self._channel_selection:
            if ch in channel_names:
                idx = channel_names.index(ch)
                indices.append(idx)
                selected_names.append(ch)
            else:
                logger.warning(f"Channel '{ch}' not found, skipping")
        
        if not indices:
            raise ValueError(
                f"No valid channels found from selection: {self._channel_selection}"
            )
        
        selected_signals = signals[indices, :]
        logger.debug(f"Selected {len(indices)} channels out of {len(channel_names)}")
        
        return selected_signals, selected_names
    
    def _concatenate_data(self, data_list: List[EEGData]) -> EEGData:
        """
        Concatenate multiple EEGData objects.
        
        Args:
            data_list: List of EEGData objects to concatenate
        
        Returns:
            EEGData: Concatenated data
        
        Raises:
            ValueError: If data has incompatible properties
        """
        import numpy as np
        
        if not data_list:
            raise ValueError("Cannot concatenate empty list")
        
        if len(data_list) == 1:
            return data_list[0]
        
        # Validate compatibility
        base = data_list[0]
        for eeg_data in data_list[1:]:
            if eeg_data.sampling_rate != base.sampling_rate:
                raise ValueError(
                    f"Incompatible sampling rates: {base.sampling_rate} vs "
                    f"{eeg_data.sampling_rate}"
                )
            if eeg_data.n_channels != base.n_channels:
                raise ValueError(
                    f"Incompatible channel counts: {base.n_channels} vs "
                    f"{eeg_data.n_channels}"
                )
        
        # Concatenate signals
        all_signals = [d.signals for d in data_list]
        concatenated_signals = np.concatenate(all_signals, axis=1)
        
        # Adjust event positions and concatenate
        all_events = []
        sample_offset = 0
        
        for eeg_data in data_list:
            for event in eeg_data.events:
                new_event = EventMarker(
                    sample=event.sample + sample_offset,
                    code=event.code,
                    label=event.label,
                    duration_samples=event.duration_samples,
                    metadata={**event.metadata, 'original_file': eeg_data.source_file}
                )
                all_events.append(new_event)
            sample_offset += eeg_data.n_samples
        
        # Create concatenated EEGData
        return EEGData(
            signals=concatenated_signals,
            sampling_rate=base.sampling_rate,
            channel_names=base.channel_names.copy(),
            events=all_events,
            subject_id=base.subject_id,
            session_id='concatenated',
            source_file=f"concatenated_{len(data_list)}_files",
            metadata={
                'concatenated': True,
                'source_files': [d.source_file for d in data_list],
                'n_files': len(data_list)
            }
        )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current loader configuration.
        
        Returns:
            Dict: Current configuration
        """
        return self._config.copy()
    
    def __repr__(self) -> str:
        """String representation of the loader."""
        init_status = "initialized" if self._initialized else "not initialized"
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"extensions={self.supported_extensions}, "
            f"status={init_status})"
        )
