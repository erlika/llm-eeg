"""
IDataLoader Interface
=====================

This module defines the abstract interface for all data loaders in the EEG-BCI framework.

Data loaders are responsible for:
- Loading EEG data from various file formats (.mat, .gdf, .edf, etc.)
- Parsing metadata (sampling rate, channel names, events, etc.)
- Converting raw data into standardized EEGData objects
- Validating data integrity

Design Principles:
- All data loaders MUST implement this interface
- New file formats can be added by creating new loaders that implement IDataLoader
- Loaders are registered in the component registry for dynamic instantiation

Example Usage:
    ```python
    from src.data.factory import DataLoaderFactory
    
    # Create loader via factory (recommended)
    loader = DataLoaderFactory.create("mat", config)
    
    # Load data
    eeg_data = loader.load("path/to/file.mat")
    
    # Access metadata
    print(f"Sampling rate: {eeg_data.sampling_rate} Hz")
    print(f"Channels: {eeg_data.channel_names}")
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Note: EEGData type will be defined in src/core/types/eeg_data.py
# Using forward reference here to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.types.eeg_data import EEGData


class IDataLoader(ABC):
    """
    Abstract interface for EEG data loaders.
    
    All data loader implementations must inherit from this class and implement
    all abstract methods. This ensures consistent behavior across different
    file format loaders.
    
    Attributes:
        name (str): Unique identifier for the loader (e.g., "mat", "gdf", "edf")
        supported_extensions (List[str]): File extensions this loader can handle
    
    Methods:
        initialize: Configure the loader with settings
        load: Load EEG data from a file
        load_multiple: Load data from multiple files
        validate_file: Check if a file can be loaded by this loader
        get_file_info: Get metadata without loading full data
        get_supported_extensions: Return list of supported file extensions
    """
    
    # =========================================================================
    # ABSTRACT PROPERTIES - Must be implemented by all subclasses
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this data loader.
        
        Returns:
            str: Loader name (e.g., "mat", "gdf", "edf", "moabb")
        
        Example:
            >>> loader.name
            'mat'
        """
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        List of file extensions this loader can handle.
        
        Returns:
            List[str]: Supported extensions including the dot (e.g., [".mat", ".MAT"])
        
        Example:
            >>> loader.supported_extensions
            ['.mat', '.MAT']
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by all subclasses
    # =========================================================================
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the data loader with configuration settings.
        
        This method is called after instantiation to configure the loader.
        Configuration may include:
        - Default channel selection
        - Event code mappings
        - Preprocessing flags
        - Memory management settings
        
        Args:
            config: Dictionary containing loader configuration
                Expected keys (all optional):
                - 'channels': List of channel names/indices to load
                - 'event_mapping': Dict mapping event codes to labels
                - 'preload': Whether to load data into memory immediately
                - 'verbose': Logging verbosity level
        
        Raises:
            ValueError: If configuration is invalid
            
        Example:
            >>> loader.initialize({
            ...     'channels': ['C3', 'C4', 'Cz'],
            ...     'event_mapping': {1: 'left_hand', 2: 'right_hand'},
            ...     'preload': True
            ... })
        """
        pass
    
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> 'EEGData':
        """
        Load EEG data from a single file.
        
        This is the primary method for loading data. It reads the file,
        parses all relevant information, and returns a standardized EEGData object.
        
        Args:
            file_path: Path to the EEG data file (absolute or relative)
        
        Returns:
            EEGData: Standardized EEG data object containing:
                - Raw signal data (numpy array)
                - Sampling rate
                - Channel names
                - Events/markers
                - Metadata
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid or unsupported
            IOError: If file cannot be read
            
        Example:
            >>> eeg_data = loader.load("data/raw/A01T.mat")
            >>> print(eeg_data.signals.shape)
            (22, 250000)  # (channels, samples)
        """
        pass
    
    @abstractmethod
    def load_multiple(self, 
                      file_paths: List[Union[str, Path]],
                      concatenate: bool = False) -> Union[List['EEGData'], 'EEGData']:
        """
        Load EEG data from multiple files.
        
        Useful for loading multiple sessions or subjects at once.
        
        Args:
            file_paths: List of paths to EEG data files
            concatenate: If True, concatenate all data into single EEGData object
                        If False, return list of separate EEGData objects
        
        Returns:
            Union[List[EEGData], EEGData]: 
                - List of EEGData objects (if concatenate=False)
                - Single concatenated EEGData (if concatenate=True)
        
        Raises:
            FileNotFoundError: If any file does not exist
            ValueError: If files have incompatible properties (when concatenating)
            
        Example:
            >>> files = ["A01T.mat", "A01E.mat"]
            >>> data_list = loader.load_multiple(files, concatenate=False)
            >>> len(data_list)
            2
        """
        pass
    
    @abstractmethod
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file can be loaded by this loader.
        
        Performs validation without loading the full data:
        - Checks file existence
        - Validates file extension
        - Optionally checks file header/magic bytes
        
        Args:
            file_path: Path to the file to validate
        
        Returns:
            bool: True if file can be loaded, False otherwise
            
        Example:
            >>> loader.validate_file("data/A01T.mat")
            True
            >>> loader.validate_file("data/A01T.csv")
            False
        """
        pass
    
    @abstractmethod
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file metadata without loading the full data.
        
        Useful for previewing file contents or validating compatibility
        without the memory overhead of loading all signal data.
        
        Args:
            file_path: Path to the EEG data file
        
        Returns:
            Dict containing file metadata:
                - 'sampling_rate': Sampling frequency in Hz
                - 'n_channels': Number of channels
                - 'channel_names': List of channel names
                - 'duration_seconds': Total recording duration
                - 'n_samples': Total number of samples
                - 'n_events': Number of events/markers
                - 'file_format': Format identifier string
                - 'file_size_mb': File size in megabytes
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format cannot be parsed
            
        Example:
            >>> info = loader.get_file_info("data/A01T.mat")
            >>> print(f"Duration: {info['duration_seconds']:.1f}s")
            Duration: 1000.0s
        """
        pass
    
    # =========================================================================
    # CONCRETE METHODS - Default implementations (can be overridden)
    # =========================================================================
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this loader.
        
        This is a convenience method that returns the supported_extensions property.
        Subclasses typically don't need to override this.
        
        Returns:
            List[str]: Supported file extensions
        """
        return self.supported_extensions
    
    def can_load(self, file_path: Union[str, Path]) -> bool:
        """
        Quick check if this loader can handle a given file.
        
        Checks file extension against supported extensions.
        For more thorough validation, use validate_file().
        
        Args:
            file_path: Path to check
        
        Returns:
            bool: True if extension is supported
        """
        path = Path(file_path)
        return path.suffix.lower() in [ext.lower() for ext in self.supported_extensions]
    
    def __repr__(self) -> str:
        """String representation of the loader."""
        return f"{self.__class__.__name__}(name='{self.name}', extensions={self.supported_extensions})"
