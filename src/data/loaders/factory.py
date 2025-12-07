"""
Data Loader Factory
===================

This module implements the Factory pattern for creating data loaders.
It provides a centralized way to instantiate appropriate loaders based on
file format or explicit type specification.

The factory supports:
- Automatic loader selection based on file extension
- Manual loader creation by type name
- Registration of custom loaders
- Integration with the component registry

Design Patterns Used:
- Factory Method: Creates loader instances
- Registry Pattern: Maintains available loader types
- Strategy Pattern: Different loaders for different formats

Usage Examples:
    ```python
    from src.data.loaders import DataLoaderFactory
    
    # Create loader by file extension (automatic detection)
    loader = DataLoaderFactory.create_for_file('data/A01T.mat')
    
    # Create loader by type name
    loader = DataLoaderFactory.create('mat')
    
    # Create with configuration
    loader = DataLoaderFactory.create('mat', config={
        'include_eog': False,
        'channels': ['C3', 'C4', 'Cz']
    })
    
    # Register custom loader
    DataLoaderFactory.register('custom', MyCustomLoader)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, Type, Optional, Any, Union, List
from pathlib import Path
import logging

from src.core.interfaces.i_data_loader import IDataLoader


# Configure module logger
logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory class for creating data loader instances.
    
    This class provides methods for creating data loaders based on
    file type or explicit specification. It maintains a registry of
    available loader types.
    
    Class Attributes:
        _loaders (Dict): Registry of loader name -> loader class mappings
        _extension_map (Dict): Registry of extension -> loader name mappings
    
    Thread Safety:
        The factory is thread-safe for read operations. Registration
        should be done during initialization before concurrent access.
    
    Example:
        >>> factory = DataLoaderFactory()
        >>> loader = factory.create('mat')
        >>> eeg_data = loader.load('A01T.mat')
    """
    
    # Registry of available loaders: name -> class
    _loaders: Dict[str, Type[IDataLoader]] = {}
    
    # Extension to loader name mapping
    _extension_map: Dict[str, str] = {}
    
    # Track initialization state
    _initialized: bool = False
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """
        Ensure the factory is initialized with default loaders.
        
        This method lazily initializes the factory with built-in loaders
        on first use.
        """
        if cls._initialized:
            return
        
        # Import and register built-in loaders
        try:
            from src.data.loaders.mat_loader import MATLoader
            cls.register('mat', MATLoader)
        except ImportError as e:
            logger.warning(f"Could not import MATLoader: {e}")
        
        # Future loaders can be registered here:
        # - GDFLoader for .gdf files
        # - EDFLoader for .edf files
        # - MOABBLoader for MOABB datasets
        
        cls._initialized = True
        logger.debug(f"DataLoaderFactory initialized with {len(cls._loaders)} loaders")
    
    @classmethod
    def register(
        cls,
        name: str,
        loader_class: Type[IDataLoader],
        extensions: Optional[List[str]] = None
    ) -> None:
        """
        Register a new loader type with the factory.
        
        This method adds a new loader class to the factory registry,
        allowing it to be created using the factory methods.
        
        Args:
            name: Unique name for the loader (e.g., 'mat', 'gdf')
            loader_class: The loader class (must implement IDataLoader)
            extensions: Optional list of file extensions (auto-detected if None)
        
        Raises:
            ValueError: If name already registered or class is invalid
            TypeError: If loader_class doesn't implement IDataLoader
        
        Example:
            >>> DataLoaderFactory.register('custom', MyCustomLoader)
            >>> loader = DataLoaderFactory.create('custom')
        """
        # Validate loader class
        if not isinstance(loader_class, type):
            raise TypeError(f"loader_class must be a class, got {type(loader_class)}")
        
        if not issubclass(loader_class, IDataLoader):
            raise TypeError(
                f"{loader_class.__name__} must implement IDataLoader interface"
            )
        
        # Check for duplicate registration
        if name in cls._loaders:
            logger.warning(
                f"Overwriting existing loader registration: '{name}'"
            )
        
        # Register the loader
        cls._loaders[name] = loader_class
        logger.info(f"Registered loader: '{name}' -> {loader_class.__name__}")
        
        # Register extension mappings
        if extensions is None:
            # Try to get extensions from an instance
            try:
                instance = loader_class()
                extensions = instance.supported_extensions
            except Exception:
                extensions = []
        
        for ext in extensions:
            ext_lower = ext.lower()
            if ext_lower in cls._extension_map:
                logger.debug(
                    f"Extension '{ext}' remapped: "
                    f"{cls._extension_map[ext_lower]} -> {name}"
                )
            cls._extension_map[ext_lower] = name
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Remove a loader from the registry.
        
        Args:
            name: Name of the loader to remove
        
        Returns:
            bool: True if loader was removed, False if not found
        
        Example:
            >>> DataLoaderFactory.unregister('custom')
            True
        """
        if name not in cls._loaders:
            return False
        
        # Remove from loaders
        del cls._loaders[name]
        
        # Remove associated extension mappings
        extensions_to_remove = [
            ext for ext, loader in cls._extension_map.items()
            if loader == name
        ]
        for ext in extensions_to_remove:
            del cls._extension_map[ext]
        
        logger.info(f"Unregistered loader: '{name}'")
        return True
    
    @classmethod
    def create(
        cls,
        loader_type: str,
        config: Optional[Dict[str, Any]] = None,
        auto_initialize: bool = True
    ) -> IDataLoader:
        """
        Create a data loader instance by type name.
        
        This method creates a new loader instance of the specified type,
        optionally initializing it with the provided configuration.
        
        Args:
            loader_type: Type of loader to create (e.g., 'mat', 'gdf')
            config: Optional configuration dictionary for initialization
            auto_initialize: Whether to call initialize() automatically
        
        Returns:
            IDataLoader: New loader instance
        
        Raises:
            ValueError: If loader_type is not registered
        
        Example:
            >>> loader = DataLoaderFactory.create('mat', config={
            ...     'include_eog': False,
            ...     'channels': ['C3', 'C4', 'Cz']
            ... })
            >>> eeg_data = loader.load('A01T.mat')
        """
        cls._ensure_initialized()
        
        # Normalize type name
        loader_type = loader_type.lower()
        
        # Check if type is registered
        if loader_type not in cls._loaders:
            available = ', '.join(cls._loaders.keys())
            raise ValueError(
                f"Unknown loader type: '{loader_type}'. "
                f"Available types: {available}"
            )
        
        # Create instance
        loader_class = cls._loaders[loader_type]
        loader = loader_class()
        
        logger.debug(f"Created loader instance: {loader_class.__name__}")
        
        # Initialize if requested
        if auto_initialize:
            loader.initialize(config or {})
        
        return loader
    
    @classmethod
    def create_for_file(
        cls,
        file_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None
    ) -> IDataLoader:
        """
        Create a loader appropriate for the given file.
        
        Automatically detects the file type based on extension and
        creates the appropriate loader.
        
        Args:
            file_path: Path to the file to load
            config: Optional configuration dictionary
        
        Returns:
            IDataLoader: Appropriate loader for the file type
        
        Raises:
            ValueError: If file extension is not supported
        
        Example:
            >>> loader = DataLoaderFactory.create_for_file('data/A01T.mat')
            >>> eeg_data = loader.load('data/A01T.mat')
        """
        cls._ensure_initialized()
        
        # Get file extension
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Look up loader type
        if extension not in cls._extension_map:
            supported = ', '.join(sorted(cls._extension_map.keys()))
            raise ValueError(
                f"Unsupported file extension: '{extension}'. "
                f"Supported extensions: {supported}"
            )
        
        loader_type = cls._extension_map[extension]
        logger.debug(f"Auto-detected loader type '{loader_type}' for '{extension}'")
        
        return cls.create(loader_type, config)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """
        Get list of registered loader types.
        
        Returns:
            List[str]: Names of available loader types
        
        Example:
            >>> types = DataLoaderFactory.get_available_types()
            >>> print(types)
            ['mat', 'gdf', 'edf']
        """
        cls._ensure_initialized()
        return list(cls._loaders.keys())
    
    @classmethod
    def get_supported_extensions(cls) -> Dict[str, str]:
        """
        Get mapping of supported extensions to loader types.
        
        Returns:
            Dict[str, str]: Extension -> loader type mapping
        
        Example:
            >>> extensions = DataLoaderFactory.get_supported_extensions()
            >>> print(extensions)
            {'.mat': 'mat', '.gdf': 'gdf'}
        """
        cls._ensure_initialized()
        return cls._extension_map.copy()
    
    @classmethod
    def get_loader_class(cls, loader_type: str) -> Type[IDataLoader]:
        """
        Get the loader class for a given type name.
        
        Args:
            loader_type: Type name
        
        Returns:
            Type[IDataLoader]: The loader class
        
        Raises:
            ValueError: If type not registered
        """
        cls._ensure_initialized()
        
        loader_type = loader_type.lower()
        if loader_type not in cls._loaders:
            raise ValueError(f"Unknown loader type: '{loader_type}'")
        
        return cls._loaders[loader_type]
    
    @classmethod
    def can_load(cls, file_path: Union[str, Path]) -> bool:
        """
        Check if the factory can create a loader for the given file.
        
        Args:
            file_path: Path to check
        
        Returns:
            bool: True if a loader is available for this file type
        
        Example:
            >>> if DataLoaderFactory.can_load('data.mat'):
            ...     loader = DataLoaderFactory.create_for_file('data.mat')
        """
        cls._ensure_initialized()
        
        extension = Path(file_path).suffix.lower()
        return extension in cls._extension_map
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the factory to uninitialized state.
        
        Primarily used for testing. Clears all registered loaders.
        """
        cls._loaders.clear()
        cls._extension_map.clear()
        cls._initialized = False
        logger.debug("DataLoaderFactory reset")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_loader(
    loader_type: str,
    **config
) -> IDataLoader:
    """
    Convenience function to create a data loader.
    
    Args:
        loader_type: Type of loader ('mat', 'gdf', etc.)
        **config: Configuration options passed to initialize()
    
    Returns:
        IDataLoader: Initialized loader instance
    
    Example:
        >>> loader = create_loader('mat', include_eog=False)
        >>> eeg_data = loader.load('A01T.mat')
    """
    return DataLoaderFactory.create(loader_type, config=config)


def load_eeg_file(
    file_path: Union[str, Path],
    **config
) -> 'EEGData':
    """
    Load an EEG file with automatic loader detection.
    
    This is the simplest way to load EEG data. It automatically
    selects the appropriate loader and returns the data.
    
    Args:
        file_path: Path to the EEG data file
        **config: Configuration options for the loader
    
    Returns:
        EEGData: Loaded EEG data
    
    Example:
        >>> eeg_data = load_eeg_file('data/A01T.mat', include_eog=False)
        >>> print(f"Loaded: {eeg_data.shape}")
    """
    from src.core.types.eeg_data import EEGData
    
    loader = DataLoaderFactory.create_for_file(file_path, config=config)
    return loader.load(file_path)
