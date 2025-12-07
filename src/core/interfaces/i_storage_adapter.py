"""
IStorageAdapter Interface
=========================

This module defines the abstract interface for storage adapters in the EEG-BCI framework.

Storage adapters are responsible for:
- Persisting data to various backends (local, Google Drive, cloud)
- Loading saved data with version management
- Checkpointing for long-running experiments
- Supporting different data formats and serialization

Storage Backends Supported:
--------------------------
1. LocalStorage:
   - File system storage
   - Fast read/write
   - Default for development

2. GoogleDriveStorage:
   - Google Drive integration (for Colab)
   - User-approved for dataset storage
   - Automatic sync and backup

3. CloudStorage (optional):
   - AWS S3, Google Cloud Storage
   - For production deployment

Data Types to Store:
-------------------
1. EEG Data: Raw and processed signals
2. Features: Extracted feature sets
3. Models: Trained classifiers
4. Agent States: Q-tables, neural network weights
5. Configurations: Experiment settings
6. Results: Metrics, predictions, reports
7. Checkpoints: Pipeline state for resumption

Design Principles:
-----------------
- Abstract storage operations from data processing
- Support atomic operations for data integrity
- Enable versioning for reproducibility
- Provide efficient batch operations

Example Usage:
    ```python
    # Create and initialize storage
    storage = GoogleDriveStorage()
    storage.initialize({
        'base_path': '/content/drive/MyDrive/BCI_Project',
        'create_if_missing': True
    })
    
    # Save data
    storage.save('models/eegnet_s01.pt', model_state)
    storage.save('results/metrics.json', metrics, format='json')
    
    # Load data
    model_state = storage.load('models/eegnet_s01.pt')
    
    # Checkpoint
    storage.create_checkpoint('experiment_v1', {
        'model': model_state,
        'agent': agent_state,
        'config': config
    })
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, BinaryIO
import numpy as np
from pathlib import Path
from datetime import datetime


class IStorageAdapter(ABC):
    """
    Abstract interface for storage adapters.
    
    All storage implementations must inherit from this class.
    This enables transparent switching between storage backends.
    
    Attributes:
        name (str): Storage backend name
        base_path (str): Root path for all storage operations
        is_connected (bool): Whether storage is ready for operations
    
    Supported Data Formats:
        - pickle (.pkl): Python objects (default for complex data)
        - numpy (.npy, .npz): Numpy arrays
        - json (.json): Configurations and metadata
        - yaml (.yaml): Human-readable configurations
        - pytorch (.pt, .pth): PyTorch models
        - hdf5 (.h5): Large datasets
    """
    
    # =========================================================================
    # ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Storage backend name.
        
        Returns:
            str: Backend name (e.g., "local", "google_drive", "s3")
        
        Example:
            >>> storage.name
            'google_drive'
        """
        pass
    
    @property
    @abstractmethod
    def base_path(self) -> str:
        """
        Base path for all storage operations.
        
        Returns:
            str: Root directory/bucket path
        """
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if storage backend is ready.
        
        Returns:
            bool: True if connected and ready
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS - Core Storage Operations
    # =========================================================================
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the storage adapter.
        
        Sets up connection and validates access.
        
        Args:
            config: Storage configuration
                Common keys:
                - 'base_path': Root path for storage
                - 'create_if_missing': Create directories if needed
                - 'mode': 'read', 'write', or 'readwrite'
                
                For Google Drive:
                - 'mount_point': Mount point in Colab
                - 'credentials_path': OAuth credentials
                
                For Cloud Storage:
                - 'bucket': Bucket name
                - 'credentials': Service account credentials
        
        Raises:
            ConnectionError: If cannot connect to storage
            PermissionError: If access is denied
            
        Example:
            >>> storage.initialize({
            ...     'base_path': '/content/drive/MyDrive/BCI_Project',
            ...     'create_if_missing': True
            ... })
        """
        pass
    
    @abstractmethod
    def save(self,
             path: str,
             data: Any,
             format: Optional[str] = None,
             metadata: Optional[Dict[str, Any]] = None,
             overwrite: bool = True) -> str:
        """
        Save data to storage.
        
        Args:
            path: Relative path within base_path
            data: Data to save (type depends on format)
            format: Data format ('pickle', 'numpy', 'json', 'yaml', 'pytorch')
                   Auto-detected from extension if not specified
            metadata: Optional metadata to store alongside data
            overwrite: Whether to overwrite existing files
        
        Returns:
            str: Full path where data was saved
        
        Raises:
            FileExistsError: If file exists and overwrite=False
            IOError: If write fails
            
        Example:
            >>> storage.save('models/eegnet.pt', model.state_dict())
            >>> storage.save('results/accuracy.json', {'accuracy': 0.85})
        """
        pass
    
    @abstractmethod
    def load(self,
             path: str,
             format: Optional[str] = None) -> Any:
        """
        Load data from storage.
        
        Args:
            path: Relative path within base_path
            format: Data format (auto-detected if not specified)
        
        Returns:
            Loaded data
        
        Raises:
            FileNotFoundError: If file does not exist
            IOError: If read fails
            ValueError: If format cannot be determined
            
        Example:
            >>> model_state = storage.load('models/eegnet.pt')
            >>> config = storage.load('configs/default.yaml')
        """
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Check if a path exists in storage.
        
        Args:
            path: Relative path to check
        
        Returns:
            bool: True if path exists
        """
        pass
    
    @abstractmethod
    def delete(self, path: str, recursive: bool = False) -> bool:
        """
        Delete a file or directory.
        
        Args:
            path: Relative path to delete
            recursive: If True, delete directories recursively
        
        Returns:
            bool: True if deletion was successful
        
        Raises:
            FileNotFoundError: If path does not exist
            PermissionError: If deletion is not allowed
        """
        pass
    
    @abstractmethod
    def list(self,
             path: str = "",
             pattern: Optional[str] = None,
             recursive: bool = False) -> List[str]:
        """
        List files/directories in a path.
        
        Args:
            path: Directory path (relative to base_path)
            pattern: Glob pattern to filter results
            recursive: If True, list recursively
        
        Returns:
            List[str]: List of file/directory paths
            
        Example:
            >>> storage.list('models', pattern='*.pt')
            ['models/eegnet_s01.pt', 'models/eegnet_s02.pt']
        """
        pass
    
    @abstractmethod
    def get_info(self, path: str) -> Dict[str, Any]:
        """
        Get file/directory information.
        
        Args:
            path: Path to get info for
        
        Returns:
            Dict containing:
                - 'path': Full path
                - 'size': Size in bytes
                - 'created': Creation timestamp
                - 'modified': Last modification timestamp
                - 'is_directory': Whether path is a directory
                - 'metadata': Stored metadata (if any)
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS - Advanced Operations
    # =========================================================================
    
    @abstractmethod
    def create_checkpoint(self,
                          name: str,
                          data: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a named checkpoint.
        
        Checkpoints are versioned snapshots of experiment state.
        
        Args:
            name: Checkpoint name (will be versioned)
            data: Dictionary of data to checkpoint
            metadata: Optional metadata (timestamp added automatically)
        
        Returns:
            str: Checkpoint path/identifier
        
        Example:
            >>> storage.create_checkpoint('experiment_1', {
            ...     'model_state': model.state_dict(),
            ...     'agent_state': agent.get_state(),
            ...     'epoch': 50,
            ...     'accuracy': 0.82
            ... })
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self,
                        name: str,
                        version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            name: Checkpoint name
            version: Specific version to load (latest if None)
        
        Returns:
            Dict containing checkpointed data and metadata
            
        Example:
            >>> checkpoint = storage.load_checkpoint('experiment_1')
            >>> model.load_state_dict(checkpoint['model_state'])
        """
        pass
    
    @abstractmethod
    def list_checkpoints(self, 
                         name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints.
        
        Args:
            name: Filter by checkpoint name (all if None)
        
        Returns:
            List of checkpoint info dicts with:
                - 'name': Checkpoint name
                - 'version': Version string
                - 'timestamp': Creation time
                - 'metadata': Stored metadata
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS - Override as needed
    # =========================================================================
    
    def copy(self, src: str, dst: str, overwrite: bool = False) -> str:
        """
        Copy a file or directory.
        
        Args:
            src: Source path
            dst: Destination path
            overwrite: Whether to overwrite existing
        
        Returns:
            str: Destination path
        """
        data = self.load(src)
        return self.save(dst, data, overwrite=overwrite)
    
    def move(self, src: str, dst: str, overwrite: bool = False) -> str:
        """
        Move a file or directory.
        
        Args:
            src: Source path
            dst: Destination path
            overwrite: Whether to overwrite existing
        
        Returns:
            str: New path
        """
        result = self.copy(src, dst, overwrite=overwrite)
        self.delete(src)
        return result
    
    def get_full_path(self, path: str) -> str:
        """
        Get full path including base_path.
        
        Args:
            path: Relative path
        
        Returns:
            str: Full absolute path
        """
        return str(Path(self.base_path) / path)
    
    def ensure_directory(self, path: str) -> str:
        """
        Ensure a directory exists.
        
        Args:
            path: Directory path
        
        Returns:
            str: Full path of directory
        """
        pass
    
    def get_available_space(self) -> Optional[int]:
        """
        Get available storage space in bytes.
        
        Returns:
            int: Available bytes, or None if unknown
        """
        return None
    
    def sync(self) -> None:
        """
        Synchronize with remote storage (if applicable).
        
        Forces any pending writes to be committed.
        """
        pass
    
    def close(self) -> None:
        """
        Close the storage connection.
        
        Releases any resources held by the adapter.
        """
        pass
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _detect_format(self, path: str) -> str:
        """
        Detect data format from file extension.
        
        Args:
            path: File path
        
        Returns:
            str: Format identifier
        """
        extension = Path(path).suffix.lower()
        format_map = {
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.npy': 'numpy',
            '.npz': 'numpy_compressed',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.pt': 'pytorch',
            '.pth': 'pytorch',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.csv': 'csv',
            '.mat': 'matlab'
        }
        return format_map.get(extension, 'pickle')
    
    def _add_timestamp_metadata(self, 
                                 metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Add timestamp to metadata."""
        metadata = metadata or {}
        metadata['saved_at'] = datetime.now().isoformat()
        return metadata
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"base_path='{self.base_path}', "
            f"connected={self.is_connected})"
        )
    
    def __enter__(self) -> 'IStorageAdapter':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# =============================================================================
# STORAGE BACKEND TYPES (for reference and factory)
# =============================================================================

class StorageBackendType:
    """Enumeration of supported storage backends."""
    
    LOCAL = "local"           # Local file system
    GOOGLE_DRIVE = "google_drive"  # Google Drive (Colab)
    S3 = "s3"                 # AWS S3
    GCS = "gcs"               # Google Cloud Storage
    AZURE_BLOB = "azure_blob"  # Azure Blob Storage
    MEMORY = "memory"          # In-memory (for testing)


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

LOCAL_STORAGE_CONFIG = {
    'base_path': './data',
    'create_if_missing': True,
    'mode': 'readwrite'
}

GOOGLE_DRIVE_CONFIG = {
    'mount_point': '/content/drive',
    'base_path': '/content/drive/MyDrive/BCI_Project',
    'create_if_missing': True,
    'mode': 'readwrite'
}

COLAB_CHECKPOINT_DIRS = {
    'checkpoints': 'checkpoints',
    'models': 'models',
    'results': 'results',
    'data': 'data',
    'logs': 'logs'
}
