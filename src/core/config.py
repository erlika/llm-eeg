"""
Configuration Manager
=====================

This module implements the central configuration management for the EEG-BCI framework.

The configuration manager is responsible for:
- Loading configuration from YAML/JSON files
- Managing hierarchical configurations with inheritance
- Providing type-safe access to configuration values
- Supporting environment-specific overrides
- Validating configuration against schemas

Configuration Hierarchy:
-----------------------
1. Default config (hardcoded fallbacks)
2. Base config file (default.yaml)
3. Environment config (development.yaml, production.yaml)
4. User config (user_config.yaml)
5. Runtime overrides (programmatic)

Each level overrides values from previous levels.

Configuration Structure:
-----------------------
configs/
├── default.yaml        # Base configuration
├── development.yaml    # Development overrides
├── production.yaml     # Production settings
├── data_loaders.yaml   # Data loader configurations
├── preprocessors.yaml  # Preprocessing settings
├── classifiers.yaml    # Classifier configurations
├── agents.yaml         # Agent configurations (APA, DVA)
└── llm_providers.yaml  # LLM provider settings

Example Usage:
    ```python
    from src.core.config import ConfigManager, get_config
    
    # Get singleton instance
    config = ConfigManager.get_instance()
    
    # Load configuration
    config.load('configs/default.yaml')
    config.load('configs/agents.yaml')
    
    # Access values with dot notation
    confidence_threshold = config.get('agents.dva.confidence_threshold')
    # Returns: 0.8 (user-approved value)
    
    # Access with type safety
    learning_rate = config.get_float('agents.apa.policy.learning_rate', default=0.1)
    
    # Set runtime override
    config.set('debug', True)
    
    # Get nested config as dict
    apa_config = config.get_section('agents.apa')
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Union, Type, TypeVar
from pathlib import Path
import yaml
import json
import os
import logging
import threading
from copy import deepcopy
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic type hints
T = TypeVar('T')


@dataclass
class ConfigValue:
    """
    Wrapper for configuration values with metadata.
    
    Attributes:
        value: The configuration value
        source: Where the value came from (file path or 'default')
        type_hint: Expected type for validation
        description: Human-readable description
    """
    value: Any
    source: str = 'default'
    type_hint: Optional[Type] = None
    description: str = ''


class ConfigManager:
    """
    Central configuration manager for the framework.
    
    Implements the Singleton pattern with thread-safe access.
    
    Attributes:
        _instance: Singleton instance
        _lock: Thread lock
        _config: Flattened configuration dictionary
        _sources: Track which file each value came from
        _defaults: Default values
    """
    
    _instance: Optional['ConfigManager'] = None
    _lock: threading.Lock = threading.Lock()
    
    # =========================================================================
    # SINGLETON PATTERN
    # =========================================================================
    
    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager."""
        if self._initialized:
            return
        
        # Main configuration storage (hierarchical)
        self._config: Dict[str, Any] = {}
        
        # Track value sources
        self._sources: Dict[str, str] = {}
        
        # Loaded file paths
        self._loaded_files: List[str] = []
        
        # Default values (set programmatically)
        self._defaults: Dict[str, Any] = {}
        
        # Environment name
        self._environment: str = os.getenv('BCI_ENV', 'development')
        
        # Initialize with hardcoded defaults
        self._init_defaults()
        
        self._initialized = True
        logger.info(f"ConfigManager initialized (env: {self._environment})")
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """
        Get the singleton configuration manager.
        
        Returns:
            ConfigManager: The singleton instance
        """
        return cls()
    
    @classmethod
    def reset(cls) -> None:
        """Reset the configuration manager (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._config.clear()
                cls._instance._sources.clear()
                cls._instance._loaded_files.clear()
                cls._instance._initialized = False
            cls._instance = None
        logger.info("ConfigManager reset")
    
    # =========================================================================
    # DEFAULT CONFIGURATION
    # =========================================================================
    
    def _init_defaults(self) -> None:
        """Initialize hardcoded default values."""
        self._defaults = {
            # Project settings
            'project': {
                'name': 'EEG-BCI Framework',
                'version': '1.0.0',
                'description': 'BCI Framework with LLM and AI Agents'
            },
            
            # Data settings
            'data': {
                'sampling_rate': 250,          # BCI Competition IV-2a
                'n_channels': 22,              # 22 EEG channels
                'n_classes': 4,                # 4 MI classes
                'trial_duration_sec': 4.0,     # 4 second trials
                'event_codes': {
                    'left_hand': 1,
                    'right_hand': 2,
                    'feet': 3,
                    'tongue': 4
                },
                'channel_names': [
                    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                    'P1', 'Pz', 'P2', 'POz'
                ]
            },
            
            # Preprocessing defaults
            'preprocessing': {
                'bandpass': {
                    'low_freq': 8.0,
                    'high_freq': 30.0,
                    'filter_order': 5
                },
                'notch': {
                    'freq': 50.0,              # 50 Hz for EU, 60 Hz for US
                    'quality_factor': 30
                },
                'artifact_threshold_uv': 100,
                'normalization': 'zscore'      # 'zscore', 'minmax', None
            },
            
            # Agent settings (USER-APPROVED VALUES)
            'agents': {
                # Adaptive Preprocessing Agent
                'apa': {
                    'enabled': True,
                    'policy': {
                        'type': 'q_learning',  # User-approved: RL-based
                        'learning_rate': 0.1,
                        'discount_factor': 0.99,
                        'epsilon_start': 1.0,
                        'epsilon_decay': 0.995,
                        'epsilon_min': 0.01,
                        'initial_q_value': 0.0
                    },
                    'state_bins': {
                        'snr': [0, 5, 10, 20, float('inf')],
                        'artifact_ratio': [0, 0.1, 0.3, 0.5, 1.0],
                        'line_noise': [0, 0.5, 1.0, 2.0, float('inf')]
                    },
                    'action_space': ['conservative', 'moderate', 'aggressive'],
                    'cross_trial_learning': True  # User-approved
                },
                
                # Decision Validation Agent
                'dva': {
                    'enabled': True,
                    'confidence_threshold': 0.8,  # User-approved: 0.8
                    'adaptive_threshold': True,
                    'validators': [
                        'confidence',
                        'margin',
                        'signal_quality',
                        'historical_consistency'
                    ],
                    'cross_trial_learning': True  # User-approved
                }
            },
            
            # LLM settings
            'llm': {
                'provider': 'phi3',           # User-approved: Phi-3
                'model_path': 'microsoft/phi-3-mini-4k-instruct',
                'device': 'cuda',
                'dtype': 'float16',
                'quantization': '4bit',       # For Colab compatibility
                'max_tokens': 256,
                'temperature': 0.7,
                'enabled': True
            },
            
            # Classifier settings
            'classifiers': {
                'default': 'eegnet',
                'eegnet': {
                    'n_temporal_filters': 8,
                    'temporal_filter_length': 64,
                    'n_spatial_filters': 2,
                    'dropout_rate': 0.5,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            },
            
            # Training settings
            'training': {
                'validation_split': 0.2,
                'early_stopping': True,
                'patience': 10,
                'random_seed': 42,
                'cross_validation': {
                    'enabled': True,
                    'n_folds': 5,
                    'shuffle': True
                }
            },
            
            # Storage settings
            'storage': {
                'backend': 'local',
                'base_path': './data',
                'checkpoints_dir': 'checkpoints',
                'models_dir': 'models',
                'results_dir': 'results'
            },
            
            # Logging settings
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/bci_framework.log'
            },
            
            # Performance targets (from project plan)
            'targets': {
                'subject_dependent_accuracy': 0.85,  # >85%
                'subject_independent_accuracy': 0.70,  # >70%
                'min_kappa': 0.80,
                'min_itr_bits_per_min': 100
            }
        }
        
        # Load defaults into config
        self._config = deepcopy(self._defaults)
        logger.debug("Default configuration initialized")
    
    # =========================================================================
    # LOADING CONFIGURATION
    # =========================================================================
    
    def load(self, 
             path: Union[str, Path],
             merge: bool = True) -> 'ConfigManager':
        """
        Load configuration from a file.
        
        Args:
            path: Path to configuration file (YAML or JSON)
            merge: If True, merge with existing config. If False, replace.
        
        Returns:
            Self for method chaining
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
            
        Example:
            >>> config.load('configs/default.yaml')
            >>> config.load('configs/agents.yaml')  # Merges with existing
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Load based on extension
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        # Merge or replace
        if merge:
            self._merge_config(data, str(path))
        else:
            self._config = data
            self._sources = {self._flatten_key(k, data): str(path) for k in data}
        
        self._loaded_files.append(str(path))
        logger.info(f"Loaded configuration from {path}")
        
        return self
    
    def load_directory(self, 
                       directory: Union[str, Path],
                       pattern: str = '*.yaml') -> 'ConfigManager':
        """
        Load all configuration files from a directory.
        
        Args:
            directory: Directory path
            pattern: Glob pattern for files to load
        
        Returns:
            Self for method chaining
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Config directory not found: {directory}")
            return self
        
        for config_file in sorted(directory.glob(pattern)):
            self.load(config_file)
        
        return self
    
    def _merge_config(self, 
                      new_config: Dict[str, Any], 
                      source: str) -> None:
        """
        Deep merge new configuration into existing.
        
        Args:
            new_config: New configuration to merge
            source: Source identifier (file path)
        """
        def deep_merge(base: Dict, update: Dict, prefix: str = '') -> Dict:
            for key, value in update.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value, full_key)
                else:
                    base[key] = value
                    self._sources[full_key] = source
            
            return base
        
        deep_merge(self._config, new_config)
    
    # =========================================================================
    # ACCESSING CONFIGURATION
    # =========================================================================
    
    def get(self, 
            key: str, 
            default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'agents.dva.confidence_threshold')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('agents.dva.confidence_threshold')
            0.8
            >>> config.get('nonexistent', default='fallback')
            'fallback'
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get configuration value as integer."""
        value = self.get(key, default)
        return int(value) if value is not None else default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get configuration value as float."""
        value = self.get(key, default)
        return float(value) if value is not None else default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get configuration value as boolean."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return bool(value) if value is not None else default
    
    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get configuration value as list."""
        value = self.get(key, default)
        if value is None:
            return default or []
        if isinstance(value, list):
            return value
        return [value]
    
    def get_section(self, key: str) -> Dict[str, Any]:
        """
        Get a configuration section as dictionary.
        
        Args:
            key: Section key (e.g., 'agents.apa')
        
        Returns:
            Configuration section as dict (deep copy)
        """
        value = self.get(key, {})
        return deepcopy(value) if isinstance(value, dict) else {}
    
    def get_source(self, key: str) -> str:
        """
        Get the source (file) where a value was defined.
        
        Args:
            key: Configuration key
        
        Returns:
            Source identifier or 'default'
        """
        return self._sources.get(key, 'default')
    
    # =========================================================================
    # MODIFYING CONFIGURATION
    # =========================================================================
    
    def set(self, 
            key: str, 
            value: Any,
            source: str = 'runtime') -> 'ConfigManager':
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (dot notation)
            value: Value to set
            source: Source identifier
        
        Returns:
            Self for method chaining
            
        Example:
            >>> config.set('agents.dva.confidence_threshold', 0.9)
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
        self._sources[key] = source
        
        logger.debug(f"Set {key} = {value}")
        return self
    
    def update(self, 
               values: Dict[str, Any],
               source: str = 'runtime') -> 'ConfigManager':
        """
        Update multiple configuration values.
        
        Args:
            values: Dictionary of key-value pairs
            source: Source identifier
        
        Returns:
            Self for method chaining
        """
        for key, value in values.items():
            self.set(key, value, source)
        return self
    
    def set_defaults(self, defaults: Dict[str, Any]) -> 'ConfigManager':
        """
        Set default values (won't override existing).
        
        Args:
            defaults: Default values dictionary
        
        Returns:
            Self for method chaining
        """
        for key, value in defaults.items():
            if self.get(key) is None:
                self.set(key, value, 'default')
        return self
    
    # =========================================================================
    # SAVING CONFIGURATION
    # =========================================================================
    
    def save(self, 
             path: Union[str, Path],
             sections: Optional[List[str]] = None) -> None:
        """
        Save configuration to a file.
        
        Args:
            path: Output file path
            sections: If specified, only save these sections
        """
        path = Path(path)
        
        # Get data to save
        if sections:
            data = {s: self.get_section(s) for s in sections}
        else:
            data = deepcopy(self._config)
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on extension
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        logger.info(f"Saved configuration to {path}")
    
    def export(self) -> Dict[str, Any]:
        """
        Export full configuration as dictionary.
        
        Returns:
            Deep copy of configuration
        """
        return deepcopy(self._config)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _flatten_key(self, 
                     key: str, 
                     data: Dict[str, Any], 
                     prefix: str = '') -> str:
        """Flatten nested keys for source tracking."""
        return f"{prefix}.{key}" if prefix else key
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Validate configuration against a schema.
        
        Args:
            schema: Validation schema (optional)
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic validation - check required values
        required_keys = [
            'data.sampling_rate',
            'data.n_channels',
            'data.n_classes',
            'agents.dva.confidence_threshold'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                errors.append(f"Missing required configuration: {key}")
        
        # Validate specific constraints
        dva_threshold = self.get('agents.dva.confidence_threshold', 0)
        if not 0 <= dva_threshold <= 1:
            errors.append(f"DVA confidence_threshold must be 0-1, got {dva_threshold}")
        
        return errors
    
    def get_environment(self) -> str:
        """Get current environment name."""
        return self._environment
    
    def set_environment(self, env: str) -> 'ConfigManager':
        """Set environment and load environment-specific config."""
        self._environment = env
        # Load environment config if exists
        env_config = f"configs/{env}.yaml"
        if Path(env_config).exists():
            self.load(env_config)
        return self
    
    def summary(self) -> str:
        """Get configuration summary."""
        lines = [
            "Configuration Summary",
            "=" * 40,
            f"Environment: {self._environment}",
            f"Loaded files: {len(self._loaded_files)}",
            ""
        ]
        
        for file in self._loaded_files:
            lines.append(f"  - {file}")
        
        lines.append("\nKey Settings:")
        lines.append(f"  - DVA Confidence: {self.get('agents.dva.confidence_threshold')}")
        lines.append(f"  - APA Policy: {self.get('agents.apa.policy.type')}")
        lines.append(f"  - Cross-trial Learning: {self.get('agents.apa.cross_trial_learning')}")
        lines.append(f"  - LLM Provider: {self.get('llm.provider')}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigManager(env='{self._environment}', files={len(self._loaded_files)})"
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access: config['key']."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style setting: config['key'] = value."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Allow 'key in config' syntax."""
        return self.get(key) is not None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_config() -> ConfigManager:
    """
    Get the singleton ConfigManager instance.
    
    Returns:
        ConfigManager: The singleton instance
    """
    return ConfigManager.get_instance()


def load_config(path: Union[str, Path]) -> ConfigManager:
    """
    Load configuration from file.
    
    Args:
        path: Configuration file path
    
    Returns:
        ConfigManager instance
    """
    return get_config().load(path)
