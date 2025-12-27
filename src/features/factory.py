"""
Feature Extractor Factory
=========================

This module provides factory functionality for creating and managing feature extractors.

The factory pattern enables:
- Centralized extractor creation
- Configuration-driven instantiation
- Automatic registration discovery
- Easy switching between extractors

Factory Methods:
---------------
1. `create(name, **kwargs)` - Create extractor by name
2. `create_from_config(config)` - Create from configuration dict
3. `register(name, extractor_class)` - Register custom extractor
4. `list_available()` - List all registered extractors

Usage Example:
    ```python
    from src.features.factory import FeatureExtractorFactory
    
    # Create by name
    csp = FeatureExtractorFactory.create('csp', n_components=6)
    
    # Create from config
    config = {
        'type': 'band_power',
        'params': {
            'bands': {'mu': (8, 12), 'beta': (12, 30)},
            'method': 'welch'
        }
    }
    bp = FeatureExtractorFactory.create_from_config(config)
    
    # List available extractors
    print(FeatureExtractorFactory.list_available())
    # ['csp', 'band_power', 'time_domain']
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Type, Union
import logging

from src.features.base import BaseFeatureExtractor
from src.core.registry import get_registry, ComponentRegistry

# Configure logging
logger = logging.getLogger(__name__)


class FeatureExtractorFactory:
    """
    Factory for creating feature extractors.
    
    Provides a centralized way to create, configure, and manage
    feature extractors. Integrates with the component registry
    for automatic discovery.
    
    Class Methods:
        create: Create extractor by name
        create_from_config: Create from configuration dictionary
        register: Register custom extractor
        unregister: Remove registered extractor
        list_available: List all available extractors
        get_extractor_info: Get information about an extractor
    
    Example:
        >>> csp = FeatureExtractorFactory.create('csp', n_components=6)
        >>> csp.fit(X_train, y_train)
        >>> features = csp.extract(X_test)
    """
    
    # Built-in extractors (loaded lazily)
    _builtin_extractors: Dict[str, Type[BaseFeatureExtractor]] = {}
    _initialized: bool = False
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure built-in extractors are registered."""
        if cls._initialized:
            return
        
        # Import built-in extractors
        from src.features.extractors.csp import CSPExtractor
        from src.features.extractors.band_power import BandPowerExtractor
        from src.features.extractors.time_domain import TimeDomainExtractor
        
        # Register built-ins
        cls._builtin_extractors = {
            'csp': CSPExtractor,
            'band_power': BandPowerExtractor,
            'time_domain': TimeDomainExtractor,
        }
        
        cls._initialized = True
        logger.debug("FeatureExtractorFactory initialized with built-in extractors")
    
    @classmethod
    def create(cls,
               name: str,
               config: Optional[Dict[str, Any]] = None,
               **kwargs) -> BaseFeatureExtractor:
        """
        Create a feature extractor by name.
        
        Args:
            name: Extractor name (e.g., 'csp', 'band_power', 'time_domain')
            config: Optional configuration dictionary for initialize()
            **kwargs: Constructor arguments for the extractor
        
        Returns:
            Configured feature extractor instance
        
        Raises:
            ValueError: If extractor name is not found
        
        Example:
            >>> csp = FeatureExtractorFactory.create('csp', n_components=6)
            >>> bp = FeatureExtractorFactory.create(
            ...     'band_power',
            ...     bands={'mu': (8, 12)},
            ...     config={'sampling_rate': 250}
            ... )
        """
        cls._ensure_initialized()
        
        # Normalize name
        name = name.lower().strip()
        
        # Check built-in extractors first
        if name in cls._builtin_extractors:
            extractor_class = cls._builtin_extractors[name]
            extractor = extractor_class(**kwargs)
            
            if config:
                extractor.initialize(config)
            
            logger.debug(f"Created extractor: {name}")
            return extractor
        
        # Check component registry
        registry = get_registry()
        if registry.has('feature_extractor', name):
            return registry.create('feature_extractor', name, config, **kwargs)
        
        # Not found
        available = cls.list_available()
        raise ValueError(
            f"Unknown feature extractor '{name}'. "
            f"Available: {available}"
        )
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseFeatureExtractor:
        """
        Create extractor from configuration dictionary.
        
        Args:
            config: Configuration dictionary with structure:
                {
                    'type': 'csp',           # Required: extractor type
                    'params': {              # Optional: constructor args
                        'n_components': 6
                    },
                    'init_config': {         # Optional: initialize() args
                        'sampling_rate': 250
                    }
                }
        
        Returns:
            Configured feature extractor
        
        Raises:
            ValueError: If 'type' is missing or invalid
        
        Example:
            >>> config = {
            ...     'type': 'csp',
            ...     'params': {'n_components': 6, 'reg': 0.01}
            ... }
            >>> csp = FeatureExtractorFactory.create_from_config(config)
        """
        if 'type' not in config:
            raise ValueError("Configuration must include 'type' key")
        
        extractor_type = config['type']
        params = config.get('params', {})
        init_config = config.get('init_config', config.get('config', {}))
        
        return cls.create(extractor_type, config=init_config, **params)
    
    @classmethod
    def register(cls,
                 name: str,
                 extractor_class: Type[BaseFeatureExtractor],
                 overwrite: bool = False) -> None:
        """
        Register a custom feature extractor.
        
        Args:
            name: Unique name for the extractor
            extractor_class: Extractor class (must inherit BaseFeatureExtractor)
            overwrite: If True, overwrite existing registration
        
        Raises:
            ValueError: If name exists and overwrite=False
            TypeError: If extractor_class is not a valid extractor
        
        Example:
            >>> class MyExtractor(BaseFeatureExtractor):
            ...     pass
            >>> FeatureExtractorFactory.register('my_extractor', MyExtractor)
        """
        cls._ensure_initialized()
        
        # Validate extractor class
        if not isinstance(extractor_class, type):
            raise TypeError("extractor_class must be a class")
        
        if not issubclass(extractor_class, BaseFeatureExtractor):
            raise TypeError(
                "extractor_class must inherit from BaseFeatureExtractor"
            )
        
        name = name.lower().strip()
        
        # Check existing
        if name in cls._builtin_extractors and not overwrite:
            raise ValueError(
                f"Extractor '{name}' already registered. "
                f"Use overwrite=True to replace."
            )
        
        cls._builtin_extractors[name] = extractor_class
        
        # Also register with component registry
        registry = get_registry()
        registry.register(
            'feature_extractor', 
            name, 
            extractor_class,
            overwrite=overwrite
        )
        
        logger.info(f"Registered feature extractor: {name}")
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Remove a registered extractor.
        
        Args:
            name: Extractor name to remove
        
        Returns:
            True if removed, False if not found
        """
        cls._ensure_initialized()
        name = name.lower().strip()
        
        if name in cls._builtin_extractors:
            del cls._builtin_extractors[name]
            
            # Also remove from registry
            registry = get_registry()
            if registry.has('feature_extractor', name):
                registry.unregister('feature_extractor', name)
            
            logger.info(f"Unregistered feature extractor: {name}")
            return True
        
        return False
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available feature extractors.
        
        Returns:
            List of extractor names
        
        Example:
            >>> FeatureExtractorFactory.list_available()
            ['csp', 'band_power', 'time_domain']
        """
        cls._ensure_initialized()
        
        # Combine built-ins and registry
        extractors = set(cls._builtin_extractors.keys())
        
        registry = get_registry()
        registry_extractors = registry.list('feature_extractor')
        if registry_extractors:
            extractors.update(registry_extractors)
        
        return sorted(list(extractors))
    
    @classmethod
    def get_extractor_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a feature extractor.
        
        Args:
            name: Extractor name
        
        Returns:
            Dictionary with extractor information
        
        Example:
            >>> info = FeatureExtractorFactory.get_extractor_info('csp')
            >>> print(info['description'])
        """
        cls._ensure_initialized()
        name = name.lower().strip()
        
        if name not in cls._builtin_extractors:
            available = cls.list_available()
            raise ValueError(
                f"Unknown extractor '{name}'. Available: {available}"
            )
        
        extractor_class = cls._builtin_extractors[name]
        
        # Create temp instance to get info
        temp_instance = extractor_class()
        
        info = {
            'name': name,
            'class': extractor_class.__name__,
            'module': extractor_class.__module__,
            'is_trainable': temp_instance.is_trainable,
            'description': extractor_class.__doc__.split('\n')[0] if extractor_class.__doc__ else '',
        }
        
        return info
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the factory (mainly for testing).
        
        Clears all custom registrations and reinitializes.
        """
        cls._builtin_extractors.clear()
        cls._initialized = False
        logger.debug("FeatureExtractorFactory reset")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_extractor(name: str, **kwargs) -> BaseFeatureExtractor:
    """
    Convenience function to create an extractor.
    
    Args:
        name: Extractor name
        **kwargs: Constructor and config arguments
    
    Returns:
        Feature extractor instance
    
    Example:
        >>> csp = create_extractor('csp', n_components=6)
    """
    # Separate config from constructor args
    config_keys = ['sampling_rate', 'n_channels', 'n_samples', 'channel_names']
    config = {}
    constructor_kwargs = {}
    
    for key, value in kwargs.items():
        if key in config_keys or key == 'config':
            if key == 'config':
                config.update(value)
            else:
                config[key] = value
        else:
            constructor_kwargs[key] = value
    
    extractor = FeatureExtractorFactory.create(
        name, 
        config=config if config else None,
        **constructor_kwargs
    )
    
    return extractor


def list_extractors() -> List[str]:
    """
    List all available feature extractors.
    
    Returns:
        List of extractor names
    """
    return FeatureExtractorFactory.list_available()
