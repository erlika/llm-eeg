"""
Component Registry
==================

This module implements the central component registry for the EEG-BCI framework.

The registry is responsible for:
- Managing all pluggable components (loaders, preprocessors, classifiers, etc.)
- Dynamic component instantiation based on configuration
- Component discovery and registration
- Dependency injection for loose coupling

Design Pattern: Service Locator + Factory
----------------------------------------
The registry acts as a service locator for component types and a factory
for creating instances. This enables:
1. Plugin architecture: New components can be added without modifying existing code
2. Configuration-driven instantiation: Components specified in YAML/JSON
3. Dependency injection: Components receive dependencies via constructor
4. Testing: Easy mocking by registering test implementations

Component Categories:
--------------------
- data_loader: EEG data loaders (mat, gdf, edf, moabb)
- preprocessor: Signal preprocessing steps (filter, artifact removal)
- feature_extractor: Feature extraction methods (CSP, band power)
- classifier: Classification models (SVM, EEGNet, EEG-DCNet)
- agent: AI agents (APA, DVA)
- policy: RL policies (Q-learning, DQN)
- reward: Reward functions for agents
- llm_provider: LLM providers (Phi-3, Llama)
- storage: Storage backends (local, Google Drive)

Example Usage:
    ```python
    from src.core.registry import ComponentRegistry
    
    # Get singleton instance
    registry = ComponentRegistry.get_instance()
    
    # Register a custom component
    registry.register('data_loader', 'custom', MyCustomLoader)
    
    # Create component from configuration
    loader = registry.create('data_loader', 'mat', {
        'channels': ['C3', 'C4', 'Cz']
    })
    
    # List available components
    print(registry.list('data_loader'))
    # ['mat', 'gdf', 'edf', 'moabb', 'custom']
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Type, Callable, Union
from collections import defaultdict
import logging
import threading

# Configure logging
logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Central registry for all framework components.
    
    Implements the Singleton pattern to ensure a single registry instance.
    Thread-safe for concurrent access.
    
    Attributes:
        _instance: Singleton instance
        _lock: Thread lock for safe concurrent access
        _components: Dictionary of registered components
        _factories: Dictionary of factory functions
        _metadata: Component metadata for documentation
    
    Component Registration:
        Components can be registered as:
        1. Class: Will be instantiated when created
        2. Factory function: Will be called to create instance
        3. Instance: Will be returned directly (singleton component)
    """
    
    _instance: Optional['ComponentRegistry'] = None
    _lock: threading.Lock = threading.Lock()
    
    # =========================================================================
    # SINGLETON PATTERN
    # =========================================================================
    
    def __new__(cls) -> 'ComponentRegistry':
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry (only once for singleton)."""
        if self._initialized:
            return
        
        # Component storage: category -> name -> class/factory
        self._components: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Factory functions: category -> name -> callable
        self._factories: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        
        # Component metadata: category -> name -> metadata dict
        self._metadata: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        
        # Valid component categories
        self._categories: List[str] = [
            'data_loader',
            'preprocessor',
            'feature_extractor',
            'classifier',
            'agent',
            'policy',
            'reward',
            'llm_provider',
            'storage',
            'validator',       # For DVA validators
            'pipeline',        # Pipeline components
            'metric'           # Evaluation metrics
        ]
        
        self._initialized = True
        logger.info("ComponentRegistry initialized")
    
    @classmethod
    def get_instance(cls) -> 'ComponentRegistry':
        """
        Get the singleton registry instance.
        
        Returns:
            ComponentRegistry: The singleton instance
            
        Example:
            >>> registry = ComponentRegistry.get_instance()
        """
        return cls()
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the registry (mainly for testing).
        
        Clears all registrations and allows re-initialization.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._components.clear()
                cls._instance._factories.clear()
                cls._instance._metadata.clear()
                cls._instance._initialized = False
            cls._instance = None
        logger.info("ComponentRegistry reset")
    
    # =========================================================================
    # REGISTRATION METHODS
    # =========================================================================
    
    def register(self,
                 category: str,
                 name: str,
                 component: Union[Type, Callable],
                 metadata: Optional[Dict[str, Any]] = None,
                 overwrite: bool = False) -> 'ComponentRegistry':
        """
        Register a component with the registry.
        
        Args:
            category: Component category (e.g., 'data_loader', 'classifier')
            name: Unique name within the category
            component: Component class or factory function
            metadata: Optional metadata (description, version, etc.)
            overwrite: Whether to overwrite existing registration
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If category is invalid or component already exists
            
        Example:
            >>> registry.register('data_loader', 'mat', MatLoader, {
            ...     'description': 'MATLAB .mat file loader',
            ...     'version': '1.0.0',
            ...     'extensions': ['.mat']
            ... })
        """
        # Validate category
        if category not in self._categories:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Valid categories: {self._categories}"
            )
        
        # Check for existing registration
        if name in self._components[category] and not overwrite:
            raise ValueError(
                f"Component '{name}' already registered in category '{category}'. "
                f"Use overwrite=True to replace."
            )
        
        # Register component
        self._components[category][name] = component
        
        # Store metadata
        default_metadata = {
            'name': name,
            'category': category,
            'type': 'class' if isinstance(component, type) else 'factory',
            'module': getattr(component, '__module__', 'unknown'),
            'class_name': getattr(component, '__name__', str(component))
        }
        self._metadata[category][name] = {**default_metadata, **(metadata or {})}
        
        logger.debug(f"Registered {category}/{name}")
        return self
    
    def register_factory(self,
                         category: str,
                         name: str,
                         factory: Callable,
                         metadata: Optional[Dict[str, Any]] = None) -> 'ComponentRegistry':
        """
        Register a factory function for component creation.
        
        Factory functions provide more control over instantiation,
        allowing complex setup, caching, or conditional creation.
        
        Args:
            category: Component category
            name: Component name
            factory: Callable that creates the component
            metadata: Optional metadata
        
        Returns:
            Self for method chaining
            
        Example:
            >>> def create_eegnet(config):
            ...     model = EEGNet()
            ...     model.initialize(config)
            ...     return model
            >>> registry.register_factory('classifier', 'eegnet', create_eegnet)
        """
        self._factories[category][name] = factory
        
        # Also register in components for listing
        self._components[category][name] = factory
        self._metadata[category][name] = {
            'name': name,
            'category': category,
            'type': 'factory',
            **(metadata or {})
        }
        
        logger.debug(f"Registered factory {category}/{name}")
        return self
    
    def unregister(self, category: str, name: str) -> 'ComponentRegistry':
        """
        Remove a component from the registry.
        
        Args:
            category: Component category
            name: Component name
        
        Returns:
            Self for method chaining
        """
        if name in self._components.get(category, {}):
            del self._components[category][name]
            self._metadata[category].pop(name, None)
            self._factories[category].pop(name, None)
            logger.debug(f"Unregistered {category}/{name}")
        
        return self
    
    # =========================================================================
    # COMPONENT CREATION
    # =========================================================================
    
    def create(self,
               category: str,
               name: str,
               config: Optional[Dict[str, Any]] = None,
               **kwargs) -> Any:
        """
        Create a component instance.
        
        Instantiates the component and optionally initializes it with config.
        
        Args:
            category: Component category
            name: Component name
            config: Configuration dictionary for initialization
            **kwargs: Additional arguments passed to constructor
        
        Returns:
            Component instance
        
        Raises:
            KeyError: If component is not registered
            RuntimeError: If instantiation fails
            
        Example:
            >>> loader = registry.create('data_loader', 'mat', {
            ...     'channels': ['C3', 'C4'],
            ...     'preload': True
            ... })
        """
        # Check registration
        if name not in self._components.get(category, {}):
            available = self.list(category)
            raise KeyError(
                f"Component '{name}' not found in category '{category}'. "
                f"Available: {available}"
            )
        
        component = self._components[category][name]
        config = config or {}
        
        try:
            # Check if it's a factory function
            if name in self._factories.get(category, {}):
                instance = self._factories[category][name](config, **kwargs)
            
            # If it's a class, instantiate it
            elif isinstance(component, type):
                instance = component(**kwargs)
                # Initialize if the method exists and config provided
                if hasattr(instance, 'initialize') and config:
                    instance.initialize(config)
            
            # If it's already an instance (singleton), return it
            else:
                instance = component
            
            logger.debug(f"Created {category}/{name}")
            return instance
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to create component '{category}/{name}': {e}"
            ) from e
    
    def create_from_config(self, config: Dict[str, Any]) -> Any:
        """
        Create component from a full configuration dictionary.
        
        Useful for creating components specified in YAML/JSON configs.
        
        Args:
            config: Configuration with required 'category' and 'name' keys
                - 'category': Component category
                - 'name': Component name
                - 'params': Optional initialization parameters
        
        Returns:
            Component instance
            
        Example:
            >>> config = {
            ...     'category': 'classifier',
            ...     'name': 'eegnet',
            ...     'params': {'n_classes': 4, 'n_channels': 22}
            ... }
            >>> classifier = registry.create_from_config(config)
        """
        category = config['category']
        name = config['name']
        params = config.get('params', {})
        
        return self.create(category, name, params)
    
    # =========================================================================
    # QUERY METHODS
    # =========================================================================
    
    def list(self, category: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
        """
        List registered components.
        
        Args:
            category: If specified, list only this category.
                     If None, list all categories.
        
        Returns:
            List of component names (if category specified)
            or Dict of category -> names (if no category)
            
        Example:
            >>> registry.list('classifier')
            ['svm', 'lda', 'eegnet', 'eeg_dcnet']
            >>> registry.list()
            {'data_loader': ['mat', 'gdf'], 'classifier': ['svm', 'eegnet'], ...}
        """
        if category:
            return list(self._components.get(category, {}).keys())
        else:
            return {
                cat: list(comps.keys())
                for cat, comps in self._components.items()
                if comps  # Only include non-empty categories
            }
    
    def get(self, category: str, name: str) -> Any:
        """
        Get a registered component (class/factory, not instance).
        
        Args:
            category: Component category
            name: Component name
        
        Returns:
            Registered component class or factory
        
        Raises:
            KeyError: If not found
        """
        if name not in self._components.get(category, {}):
            raise KeyError(f"Component '{name}' not found in '{category}'")
        return self._components[category][name]
    
    def has(self, category: str, name: str) -> bool:
        """
        Check if a component is registered.
        
        Args:
            category: Component category
            name: Component name
        
        Returns:
            bool: True if registered
        """
        return name in self._components.get(category, {})
    
    def get_metadata(self, 
                     category: str, 
                     name: str) -> Dict[str, Any]:
        """
        Get component metadata.
        
        Args:
            category: Component category
            name: Component name
        
        Returns:
            Metadata dictionary
        """
        return self._metadata.get(category, {}).get(name, {})
    
    def get_categories(self) -> List[str]:
        """
        Get all valid component categories.
        
        Returns:
            List of category names
        """
        return self._categories.copy()
    
    # =========================================================================
    # DISCOVERY AND AUTO-REGISTRATION
    # =========================================================================
    
    def discover(self, 
                 package: str,
                 category: str,
                 base_class: Optional[Type] = None) -> int:
        """
        Discover and register components from a package.
        
        Scans a Python package for classes implementing a base interface
        and automatically registers them.
        
        Args:
            package: Package path to scan (e.g., 'src.data.loaders')
            category: Category to register discovered components
            base_class: Optional base class to filter (only register subclasses)
        
        Returns:
            int: Number of components discovered and registered
            
        Example:
            >>> count = registry.discover(
            ...     'src.data.loaders',
            ...     'data_loader',
            ...     base_class=IDataLoader
            ... )
            >>> print(f"Discovered {count} loaders")
        """
        import importlib
        import pkgutil
        
        count = 0
        
        try:
            module = importlib.import_module(package)
            
            for _, name, _ in pkgutil.walk_packages(
                module.__path__, 
                prefix=f"{package}."
            ):
                try:
                    submodule = importlib.import_module(name)
                    
                    for attr_name in dir(submodule):
                        attr = getattr(submodule, attr_name)
                        
                        # Check if it's a class and optionally a subclass
                        if isinstance(attr, type):
                            if base_class is None or (
                                issubclass(attr, base_class) and 
                                attr is not base_class
                            ):
                                # Use class name in lowercase as component name
                                component_name = attr_name.lower().replace('_', '')
                                
                                if not self.has(category, component_name):
                                    self.register(category, component_name, attr)
                                    count += 1
                                    
                except ImportError as e:
                    logger.warning(f"Failed to import {name}: {e}")
                    
        except ImportError as e:
            logger.error(f"Failed to import package {package}: {e}")
        
        logger.info(f"Discovered {count} components in {package}")
        return count
    
    def register_defaults(self) -> 'ComponentRegistry':
        """
        Register default framework components.
        
        Called during framework initialization to register built-in components.
        
        Returns:
            Self for method chaining
        """
        # This will be populated as we implement components
        # For now, just log that it was called
        logger.info("Registering default components...")
        
        # TODO: Import and register built-in components
        # Example:
        # from src.data.loaders import MatLoader, GDFLoader
        # self.register('data_loader', 'mat', MatLoader)
        # self.register('data_loader', 'gdf', GDFLoader)
        
        return self
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def summary(self) -> str:
        """
        Get a summary of registered components.
        
        Returns:
            str: Formatted summary string
        """
        lines = ["Component Registry Summary", "=" * 40]
        
        for category in sorted(self._categories):
            components = self.list(category)
            if components:
                lines.append(f"\n{category}:")
                for name in sorted(components):
                    meta = self.get_metadata(category, name)
                    desc = meta.get('description', 'No description')
                    lines.append(f"  - {name}: {desc[:50]}...")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation."""
        total = sum(len(c) for c in self._components.values())
        return f"ComponentRegistry(components={total})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_registry() -> ComponentRegistry:
    """
    Get the singleton ComponentRegistry instance.
    
    Convenience function for easy access.
    
    Returns:
        ComponentRegistry: The singleton instance
    """
    return ComponentRegistry.get_instance()


def register(category: str, 
             name: str, 
             component: Union[Type, Callable],
             **kwargs) -> None:
    """
    Register a component with the global registry.
    
    Convenience function for registration.
    
    Args:
        category: Component category
        name: Component name
        component: Component class or factory
        **kwargs: Additional registration options
    """
    get_registry().register(category, name, component, **kwargs)


def create(category: str,
           name: str,
           config: Optional[Dict[str, Any]] = None,
           **kwargs) -> Any:
    """
    Create a component from the global registry.
    
    Convenience function for component creation.
    
    Args:
        category: Component category
        name: Component name
        config: Configuration dictionary
        **kwargs: Additional arguments
    
    Returns:
        Component instance
    """
    return get_registry().create(category, name, config, **kwargs)


# =============================================================================
# DECORATOR FOR REGISTRATION
# =============================================================================

def registered(category: str,
               name: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for automatic component registration.
    
    Apply to a class to automatically register it with the registry.
    
    Args:
        category: Component category
        name: Component name (defaults to lowercase class name)
        metadata: Optional metadata
    
    Example:
        >>> @registered('classifier', 'my_svm')
        ... class MySVMClassifier(IClassifier):
        ...     pass
    """
    def decorator(cls):
        component_name = name or cls.__name__.lower()
        get_registry().register(category, component_name, cls, metadata)
        return cls
    
    return decorator
