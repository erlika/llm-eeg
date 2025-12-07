"""
Custom Exceptions
=================

This module defines all custom exceptions for the EEG-BCI framework.

Exception Hierarchy:
-------------------
BCIFrameworkError (Base)
├── DataError
│   ├── DataLoadError
│   ├── DataValidationError
│   ├── DataFormatError
│   └── MissingDataError
├── ProcessingError
│   ├── PreprocessingError
│   ├── FeatureExtractionError
│   └── FilterError
├── ClassificationError
│   ├── ModelNotFittedError
│   ├── ModelNotFoundError
│   └── PredictionError
├── AgentError
│   ├── PolicyError
│   ├── RewardError
│   └── AgentNotInitializedError
├── LLMError
│   ├── LLMNotLoadedError
│   ├── GenerationError
│   └── PromptError
├── ConfigurationError
│   ├── ConfigNotFoundError
│   ├── ConfigValidationError
│   └── MissingConfigError
├── StorageError
│   ├── StorageReadError
│   ├── StorageWriteError
│   └── CheckpointError
└── ComponentError
    ├── ComponentNotFoundError
    ├── RegistrationError
    └── InitializationError

Example Usage:
    ```python
    from src.core.exceptions import DataLoadError, ModelNotFittedError
    
    try:
        data = loader.load(path)
    except DataLoadError as e:
        logger.error(f"Failed to load data: {e}")
        # Handle error...
    ```

Author: EEG-BCI Framework
Date: 2024
"""


# =============================================================================
# BASE EXCEPTION
# =============================================================================

class BCIFrameworkError(Exception):
    """
    Base exception for all BCI framework errors.
    
    All custom exceptions should inherit from this class.
    Provides consistent error message formatting.
    
    Attributes:
        message: Error message
        details: Additional error details
        suggestion: Suggestion for fixing the error
    """
    
    def __init__(self, 
                 message: str, 
                 details: str = '',
                 suggestion: str = ''):
        self.message = message
        self.details = details
        self.suggestion = suggestion
        
        full_message = message
        if details:
            full_message += f"\nDetails: {details}"
        if suggestion:
            full_message += f"\nSuggestion: {suggestion}"
        
        super().__init__(full_message)


# =============================================================================
# DATA ERRORS
# =============================================================================

class DataError(BCIFrameworkError):
    """Base exception for data-related errors."""
    pass


class DataLoadError(DataError):
    """Raised when data cannot be loaded from a file."""
    
    def __init__(self, 
                 file_path: str, 
                 reason: str = 'Unknown error',
                 original_error: Exception = None):
        message = f"Failed to load data from '{file_path}'"
        details = reason
        
        if original_error:
            details += f" (Original error: {original_error})"
        
        suggestion = "Check that the file exists and has the correct format."
        
        super().__init__(message, details, suggestion)
        self.file_path = file_path
        self.original_error = original_error


class DataValidationError(DataError):
    """Raised when data validation fails."""
    
    def __init__(self, 
                 field: str, 
                 expected: str, 
                 actual: str):
        message = f"Data validation failed for '{field}'"
        details = f"Expected: {expected}, Got: {actual}"
        suggestion = "Check data format and preprocessing steps."
        
        super().__init__(message, details, suggestion)
        self.field = field
        self.expected = expected
        self.actual = actual


class DataFormatError(DataError):
    """Raised when data has incorrect format."""
    
    def __init__(self, 
                 expected_format: str, 
                 actual_format: str = 'unknown'):
        message = "Invalid data format"
        details = f"Expected: {expected_format}, Got: {actual_format}"
        suggestion = "Convert data to the expected format."
        
        super().__init__(message, details, suggestion)


class MissingDataError(DataError):
    """Raised when required data is missing."""
    
    def __init__(self, 
                 data_type: str, 
                 location: str = ''):
        message = f"Required data '{data_type}' is missing"
        details = f"Expected location: {location}" if location else ""
        suggestion = "Ensure all required data files are available."
        
        super().__init__(message, details, suggestion)


class ChannelNotFoundError(DataError):
    """Raised when a requested channel is not found."""
    
    def __init__(self, 
                 channel: str, 
                 available_channels: list = None):
        message = f"Channel '{channel}' not found"
        details = f"Available channels: {available_channels}" if available_channels else ""
        suggestion = "Check channel name spelling or select from available channels."
        
        super().__init__(message, details, suggestion)
        self.channel = channel
        self.available_channels = available_channels


# =============================================================================
# PROCESSING ERRORS
# =============================================================================

class ProcessingError(BCIFrameworkError):
    """Base exception for signal processing errors."""
    pass


class PreprocessingError(ProcessingError):
    """Raised when preprocessing fails."""
    
    def __init__(self, 
                 step: str, 
                 reason: str = ''):
        message = f"Preprocessing step '{step}' failed"
        details = reason
        suggestion = "Check preprocessing parameters and input data."
        
        super().__init__(message, details, suggestion)
        self.step = step


class FeatureExtractionError(ProcessingError):
    """Raised when feature extraction fails."""
    
    def __init__(self, 
                 extractor: str, 
                 reason: str = ''):
        message = f"Feature extraction with '{extractor}' failed"
        details = reason
        suggestion = "Check that input data is properly preprocessed."
        
        super().__init__(message, details, suggestion)
        self.extractor = extractor


class FilterError(ProcessingError):
    """Raised when filtering fails."""
    
    def __init__(self, 
                 filter_type: str, 
                 reason: str = ''):
        message = f"Filter '{filter_type}' failed"
        details = reason
        suggestion = "Check filter parameters (e.g., frequency bounds, sampling rate)."
        
        super().__init__(message, details, suggestion)


# =============================================================================
# CLASSIFICATION ERRORS
# =============================================================================

class ClassificationError(BCIFrameworkError):
    """Base exception for classification errors."""
    pass


class ModelNotFittedError(ClassificationError):
    """Raised when trying to use an unfitted model."""
    
    def __init__(self, model_name: str = 'Model'):
        message = f"{model_name} has not been fitted"
        details = "The model must be trained before making predictions."
        suggestion = "Call fit() with training data before predict()."
        
        super().__init__(message, details, suggestion)
        self.model_name = model_name


class ModelNotFoundError(ClassificationError):
    """Raised when a model file is not found."""
    
    def __init__(self, path: str):
        message = f"Model not found at '{path}'"
        details = "The specified model file does not exist."
        suggestion = "Check the model path or train a new model."
        
        super().__init__(message, details, suggestion)
        self.path = path


class PredictionError(ClassificationError):
    """Raised when prediction fails."""
    
    def __init__(self, 
                 reason: str = '', 
                 input_shape: tuple = None):
        message = "Prediction failed"
        details = reason
        if input_shape:
            details += f" (Input shape: {input_shape})"
        suggestion = "Check that input data matches the expected format."
        
        super().__init__(message, details, suggestion)


# =============================================================================
# AGENT ERRORS
# =============================================================================

class AgentError(BCIFrameworkError):
    """Base exception for agent-related errors."""
    pass


class AgentNotInitializedError(AgentError):
    """Raised when agent is used before initialization."""
    
    def __init__(self, agent_name: str):
        message = f"Agent '{agent_name}' has not been initialized"
        details = "The agent must be initialized before use."
        suggestion = "Call initialize() with appropriate configuration."
        
        super().__init__(message, details, suggestion)
        self.agent_name = agent_name


class PolicyError(AgentError):
    """Raised when there's an error with the agent's policy."""
    
    def __init__(self, 
                 policy_name: str, 
                 reason: str = ''):
        message = f"Policy error in '{policy_name}'"
        details = reason
        suggestion = "Check policy configuration and state space definition."
        
        super().__init__(message, details, suggestion)
        self.policy_name = policy_name


class RewardError(AgentError):
    """Raised when reward computation fails."""
    
    def __init__(self, 
                 reward_type: str, 
                 reason: str = ''):
        message = f"Reward computation failed for '{reward_type}'"
        details = reason
        suggestion = "Check reward function configuration and inputs."
        
        super().__init__(message, details, suggestion)


class InvalidStateError(AgentError):
    """Raised when agent state is invalid."""
    
    def __init__(self, 
                 expected_keys: list = None,
                 missing_keys: list = None):
        message = "Invalid agent state"
        details = ""
        if expected_keys:
            details += f"Expected keys: {expected_keys}. "
        if missing_keys:
            details += f"Missing keys: {missing_keys}"
        suggestion = "Ensure state dictionary contains all required keys."
        
        super().__init__(message, details, suggestion)


# =============================================================================
# LLM ERRORS
# =============================================================================

class LLMError(BCIFrameworkError):
    """Base exception for LLM-related errors."""
    pass


class LLMNotLoadedError(LLMError):
    """Raised when LLM is used before loading."""
    
    def __init__(self, provider: str = 'LLM'):
        message = f"{provider} model has not been loaded"
        details = "The model must be loaded before generating text."
        suggestion = "Call initialize() to load the model."
        
        super().__init__(message, details, suggestion)


class GenerationError(LLMError):
    """Raised when text generation fails."""
    
    def __init__(self, 
                 reason: str = '', 
                 prompt_length: int = None):
        message = "Text generation failed"
        details = reason
        if prompt_length:
            details += f" (Prompt length: {prompt_length} tokens)"
        suggestion = "Check prompt length and generation parameters."
        
        super().__init__(message, details, suggestion)


class PromptError(LLMError):
    """Raised when there's an issue with the prompt."""
    
    def __init__(self, 
                 issue: str, 
                 max_length: int = None):
        message = f"Prompt error: {issue}"
        details = f"Maximum allowed length: {max_length}" if max_length else ""
        suggestion = "Reduce prompt length or modify prompt content."
        
        super().__init__(message, details, suggestion)


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(BCIFrameworkError):
    """Base exception for configuration errors."""
    pass


class ConfigNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""
    
    def __init__(self, path: str):
        message = f"Configuration file not found: '{path}'"
        details = "The specified configuration file does not exist."
        suggestion = "Check the file path or create the configuration file."
        
        super().__init__(message, details, suggestion)
        self.path = path


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, 
                 key: str, 
                 expected: str, 
                 actual: str = ''):
        message = f"Invalid configuration value for '{key}'"
        details = f"Expected: {expected}"
        if actual:
            details += f", Got: {actual}"
        suggestion = "Update the configuration with a valid value."
        
        super().__init__(message, details, suggestion)
        self.key = key


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, key: str):
        message = f"Required configuration '{key}' is missing"
        details = "This configuration value must be provided."
        suggestion = "Add the required configuration to your config file."
        
        super().__init__(message, details, suggestion)
        self.key = key


# =============================================================================
# STORAGE ERRORS
# =============================================================================

class StorageError(BCIFrameworkError):
    """Base exception for storage errors."""
    pass


class StorageReadError(StorageError):
    """Raised when reading from storage fails."""
    
    def __init__(self, 
                 path: str, 
                 reason: str = ''):
        message = f"Failed to read from storage: '{path}'"
        details = reason
        suggestion = "Check storage connection and file permissions."
        
        super().__init__(message, details, suggestion)
        self.path = path


class StorageWriteError(StorageError):
    """Raised when writing to storage fails."""
    
    def __init__(self, 
                 path: str, 
                 reason: str = ''):
        message = f"Failed to write to storage: '{path}'"
        details = reason
        suggestion = "Check storage connection, permissions, and available space."
        
        super().__init__(message, details, suggestion)
        self.path = path


class CheckpointError(StorageError):
    """Raised when checkpoint operations fail."""
    
    def __init__(self, 
                 operation: str, 
                 checkpoint_name: str):
        message = f"Checkpoint {operation} failed for '{checkpoint_name}'"
        details = f"The {operation} operation could not be completed."
        suggestion = "Check checkpoint configuration and storage availability."
        
        super().__init__(message, details, suggestion)
        self.checkpoint_name = checkpoint_name


# =============================================================================
# COMPONENT ERRORS
# =============================================================================

class ComponentError(BCIFrameworkError):
    """Base exception for component registry errors."""
    pass


class ComponentNotFoundError(ComponentError):
    """Raised when a component is not found in registry."""
    
    def __init__(self, 
                 category: str, 
                 name: str, 
                 available: list = None):
        message = f"Component '{name}' not found in category '{category}'"
        details = f"Available components: {available}" if available else ""
        suggestion = "Register the component or use an existing one."
        
        super().__init__(message, details, suggestion)
        self.category = category
        self.name = name
        self.available = available


class RegistrationError(ComponentError):
    """Raised when component registration fails."""
    
    def __init__(self, 
                 category: str, 
                 name: str, 
                 reason: str = ''):
        message = f"Failed to register component '{name}' in '{category}'"
        details = reason
        suggestion = "Check component class and registration parameters."
        
        super().__init__(message, details, suggestion)


class InitializationError(ComponentError):
    """Raised when component initialization fails."""
    
    def __init__(self, 
                 component: str, 
                 reason: str = ''):
        message = f"Failed to initialize component '{component}'"
        details = reason
        suggestion = "Check initialization configuration."
        
        super().__init__(message, details, suggestion)
        self.component = component


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base
    'BCIFrameworkError',
    
    # Data
    'DataError',
    'DataLoadError',
    'DataValidationError',
    'DataFormatError',
    'MissingDataError',
    'ChannelNotFoundError',
    
    # Processing
    'ProcessingError',
    'PreprocessingError',
    'FeatureExtractionError',
    'FilterError',
    
    # Classification
    'ClassificationError',
    'ModelNotFittedError',
    'ModelNotFoundError',
    'PredictionError',
    
    # Agent
    'AgentError',
    'AgentNotInitializedError',
    'PolicyError',
    'RewardError',
    'InvalidStateError',
    
    # LLM
    'LLMError',
    'LLMNotLoadedError',
    'GenerationError',
    'PromptError',
    
    # Configuration
    'ConfigurationError',
    'ConfigNotFoundError',
    'ConfigValidationError',
    'MissingConfigError',
    
    # Storage
    'StorageError',
    'StorageReadError',
    'StorageWriteError',
    'CheckpointError',
    
    # Component
    'ComponentError',
    'ComponentNotFoundError',
    'RegistrationError',
    'InitializationError',
]
