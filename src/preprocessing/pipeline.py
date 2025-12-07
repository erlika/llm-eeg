"""
Preprocessing Pipeline
======================

This module implements the preprocessing pipeline for EEG data.
It allows composing multiple preprocessing steps into a sequential pipeline.

Pipeline Design:
---------------
The pipeline follows a linear execution model where each step's output
becomes the next step's input. This design is inspired by scikit-learn's
Pipeline but optimized for EEG data processing.

Pipeline Execution Flow:
    Raw EEG → Step1 → Step2 → ... → StepN → Processed EEG

Standard BCI Processing Pipeline:
1. Notch filter (remove 50/60 Hz line noise)
2. Bandpass filter (8-30 Hz for motor imagery)
3. Normalization (z-score or channel-wise)
4. (Optional) Artifact rejection
5. (Optional) Spatial filtering (CAR, Laplacian)

Features:
---------
- Sequential step execution
- Automatic parameter propagation (e.g., sampling rate)
- Step-wise debugging and inspection
- Configurable from YAML/dict
- Support for both EEGData and numpy arrays

Usage Example:
    ```python
    from src.preprocessing import PreprocessingPipeline
    from src.preprocessing.steps import BandpassFilter, NotchFilter, Normalization
    
    # Create pipeline
    pipeline = PreprocessingPipeline()
    
    # Add steps
    pipeline.add_step(NotchFilter(), {'notch_freq': 50})
    pipeline.add_step(BandpassFilter(), {'low_freq': 8, 'high_freq': 30})
    pipeline.add_step(Normalization(), {'method': 'zscore'})
    
    # Initialize with common config
    pipeline.initialize({'sampling_rate': 250})
    
    # Process data
    processed = pipeline.process(raw_eeg)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import logging
import time

from src.core.interfaces.i_preprocessor import IPreprocessor
from src.core.types.eeg_data import EEGData


# Configure module logger
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Composable preprocessing pipeline for EEG data.
    
    This class allows building a sequence of preprocessing steps
    that are applied in order to the input data.
    
    Attributes:
        _steps (List): List of (name, preprocessor, config) tuples
        _common_config (Dict): Configuration shared across all steps
        _is_initialized (bool): Whether pipeline has been initialized
        _verbose (bool): Enable verbose logging
        _timing (bool): Track execution time per step
    
    Pipeline Properties:
        - Steps are executed in order of addition
        - Each step receives the output of the previous step
        - Common configuration (e.g., sampling_rate) is propagated
        - Pipeline can be serialized and loaded from config
    
    Example:
        >>> pipeline = PreprocessingPipeline()
        >>> pipeline.add_step('notch', NotchFilter(), {'notch_freq': 50})
        >>> pipeline.add_step('bandpass', BandpassFilter(), {'low_freq': 8})
        >>> pipeline.initialize({'sampling_rate': 250})
        >>> processed = pipeline.process(raw_data)
    """
    
    def __init__(self, verbose: bool = False, timing: bool = False):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            verbose: Enable verbose logging
            timing: Track execution time for each step
        """
        # List of (name, preprocessor, config) tuples
        self._steps: List[Tuple[str, IPreprocessor, Dict[str, Any]]] = []
        
        # Common configuration for all steps
        self._common_config: Dict[str, Any] = {}
        
        # State tracking
        self._is_initialized: bool = False
        self._verbose: bool = verbose
        self._timing: bool = timing
        
        # Execution history
        self._execution_times: Dict[str, float] = {}
        
        logger.debug("PreprocessingPipeline instantiated")
    
    # =========================================================================
    # PIPELINE CONSTRUCTION
    # =========================================================================
    
    def add_step(
        self,
        preprocessor: IPreprocessor,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> 'PreprocessingPipeline':
        """
        Add a preprocessing step to the pipeline.
        
        Steps are executed in the order they are added.
        
        Args:
            preprocessor: Preprocessor instance implementing IPreprocessor
            config: Step-specific configuration (merged with common config)
            name: Optional name for the step (defaults to preprocessor.name)
        
        Returns:
            Self for method chaining
        
        Raises:
            TypeError: If preprocessor doesn't implement IPreprocessor
        
        Example:
            >>> pipeline.add_step(BandpassFilter(), {'low_freq': 8, 'high_freq': 30})
            >>> pipeline.add_step(Normalization(), {'method': 'zscore'}, name='norm')
        """
        # Validate preprocessor
        if not isinstance(preprocessor, IPreprocessor):
            raise TypeError(
                f"Expected IPreprocessor, got {type(preprocessor).__name__}"
            )
        
        # Generate name if not provided
        if name is None:
            name = preprocessor.name
            # Handle duplicate names
            existing_names = [n for n, _, _ in self._steps]
            if name in existing_names:
                count = sum(1 for n in existing_names if n.startswith(name))
                name = f"{name}_{count + 1}"
        
        # Store step
        self._steps.append((name, preprocessor, config or {}))
        
        logger.debug(f"Added step '{name}' to pipeline")
        
        # Reset initialization state
        self._is_initialized = False
        
        return self
    
    def insert_step(
        self,
        index: int,
        preprocessor: IPreprocessor,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> 'PreprocessingPipeline':
        """
        Insert a preprocessing step at a specific position.
        
        Args:
            index: Position to insert at (0 = first)
            preprocessor: Preprocessor instance
            config: Step-specific configuration
            name: Optional step name
        
        Returns:
            Self for method chaining
        """
        if not isinstance(preprocessor, IPreprocessor):
            raise TypeError(
                f"Expected IPreprocessor, got {type(preprocessor).__name__}"
            )
        
        if name is None:
            name = preprocessor.name
        
        self._steps.insert(index, (name, preprocessor, config or {}))
        self._is_initialized = False
        
        logger.debug(f"Inserted step '{name}' at position {index}")
        return self
    
    def remove_step(self, name_or_index: Union[str, int]) -> 'PreprocessingPipeline':
        """
        Remove a step from the pipeline.
        
        Args:
            name_or_index: Step name or index to remove
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If step not found
        """
        if isinstance(name_or_index, int):
            if 0 <= name_or_index < len(self._steps):
                removed = self._steps.pop(name_or_index)
                logger.debug(f"Removed step '{removed[0]}' from position {name_or_index}")
            else:
                raise ValueError(f"Invalid index: {name_or_index}")
        else:
            for i, (name, _, _) in enumerate(self._steps):
                if name == name_or_index:
                    self._steps.pop(i)
                    logger.debug(f"Removed step '{name}'")
                    break
            else:
                raise ValueError(f"Step not found: {name_or_index}")
        
        self._is_initialized = False
        return self
    
    def clear(self) -> 'PreprocessingPipeline':
        """
        Remove all steps from the pipeline.
        
        Returns:
            Self for method chaining
        """
        self._steps.clear()
        self._is_initialized = False
        logger.debug("Pipeline cleared")
        return self
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> 'PreprocessingPipeline':
        """
        Initialize the pipeline and all steps.
        
        Propagates common configuration to all steps and initializes them.
        
        Args:
            config: Common configuration (e.g., sampling_rate)
                This is merged with step-specific configs.
        
        Returns:
            Self for method chaining
        
        Example:
            >>> pipeline.initialize({
            ...     'sampling_rate': 250,
            ...     'verbose': True
            ... })
        """
        logger.info(f"Initializing pipeline with {len(self._steps)} steps")
        
        # Store common config
        self._common_config = config or {}
        
        # Initialize each step
        for name, preprocessor, step_config in self._steps:
            # Merge common config with step-specific config
            merged_config = {**self._common_config, **step_config}
            
            try:
                preprocessor.initialize(merged_config)
                logger.debug(f"Initialized step '{name}'")
            except Exception as e:
                logger.error(f"Failed to initialize step '{name}': {e}")
                raise
        
        self._is_initialized = True
        logger.info("Pipeline initialization complete")
        
        return self
    
    # =========================================================================
    # PROCESSING
    # =========================================================================
    
    def process(
        self,
        data: Union[np.ndarray, EEGData],
        **kwargs
    ) -> Union[np.ndarray, EEGData]:
        """
        Process data through the entire pipeline.
        
        Executes all steps in sequence, passing the output of each
        step as input to the next.
        
        Args:
            data: Input data to process
                - numpy array: Shape (channels, samples) or (trials, channels, samples)
                - EEGData: EEG data container
            **kwargs: Additional arguments passed to each step
        
        Returns:
            Processed data in the same format as input
        
        Raises:
            RuntimeError: If pipeline not initialized
        
        Example:
            >>> processed = pipeline.process(raw_eeg)
            >>> print(f"Processed shape: {processed.shape}")
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Pipeline not initialized. Call initialize() first."
            )
        
        if len(self._steps) == 0:
            logger.warning("Pipeline has no steps, returning input unchanged")
            return data
        
        # Reset timing
        self._execution_times.clear()
        
        # Process through each step
        current_data = data
        
        for name, preprocessor, _ in self._steps:
            if self._verbose:
                logger.info(f"Executing step: {name}")
            
            start_time = time.time()
            
            try:
                current_data = preprocessor.process(current_data, **kwargs)
            except Exception as e:
                logger.error(f"Step '{name}' failed: {e}")
                raise RuntimeError(f"Pipeline step '{name}' failed: {e}") from e
            
            elapsed = time.time() - start_time
            self._execution_times[name] = elapsed
            
            if self._timing:
                logger.info(f"Step '{name}' completed in {elapsed:.3f}s")
        
        return current_data
    
    def process_step(
        self,
        data: Union[np.ndarray, EEGData],
        step_name: str,
        **kwargs
    ) -> Union[np.ndarray, EEGData]:
        """
        Process data through a single step.
        
        Useful for debugging or applying specific steps.
        
        Args:
            data: Input data
            step_name: Name of the step to execute
            **kwargs: Additional arguments
        
        Returns:
            Processed data
        
        Raises:
            ValueError: If step not found
        """
        for name, preprocessor, _ in self._steps:
            if name == step_name:
                return preprocessor.process(data, **kwargs)
        
        raise ValueError(f"Step not found: {step_name}")
    
    def process_up_to(
        self,
        data: Union[np.ndarray, EEGData],
        step_name: str,
        **kwargs
    ) -> Union[np.ndarray, EEGData]:
        """
        Process data up to (and including) a specific step.
        
        Useful for debugging intermediate results.
        
        Args:
            data: Input data
            step_name: Name of the step to stop at
            **kwargs: Additional arguments
        
        Returns:
            Processed data after the specified step
        
        Raises:
            ValueError: If step not found
        """
        current_data = data
        
        for name, preprocessor, _ in self._steps:
            current_data = preprocessor.process(current_data, **kwargs)
            if name == step_name:
                return current_data
        
        raise ValueError(f"Step not found: {step_name}")
    
    # =========================================================================
    # INSPECTION
    # =========================================================================
    
    def get_steps(self) -> List[str]:
        """
        Get list of step names in execution order.
        
        Returns:
            List of step names
        """
        return [name for name, _, _ in self._steps]
    
    def get_step(self, name: str) -> Tuple[IPreprocessor, Dict[str, Any]]:
        """
        Get a specific step by name.
        
        Args:
            name: Step name
        
        Returns:
            Tuple of (preprocessor, config)
        
        Raises:
            ValueError: If step not found
        """
        for step_name, preprocessor, config in self._steps:
            if step_name == name:
                return preprocessor, config
        
        raise ValueError(f"Step not found: {name}")
    
    def get_step_params(self, name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific step.
        
        Args:
            name: Step name
        
        Returns:
            Dict of step parameters
        """
        preprocessor, _ = self.get_step(name)
        return preprocessor.get_params()
    
    def get_execution_times(self) -> Dict[str, float]:
        """
        Get execution time for each step (from last process() call).
        
        Returns:
            Dict mapping step name to execution time in seconds
        """
        return self._execution_times.copy()
    
    def get_total_time(self) -> float:
        """
        Get total execution time from last process() call.
        
        Returns:
            Total time in seconds
        """
        return sum(self._execution_times.values())
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the full pipeline configuration.
        
        Returns:
            Dict containing common config and all step configs
        """
        return {
            'common': self._common_config.copy(),
            'steps': [
                {
                    'name': name,
                    'type': preprocessor.name,
                    'config': config.copy()
                }
                for name, preprocessor, config in self._steps
            ]
        }
    
    def summary(self) -> str:
        """
        Get a human-readable summary of the pipeline.
        
        Returns:
            Summary string
        """
        lines = [
            "Preprocessing Pipeline",
            "=" * 40,
            f"Steps: {len(self._steps)}",
            f"Initialized: {self._is_initialized}",
            ""
        ]
        
        for i, (name, preprocessor, config) in enumerate(self._steps):
            lines.append(f"  {i+1}. {name} ({preprocessor.name})")
            if config:
                for k, v in list(config.items())[:3]:
                    lines.append(f"      {k}: {v}")
        
        if self._execution_times:
            lines.append("")
            lines.append("Last Execution:")
            for name, elapsed in self._execution_times.items():
                lines.append(f"  {name}: {elapsed:.3f}s")
            lines.append(f"  Total: {self.get_total_time():.3f}s")
        
        return "\n".join(lines)
    
    # =========================================================================
    # SPECIAL METHODS
    # =========================================================================
    
    def __len__(self) -> int:
        """Number of steps in the pipeline."""
        return len(self._steps)
    
    def __iter__(self):
        """Iterate over steps."""
        for name, preprocessor, config in self._steps:
            yield name, preprocessor, config
    
    def __getitem__(self, key: Union[int, str]) -> Tuple[str, IPreprocessor, Dict]:
        """Get step by index or name."""
        if isinstance(key, int):
            return self._steps[key]
        else:
            for step in self._steps:
                if step[0] == key:
                    return step
            raise KeyError(f"Step not found: {key}")
    
    def __repr__(self) -> str:
        """String representation."""
        steps = " -> ".join(name for name, _, _ in self._steps) if self._steps else "empty"
        status = "initialized" if self._is_initialized else "not initialized"
        return f"PreprocessingPipeline({steps}) [{status}]"
    
    def __call__(
        self,
        data: Union[np.ndarray, EEGData],
        **kwargs
    ) -> Union[np.ndarray, EEGData]:
        """Allow using pipeline as callable."""
        return self.process(data, **kwargs)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_standard_pipeline(
    sampling_rate: float = 250.0,
    notch_freq: float = 50.0,
    bandpass_low: float = 8.0,
    bandpass_high: float = 30.0,
    normalize: bool = True,
    normalization_method: str = 'zscore'
) -> PreprocessingPipeline:
    """
    Create a standard BCI preprocessing pipeline.
    
    This creates a typical pipeline for motor imagery BCI:
    1. Notch filter (remove line noise)
    2. Bandpass filter (isolate motor imagery frequencies)
    3. Normalization (optional)
    
    Args:
        sampling_rate: Signal sampling rate in Hz
        notch_freq: Line noise frequency (50 Hz EU, 60 Hz US)
        bandpass_low: Lower cutoff for bandpass
        bandpass_high: Upper cutoff for bandpass
        normalize: Whether to add normalization step
        normalization_method: Normalization method
    
    Returns:
        Initialized PreprocessingPipeline
    
    Example:
        >>> pipeline = create_standard_pipeline(sampling_rate=250)
        >>> processed = pipeline.process(raw_eeg)
    """
    from src.preprocessing.steps import BandpassFilter, NotchFilter, Normalization
    
    pipeline = PreprocessingPipeline(verbose=False, timing=True)
    
    # Add notch filter
    pipeline.add_step(
        NotchFilter(),
        {'notch_freq': notch_freq, 'quality_factor': 30},
        name='notch'
    )
    
    # Add bandpass filter
    pipeline.add_step(
        BandpassFilter(),
        {'low_freq': bandpass_low, 'high_freq': bandpass_high},
        name='bandpass'
    )
    
    # Add normalization (optional)
    if normalize:
        pipeline.add_step(
            Normalization(),
            {'method': normalization_method, 'axis': 'channel'},
            name='normalize'
        )
    
    # Initialize with common config
    pipeline.initialize({'sampling_rate': sampling_rate})
    
    return pipeline


def create_pipeline_from_config(config: Dict[str, Any]) -> PreprocessingPipeline:
    """
    Create a pipeline from a configuration dictionary.
    
    Args:
        config: Configuration with structure:
            {
                'common': {'sampling_rate': 250, ...},
                'steps': [
                    {'type': 'notch_filter', 'config': {...}},
                    {'type': 'bandpass_filter', 'config': {...}},
                    ...
                ]
            }
    
    Returns:
        Initialized PreprocessingPipeline
    """
    from src.preprocessing.steps import (
        BandpassFilter, NotchFilter, Normalization
    )
    
    # Map type names to classes
    step_classes = {
        'notch_filter': NotchFilter,
        'notch': NotchFilter,
        'bandpass_filter': BandpassFilter,
        'bandpass': BandpassFilter,
        'normalization': Normalization,
        'normalize': Normalization,
    }
    
    pipeline = PreprocessingPipeline()
    
    # Add steps
    for step_config in config.get('steps', []):
        step_type = step_config.get('type', '')
        step_name = step_config.get('name', step_type)
        step_params = step_config.get('config', {})
        
        if step_type not in step_classes:
            raise ValueError(f"Unknown step type: {step_type}")
        
        preprocessor = step_classes[step_type]()
        pipeline.add_step(preprocessor, step_params, name=step_name)
    
    # Initialize with common config
    pipeline.initialize(config.get('common', {}))
    
    return pipeline
