"""
IReward Interface
=================

This module defines the abstract interface for reward functions in the EEG-BCI framework.

Reward functions are responsible for:
- Computing scalar rewards from state/action transitions
- Guiding agent learning toward desired behaviors
- Supporting multi-objective optimization through weighted components

Reward Design for APA (Adaptive Preprocessing Agent):
-----------------------------------------------------
The APA optimizes preprocessing to maximize classification performance.
Reward components:
1. Signal Quality Improvement: SNR after vs before preprocessing
2. Feature Discriminability: CSP feature separation between classes
3. Classification Accuracy: Final classifier performance
4. Computational Efficiency: Processing time (optional)

Reward Design for DVA (Decision Validation Agent):
-------------------------------------------------
The DVA validates classifications to maximize system reliability.
Reward components:
1. Correct Accept: +1.0 (accepted correct classification)
2. Correct Reject: +0.8 (rejected incorrect classification)
3. Incorrect Accept: -1.0 (accepted wrong classification)
4. Incorrect Reject: -0.5 (rejected correct classification)

Design Principles:
-----------------
- Rewards are composable from multiple components
- Weights are configurable for different priorities
- Support both dense (per-step) and sparse (episodic) rewards
- Enable reward shaping for faster learning

Example Usage:
    ```python
    # Create composite reward function
    reward_fn = CompositeReward([
        SNRImprovementReward(weight=0.3),
        DiscriminabilityReward(weight=0.3),
        ClassificationReward(weight=0.4)
    ])
    
    # Compute reward
    reward = reward_fn.compute(
        state={'snr': 5.0, 'artifact_ratio': 0.3},
        action='aggressive',
        next_state={'snr': 12.0, 'artifact_ratio': 0.05},
        context={'correct_classification': True}
    )
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np


class IReward(ABC):
    """
    Abstract interface for reward functions.
    
    All reward function implementations must inherit from this class.
    Supports both simple and composite reward structures.
    
    Attributes:
        name (str): Unique identifier for this reward function
        weight (float): Weight when used in composite rewards
    
    Reward Types:
        - Dense: Computed every step (immediate feedback)
        - Sparse: Only at episode end (delayed feedback)
        - Shaped: Intermediate rewards guiding toward goal
    """
    
    # =========================================================================
    # ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this reward function.
        
        Returns:
            str: Reward name (e.g., "snr_improvement", "classification_accuracy")
        
        Example:
            >>> reward_fn.name
            'snr_improvement'
        """
        pass
    
    @property
    def weight(self) -> float:
        """
        Weight when used in composite rewards.
        
        Returns:
            float: Weight value (default: 1.0)
        """
        return getattr(self, '_weight', 1.0)
    
    @weight.setter
    def weight(self, value: float) -> None:
        """Set the weight value."""
        self._weight = value
    
    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the reward function with configuration.
        
        Args:
            config: Dictionary containing reward-specific settings
                Common keys:
                - 'weight': Weight in composite rewards
                - 'scale': Reward scaling factor
                - 'clip_min': Minimum reward value
                - 'clip_max': Maximum reward value
                
                For SNR Improvement:
                - 'baseline_snr': Reference SNR value
                - 'improvement_scale': Scaling for improvement
                
                For Classification:
                - 'correct_reward': Reward for correct classification
                - 'incorrect_penalty': Penalty for incorrect classification
        
        Example:
            >>> snr_reward.initialize({
            ...     'weight': 0.3,
            ...     'baseline_snr': 10.0,
            ...     'improvement_scale': 0.1
            ... })
        """
        pass
    
    @abstractmethod
    def compute(self,
                state: Dict[str, Any],
                action: Any,
                next_state: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute the reward for a state-action transition.
        
        Args:
            state: State before action
            action: Action that was taken
            next_state: State after action
            context: Additional context for reward computation
                - 'correct_classification': Whether classification is correct
                - 'classification_confidence': Confidence score
                - 'processing_time': Time taken for preprocessing
        
        Returns:
            float: Computed reward value
        
        Example:
            >>> reward = reward_fn.compute(
            ...     state={'snr': 5.0},
            ...     action='aggressive',
            ...     next_state={'snr': 12.0},
            ...     context={'correct_classification': True}
            ... )
            >>> print(f"Reward: {reward:.2f}")
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get current reward function parameters.
        
        Returns:
            Dict containing all parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'IReward':
        """
        Set reward function parameters.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS
    # =========================================================================
    
    def get_components(self) -> Dict[str, float]:
        """
        Get individual reward components (for composite rewards).
        
        Returns:
            Dict mapping component names to their values
        """
        return {self.name: getattr(self, '_last_reward', 0.0)}
    
    def get_explanation(self,
                        state: Dict[str, Any],
                        action: Any,
                        next_state: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get explanation for computed reward.
        
        Args:
            state: State before action
            action: Action taken
            next_state: State after action
            context: Additional context
        
        Returns:
            str: Human-readable explanation
        """
        reward = self.compute(state, action, next_state, context)
        return f"{self.name}: {reward:.3f}"
    
    def reset(self) -> None:
        """Reset any internal state (e.g., running averages)."""
        pass
    
    def __call__(self,
                 state: Dict[str, Any],
                 action: Any,
                 next_state: Dict[str, Any],
                 context: Optional[Dict[str, Any]] = None) -> float:
        """
        Allow using reward function as callable.
        
        Equivalent to compute().
        """
        return self.compute(state, action, next_state, context)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight})"


class ICompositeReward(IReward):
    """
    Abstract interface for composite reward functions.
    
    Combines multiple reward components with configurable weights.
    
    Example:
        ```python
        composite = CompositeReward([
            SNRReward(weight=0.3),
            DiscriminabilityReward(weight=0.3),
            ClassificationReward(weight=0.4)
        ])
        ```
    """
    
    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    @abstractmethod
    def add_component(self, reward: 'IReward') -> 'ICompositeReward':
        """
        Add a reward component.
        
        Args:
            reward: Reward function to add
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def remove_component(self, name: str) -> 'ICompositeReward':
        """
        Remove a reward component by name.
        
        Args:
            name: Name of component to remove
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def get_component(self, name: str) -> Optional['IReward']:
        """
        Get a component by name.
        
        Args:
            name: Component name
        
        Returns:
            Reward component or None if not found
        """
        pass
    
    @abstractmethod
    def list_components(self) -> List[str]:
        """
        List all component names.
        
        Returns:
            List of component names
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS
    # =========================================================================
    
    def normalize_weights(self) -> 'ICompositeReward':
        """
        Normalize component weights to sum to 1.0.
        
        Returns:
            Self for method chaining
        """
        pass
    
    def get_component_contributions(self,
                                     state: Dict[str, Any],
                                     action: Any,
                                     next_state: Dict[str, Any],
                                     context: Optional[Dict[str, Any]] = None
                                     ) -> Dict[str, float]:
        """
        Get individual component contributions to total reward.
        
        Args:
            state: State before action
            action: Action taken
            next_state: State after action
            context: Additional context
        
        Returns:
            Dict mapping component names to their weighted contributions
        """
        pass


# =============================================================================
# REWARD FUNCTION TYPES (for reference)
# =============================================================================

class RewardType:
    """Enumeration of common reward function types."""
    
    # Signal Quality Rewards
    SNR_IMPROVEMENT = "snr_improvement"
    ARTIFACT_REDUCTION = "artifact_reduction"
    LINE_NOISE_REDUCTION = "line_noise_reduction"
    
    # Feature Quality Rewards
    DISCRIMINABILITY = "discriminability"
    FEATURE_STABILITY = "feature_stability"
    
    # Classification Rewards
    ACCURACY = "classification_accuracy"
    CONFIDENCE = "classification_confidence"
    CALIBRATION = "prediction_calibration"
    
    # Efficiency Rewards
    PROCESSING_TIME = "processing_time"
    COMPUTATIONAL_COST = "computational_cost"
    
    # DVA-specific Rewards
    CORRECT_ACCEPT = "correct_accept"
    CORRECT_REJECT = "correct_reject"
    INCORRECT_ACCEPT = "incorrect_accept"
    INCORRECT_REJECT = "incorrect_reject"


# =============================================================================
# REWARD CONFIGURATION TEMPLATES
# =============================================================================

APA_REWARD_CONFIG = {
    """
    Default reward configuration for Adaptive Preprocessing Agent.
    
    Components:
    - snr_improvement: Rewards better signal quality
    - artifact_reduction: Rewards lower artifact ratio
    - discriminability: Rewards better feature separation
    - classification_accuracy: Rewards correct final classification
    """
    'components': [
        {
            'type': RewardType.SNR_IMPROVEMENT,
            'weight': 0.25,
            'params': {
                'baseline_snr': 10.0,
                'scale': 0.1,
                'clip_range': (-1.0, 1.0)
            }
        },
        {
            'type': RewardType.ARTIFACT_REDUCTION,
            'weight': 0.25,
            'params': {
                'target_ratio': 0.05,
                'scale': 2.0
            }
        },
        {
            'type': RewardType.DISCRIMINABILITY,
            'weight': 0.2,
            'params': {
                'method': 'fisher_ratio'
            }
        },
        {
            'type': RewardType.ACCURACY,
            'weight': 0.3,
            'params': {
                'correct_reward': 1.0,
                'incorrect_penalty': -0.5
            }
        }
    ],
    'normalize_weights': True
}

DVA_REWARD_CONFIG = {
    """
    Default reward configuration for Decision Validation Agent.
    
    Uses accept/reject outcomes with classification correctness.
    """
    'components': [
        {
            'type': RewardType.CORRECT_ACCEPT,
            'params': {'reward': 1.0}
        },
        {
            'type': RewardType.CORRECT_REJECT,
            'params': {'reward': 0.8}
        },
        {
            'type': RewardType.INCORRECT_ACCEPT,
            'params': {'reward': -1.0}
        },
        {
            'type': RewardType.INCORRECT_REJECT,
            'params': {'reward': -0.5}
        }
    ]
}
