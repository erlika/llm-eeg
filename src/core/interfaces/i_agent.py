"""
IAgent Interface
================

This module defines the abstract interface for all AI agents in the EEG-BCI framework.

Agents are responsible for:
- Making intelligent decisions based on observed states
- Learning from feedback (rewards/outcomes)
- Providing human-readable explanations for their decisions
- Supporting cross-trial learning for continuous improvement

Agent Types in Framework:
------------------------
1. Adaptive Preprocessing Agent (APA):
   - Observes signal quality metrics (SNR, artifact ratio, line noise)
   - Selects optimal preprocessing parameters for each trial
   - Uses RL-based learned policy (Q-learning approved by user)
   - Improves across trials within and across sessions

2. Decision Validation Agent (DVA):
   - Validates classification decisions using multi-criteria analysis
   - Confidence threshold: 0.8 (approved by user)
   - Supports adaptive threshold adjustment via cross-trial learning
   - Provides accept/reject/review recommendations

Design Principles:
-----------------
- All agents implement the same interface for consistency
- Agents are stateful (maintain learning across trials)
- Agents support serialization for persistence
- Agents can integrate with LLM for explanations
- Supports both rule-based and learned policies

Cross-Trial Learning (Approved by User):
---------------------------------------
- Agents maintain state across trials within a session
- Learning can be transferred across subjects (optional)
- State can be saved/loaded for continuity

Example Usage:
    ```python
    # Create and initialize agent
    apa = AdaptivePreprocessingAgent()
    apa.initialize(config)
    
    # Process trials
    for trial in trials:
        state = apa.observe(trial)  # Observe signal quality
        action = apa.act(state)      # Select preprocessing params
        processed = apply_preprocessing(trial, action)
        reward = compute_reward(processed)
        apa.learn(state, action, reward, next_state)
    
    # Get explanation
    explanation = apa.explain(state, action)
    
    # Save state for next session
    apa.save_state("apa_checkpoint.pkl")
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from pathlib import Path

# Forward references
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.types.eeg_data import EEGData
    from src.core.interfaces.i_policy import IPolicy


class IAgent(ABC):
    """
    Abstract interface for AI agents in the BCI framework.
    
    All agent implementations (APA, DVA, etc.) must inherit from this class.
    This ensures consistent behavior and enables the agent framework.
    
    Attributes:
        name (str): Unique identifier for this agent
        is_trained (bool): Whether agent has been trained/initialized
        episode_count (int): Number of episodes/trials processed
    
    Agent Lifecycle:
        1. initialize() - Set up agent with configuration
        2. observe() - Observe the current state
        3. act() - Select an action based on state
        4. learn() - Update policy based on feedback
        5. save_state()/load_state() - Persist learning
    
    Learning Modes:
        - Online: Learn during operation (cross-trial learning)
        - Offline: Learn from historical data
        - Hybrid: Pre-train offline, fine-tune online
    """
    
    # =========================================================================
    # ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this agent.
        
        Returns:
            str: Agent name (e.g., "apa", "dva", "custom_agent")
        
        Example:
            >>> agent.name
            'apa'
        """
        pass
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """
        Check if agent has been trained or is ready to operate.
        
        Returns:
            bool: True if agent is ready to make decisions
        
        Note:
            For RL-based agents, may be True after initialization
            even without explicit training (exploration mode).
        """
        pass
    
    @property
    @abstractmethod
    def episode_count(self) -> int:
        """
        Number of episodes/trials the agent has processed.
        
        Returns:
            int: Total count of processed episodes
            
        Note:
            Used to track learning progress and decay exploration.
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS - Core Agent Loop
    # =========================================================================
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the agent with configuration settings.
        
        Sets up the agent's policy, learning parameters, and initial state.
        Must be called before any other methods.
        
        Args:
            config: Dictionary containing agent-specific settings
                Common keys:
                - 'policy': Policy configuration (RL or rule-based)
                - 'learning_rate': Learning rate for policy updates
                - 'discount_factor': Gamma for RL (default: 0.99)
                - 'exploration_rate': Initial exploration epsilon
                - 'exploration_decay': Epsilon decay rate
                - 'min_exploration': Minimum epsilon value
                
                For APA:
                - 'state_bins': Binning configuration for state space
                - 'action_space': Available preprocessing parameter sets
                - 'reward_function': Reward function configuration
                
                For DVA:
                - 'confidence_threshold': 0.8 (user-approved)
                - 'validators': List of validation criteria
                - 'adaptive_threshold': Whether to adjust threshold
        
        Raises:
            ValueError: If required configuration is missing
            
        Example:
            >>> apa.initialize({
            ...     'policy': {'type': 'q_learning', 'learning_rate': 0.1},
            ...     'state_bins': {'snr': [0, 5, 10, 20], 'artifact_ratio': [0, 0.1, 0.3]},
            ...     'action_space': ['conservative', 'moderate', 'aggressive']
            ... })
        """
        pass
    
    @abstractmethod
    def observe(self, 
                data: Union[np.ndarray, 'EEGData', Dict[str, Any]]) -> Dict[str, Any]:
        """
        Observe the current environment state.
        
        Extracts relevant features from the input to form the state representation.
        
        Args:
            data: Input data to observe
                - numpy array: Raw signal data
                - EEGData: Standardized EEG data object
                - Dict: Pre-computed state features
        
        Returns:
            Dict containing state representation:
                For APA:
                - 'snr': Signal-to-noise ratio
                - 'artifact_ratio': Ratio of artifact-contaminated samples
                - 'line_noise': 50/60 Hz power
                - 'signal_quality_score': Overall quality metric
                
                For DVA:
                - 'confidence': Classification confidence
                - 'margin': Prediction margin
                - 'signal_quality': Quality metrics
                - 'historical_consistency': Consistency with past predictions
        
        Example:
            >>> state = apa.observe(trial_data)
            >>> print(state)
            {'snr': 12.5, 'artifact_ratio': 0.15, 'line_noise': 0.8}
        """
        pass
    
    @abstractmethod
    def act(self, state: Dict[str, Any]) -> Any:
        """
        Select an action based on the observed state.
        
        Uses the agent's policy (RL or rule-based) to choose an action.
        
        Args:
            state: Current state observation from observe()
        
        Returns:
            Selected action (type depends on agent):
                For APA: Dict with preprocessing parameters
                    {'bandpass': (8, 30), 'notch': 50, 'artifact_threshold': 100}
                For DVA: Decision string
                    'accept', 'reject', or 'review'
        
        Example:
            >>> action = apa.act(state)
            >>> print(action)
            {'bandpass': (8, 30), 'notch': 50, 'artifact_threshold': 100}
        """
        pass
    
    @abstractmethod
    def learn(self,
              state: Dict[str, Any],
              action: Any,
              reward: float,
              next_state: Optional[Dict[str, Any]] = None,
              done: bool = False,
              **kwargs) -> Dict[str, Any]:
        """
        Update the agent's policy based on feedback.
        
        Implements the learning mechanism (Q-learning update for RL agents,
        statistics update for rule-based agents).
        
        Args:
            state: State before action
            action: Action that was taken
            reward: Reward received after action
            next_state: State after action (for TD learning)
            done: Whether episode/trial is complete
            **kwargs: Additional learning context
                - 'classification_correct': Whether final classification was correct
                - 'signal_quality_improvement': SNR improvement metric
        
        Returns:
            Dict containing learning metrics:
                - 'td_error': Temporal difference error (if RL)
                - 'q_value_update': Change in Q-value
                - 'exploration_rate': Current epsilon
        
        Note:
            This method implements cross-trial learning as approved by user.
            Agent improves within session and can transfer learning across subjects.
        
        Example:
            >>> metrics = apa.learn(state, action, reward=0.8, next_state=new_state)
            >>> print(f"TD Error: {metrics['td_error']:.4f}")
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS - Explanation and Interpretability
    # =========================================================================
    
    @abstractmethod
    def explain(self, 
                state: Dict[str, Any],
                action: Any,
                **kwargs) -> str:
        """
        Generate human-readable explanation for a decision.
        
        Provides interpretability by explaining why an action was chosen.
        Can integrate with LLM for natural language explanations.
        
        Args:
            state: State when decision was made
            action: Action that was chosen
            **kwargs: Additional context for explanation
                - 'use_llm': Whether to use LLM for explanation
                - 'detail_level': 'brief', 'detailed', or 'technical'
        
        Returns:
            str: Human-readable explanation
        
        Example:
            >>> explanation = apa.explain(state, action)
            >>> print(explanation)
            "Selected aggressive preprocessing because SNR (5.2) is below
             threshold and artifact ratio (0.35) indicates significant noise.
             Bandpass filter set to 8-30 Hz to focus on motor imagery bands."
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS - State Persistence
    # =========================================================================
    
    @abstractmethod
    def save_state(self, path: Union[str, Path]) -> None:
        """
        Save agent state to disk for persistence.
        
        Saves:
        - Policy parameters (Q-table, neural network weights)
        - Learning statistics
        - Episode count
        - Configuration
        
        Args:
            path: File path to save state
        
        Example:
            >>> agent.save_state("checkpoints/apa_subject01_session1.pkl")
        """
        pass
    
    @abstractmethod
    def load_state(self, path: Union[str, Path]) -> 'IAgent':
        """
        Load agent state from disk.
        
        Restores agent to a previous state for continued learning
        or deployment.
        
        Args:
            path: File path to load state from
        
        Returns:
            Self for method chaining
        
        Example:
            >>> agent.load_state("checkpoints/apa_pretrained.pkl")
            >>> # Continue learning with loaded state
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get current agent parameters.
        
        Returns:
            Dict containing all agent parameters and hyperparameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'IAgent':
        """
        Set agent parameters.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS - Override as needed
    # =========================================================================
    
    def reset(self) -> None:
        """
        Reset agent state for new episode/session.
        
        Resets internal state while preserving learned policy.
        Override for custom reset behavior.
        """
        pass
    
    def get_policy(self) -> Optional['IPolicy']:
        """
        Get the agent's policy object.
        
        Returns:
            IPolicy: Policy object if using RL, None if rule-based
        """
        return getattr(self, '_policy', None)
    
    def set_policy(self, policy: 'IPolicy') -> 'IAgent':
        """
        Set the agent's policy.
        
        Allows swapping policies without re-initializing the agent.
        
        Args:
            policy: New policy object
        
        Returns:
            Self for method chaining
        """
        self._policy = policy
        return self
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics.
        
        Returns:
            Dict containing:
                - 'total_episodes': Number of episodes processed
                - 'total_reward': Cumulative reward
                - 'average_reward': Mean reward per episode
                - 'exploration_rate': Current epsilon
                - 'learning_curve': List of episode rewards
        """
        return {
            'total_episodes': self.episode_count,
            'exploration_rate': getattr(self, '_epsilon', None)
        }
    
    def enable_learning(self) -> 'IAgent':
        """Enable online learning (cross-trial learning)."""
        self._learning_enabled = True
        return self
    
    def disable_learning(self) -> 'IAgent':
        """Disable online learning (evaluation mode)."""
        self._learning_enabled = False
        return self
    
    def is_learning_enabled(self) -> bool:
        """Check if online learning is enabled."""
        return getattr(self, '_learning_enabled', True)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def validate_state(self, state: Dict[str, Any]) -> None:
        """
        Validate state dictionary format.
        
        Args:
            state: State dictionary to validate
        
        Raises:
            ValueError: If state format is invalid
        """
        if not isinstance(state, dict):
            raise ValueError(f"State must be a dictionary, got {type(state)}")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"episodes={self.episode_count}, "
            f"trained={self.is_trained})"
        )
