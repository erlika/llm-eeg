"""
IPolicy Interface
=================

This module defines the abstract interface for all RL policies in the EEG-BCI framework.

Policies are responsible for:
- Mapping states to actions (the decision-making core)
- Balancing exploration vs exploitation
- Updating based on learning signals
- Supporting various RL algorithms

Policy Types Supported:
----------------------
1. Tabular Q-Learning (User-approved for APA):
   - Discrete state/action spaces
   - Q-table for value storage
   - Suitable for small state spaces

2. DQN (Deep Q-Network):
   - Neural network function approximation
   - Suitable for large/continuous state spaces
   - Experience replay support

3. Policy Gradient:
   - Direct policy parameterization
   - Suitable for continuous action spaces

4. Actor-Critic:
   - Combined value and policy learning
   - More stable than pure policy gradient

5. Rule-Based (Deterministic):
   - Expert-defined rules
   - No learning, deterministic mapping

Design Principles:
-----------------
- Policies are modular and swappable
- Support both discrete and continuous action spaces
- Provide exploration mechanisms
- Enable state discretization for tabular methods

State Discretization (for Q-Learning):
-------------------------------------
User-approved state bins for APA:
- snr: [0, 5, 10, 20, inf] -> 4 bins
- artifact_ratio: [0, 0.1, 0.3, 0.5, 1.0] -> 4 bins
- line_noise: [0, 0.5, 1.0, 2.0, inf] -> 4 bins

Total state space: 4 x 4 x 4 = 64 states

Action Space for APA:
- 'conservative': Minimal filtering, preserve signal
- 'moderate': Standard preprocessing
- 'aggressive': Heavy filtering for noisy signals

Example Usage:
    ```python
    # Create and initialize policy
    policy = TabularQLearningPolicy()
    policy.initialize({
        'state_space_size': 64,
        'action_space_size': 3,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01
    })
    
    # Select action
    action = policy.select_action(state_index, explore=True)
    
    # Update policy
    td_error = policy.update(state, action, reward, next_state, done)
    
    # Get Q-values for visualization
    q_values = policy.get_q_values(state_index)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from pathlib import Path


class IPolicy(ABC):
    """
    Abstract interface for RL policies.
    
    All policy implementations must inherit from this class.
    This enables pluggable policies for different agents and use cases.
    
    Attributes:
        name (str): Unique identifier for this policy type
        state_space_size (int): Size of discrete state space (for tabular)
        action_space_size (int): Number of available actions
        is_discrete (bool): Whether policy uses discrete state/action spaces
    
    Policy Modes:
        - Training: Exploration enabled, learning updates active
        - Evaluation: Exploration disabled, greedy action selection
    """
    
    # =========================================================================
    # ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this policy type.
        
        Returns:
            str: Policy name (e.g., "q_learning", "dqn", "policy_gradient")
        
        Example:
            >>> policy.name
            'q_learning'
        """
        pass
    
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """
        Number of available discrete actions.
        
        Returns:
            int: Size of action space
            
        Example:
            >>> policy.action_space_size
            3  # conservative, moderate, aggressive
        """
        pass
    
    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """
        Whether policy operates on discrete state/action spaces.
        
        Returns:
            bool: True for tabular methods, False for function approximation
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS - Core Policy Functions
    # =========================================================================
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the policy with configuration.
        
        Sets up internal structures (Q-table, neural network, etc.).
        
        Args:
            config: Policy configuration
                For Q-Learning:
                - 'state_space_size': Number of discrete states
                - 'action_space_size': Number of actions
                - 'learning_rate': Alpha for Q-update (default: 0.1)
                - 'discount_factor': Gamma for future rewards (default: 0.99)
                - 'epsilon_start': Initial exploration rate (default: 1.0)
                - 'epsilon_decay': Decay rate per episode (default: 0.995)
                - 'epsilon_min': Minimum exploration rate (default: 0.01)
                - 'initial_q_value': Initial Q-values (default: 0.0)
                
                For DQN:
                - 'state_dim': State vector dimension
                - 'hidden_layers': List of hidden layer sizes
                - 'batch_size': Training batch size
                - 'replay_buffer_size': Experience replay buffer size
        
        Raises:
            ValueError: If required configuration is missing
            
        Example:
            >>> policy.initialize({
            ...     'state_space_size': 64,
            ...     'action_space_size': 3,
            ...     'learning_rate': 0.1,
            ...     'discount_factor': 0.99
            ... })
        """
        pass
    
    @abstractmethod
    def select_action(self,
                      state: Union[int, np.ndarray, Dict[str, Any]],
                      explore: bool = True) -> int:
        """
        Select an action given the current state.
        
        Uses epsilon-greedy (or other exploration strategy) when explore=True.
        Uses greedy selection when explore=False.
        
        Args:
            state: Current state representation
                - int: Discrete state index (for tabular)
                - np.ndarray: State feature vector (for DQN)
                - Dict: State dictionary (will be converted)
            explore: Whether to use exploration (default: True)
        
        Returns:
            int: Selected action index
        
        Example:
            >>> # Training mode with exploration
            >>> action = policy.select_action(state_idx, explore=True)
            >>> # Evaluation mode (greedy)
            >>> action = policy.select_action(state_idx, explore=False)
        """
        pass
    
    @abstractmethod
    def update(self,
               state: Union[int, np.ndarray],
               action: int,
               reward: float,
               next_state: Union[int, np.ndarray],
               done: bool,
               **kwargs) -> float:
        """
        Update policy based on experience.
        
        Implements the learning rule (Q-learning update, gradient descent, etc.).
        
        Args:
            state: State before action
            action: Action that was taken
            reward: Reward received
            next_state: State after action
            done: Whether episode is complete
            **kwargs: Additional update context
        
        Returns:
            float: TD error or loss value (for monitoring)
        
        Q-Learning Update Formula:
            Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        
        Example:
            >>> td_error = policy.update(
            ...     state=10, action=1, reward=0.8,
            ...     next_state=15, done=False
            ... )
            >>> print(f"TD Error: {td_error:.4f}")
        """
        pass
    
    @abstractmethod
    def get_q_values(self, state: Union[int, np.ndarray]) -> np.ndarray:
        """
        Get Q-values for all actions in a given state.
        
        Args:
            state: State to query
        
        Returns:
            np.ndarray: Q-values for each action, shape (action_space_size,)
        
        Example:
            >>> q_values = policy.get_q_values(state_idx)
            >>> print(q_values)
            array([0.5, 0.8, 0.3])  # Q-values for 3 actions
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get current policy parameters.
        
        Returns:
            Dict containing all policy parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'IPolicy':
        """
        Set policy parameters.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save policy to disk.
        
        Args:
            path: File path to save policy
        
        Saves:
            - Q-table or network weights
            - Exploration parameters
            - Training statistics
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> 'IPolicy':
        """
        Load policy from disk.
        
        Args:
            path: File path to load policy from
        
        Returns:
            Self for method chaining
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS - Override as needed
    # =========================================================================
    
    def decay_exploration(self) -> float:
        """
        Decay the exploration rate.
        
        Called after each episode to reduce exploration over time.
        
        Returns:
            float: New exploration rate
        """
        if hasattr(self, '_epsilon') and hasattr(self, '_epsilon_decay'):
            self._epsilon = max(
                getattr(self, '_epsilon_min', 0.01),
                self._epsilon * self._epsilon_decay
            )
            return self._epsilon
        return 0.0
    
    def get_exploration_rate(self) -> float:
        """
        Get current exploration rate.
        
        Returns:
            float: Current epsilon value
        """
        return getattr(self, '_epsilon', 0.0)
    
    def set_exploration_rate(self, epsilon: float) -> 'IPolicy':
        """
        Manually set exploration rate.
        
        Args:
            epsilon: New exploration rate (0-1)
        
        Returns:
            Self for method chaining
        """
        self._epsilon = np.clip(epsilon, 0.0, 1.0)
        return self
    
    def get_best_action(self, state: Union[int, np.ndarray]) -> int:
        """
        Get best action (greedy) for a state.
        
        Convenience method that calls select_action with explore=False.
        
        Args:
            state: Current state
        
        Returns:
            int: Best action index
        """
        return self.select_action(state, explore=False)
    
    def get_action_probabilities(self, 
                                  state: Union[int, np.ndarray],
                                  temperature: float = 1.0) -> np.ndarray:
        """
        Get softmax probabilities over actions.
        
        Useful for probabilistic action selection and analysis.
        
        Args:
            state: Current state
            temperature: Softmax temperature (higher = more uniform)
        
        Returns:
            np.ndarray: Action probabilities, shape (action_space_size,)
        """
        q_values = self.get_q_values(state)
        # Softmax with temperature
        exp_q = np.exp((q_values - np.max(q_values)) / temperature)
        return exp_q / np.sum(exp_q)
    
    # =========================================================================
    # STATE DISCRETIZATION HELPERS
    # =========================================================================
    
    def discretize_state(self,
                         state: Dict[str, float],
                         bins: Dict[str, List[float]]) -> int:
        """
        Convert continuous state to discrete index.
        
        Used by tabular methods to convert state dictionaries
        to single indices for Q-table lookup.
        
        Args:
            state: Dictionary of continuous state values
            bins: Dictionary mapping state keys to bin edges
        
        Returns:
            int: Discrete state index
        
        Example:
            >>> bins = {
            ...     'snr': [0, 5, 10, 20, float('inf')],
            ...     'artifact_ratio': [0, 0.1, 0.3, 0.5, 1.0]
            ... }
            >>> state = {'snr': 7.5, 'artifact_ratio': 0.25}
            >>> idx = policy.discretize_state(state, bins)
        """
        indices = []
        n_bins_list = []
        
        for key, bin_edges in sorted(bins.items()):
            value = state.get(key, 0)
            # Find bin index
            bin_idx = np.digitize(value, bin_edges[1:])  # Skip first edge (assumed 0)
            bin_idx = min(bin_idx, len(bin_edges) - 2)  # Clip to valid range
            indices.append(bin_idx)
            n_bins_list.append(len(bin_edges) - 1)
        
        # Convert multi-dimensional index to single index
        state_idx = 0
        multiplier = 1
        for idx, n_bins in zip(reversed(indices), reversed(n_bins_list)):
            state_idx += idx * multiplier
            multiplier *= n_bins
        
        return state_idx
    
    def get_state_from_index(self,
                              state_idx: int,
                              bins: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
        """
        Convert discrete index back to state bin ranges.
        
        Useful for debugging and visualization.
        
        Args:
            state_idx: Discrete state index
            bins: Dictionary mapping state keys to bin edges
        
        Returns:
            Dict mapping state keys to (min, max) tuples
        """
        sorted_keys = sorted(bins.keys())
        n_bins_list = [len(bins[k]) - 1 for k in sorted_keys]
        
        # Decode index
        indices = []
        remaining = state_idx
        for n_bins in reversed(n_bins_list):
            indices.append(remaining % n_bins)
            remaining //= n_bins
        indices.reverse()
        
        # Convert to ranges
        state_ranges = {}
        for key, bin_idx in zip(sorted_keys, indices):
            bin_edges = bins[key]
            state_ranges[key] = (bin_edges[bin_idx], bin_edges[bin_idx + 1])
        
        return state_ranges
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def reset(self) -> None:
        """Reset policy for new episode (clear temporary state)."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics.
        
        Returns:
            Dict containing training statistics
        """
        return {
            'exploration_rate': self.get_exploration_rate(),
            'action_space_size': self.action_space_size
        }
    
    def __repr__(self) -> str:
        """String representation of the policy."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"actions={self.action_space_size}, "
            f"epsilon={self.get_exploration_rate():.3f})"
        )
