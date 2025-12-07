"""
ILLMProvider Interface
======================

This module defines the abstract interface for LLM providers in the EEG-BCI framework.

LLM Providers are responsible for:
- Generating natural language explanations for agent decisions
- Semantic encoding of EEG features for enhanced interpretability
- Providing context-aware analysis and recommendations

Recommended LLM (User-Approved):
-------------------------------
Phi-3-mini-4k-instruct (3.8B parameters)
- Efficient for Google Colab (fits in 8GB VRAM)
- Strong instruction-following capability
- Good balance of quality and performance
- Open-source (MIT license)

Alternative LLMs (Pluggable):
----------------------------
1. Llama-3.1-8B: Higher quality, requires more resources
2. Mistral-7B: Good performance, moderate resources
3. Gemma-2B: Very efficient, suitable for real-time
4. GPT-4 API: Highest quality, requires API key

LLM Use Cases in Framework:
--------------------------
1. Agent Explanations:
   - Why APA chose specific preprocessing parameters
   - Why DVA accepted/rejected a classification
   
2. Semantic Feature Encoding:
   - Convert EEG features to semantic descriptions
   - Enable LLM-based feature analysis

3. Report Generation:
   - Summarize session results
   - Generate interpretable analysis reports

4. Error Analysis:
   - Analyze misclassification patterns
   - Suggest improvements

Design Principles:
-----------------
- Provider-agnostic interface (swap LLMs without code changes)
- Support both local and API-based models
- Configurable prompting strategies
- Caching for efficiency

Example Usage:
    ```python
    # Create and initialize provider
    llm = Phi3Provider()
    llm.initialize({
        'model_path': 'microsoft/phi-3-mini-4k-instruct',
        'device': 'cuda',
        'max_tokens': 256,
        'temperature': 0.7
    })
    
    # Generate explanation
    prompt = "Explain why aggressive preprocessing was selected..."
    response = llm.generate(prompt)
    
    # Semantic encoding
    features = {'snr': 12.5, 'alpha_power': 0.8}
    encoding = llm.encode_semantically(features)
    ```

Author: EEG-BCI Framework
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from pathlib import Path


class ILLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    
    All LLM provider implementations must inherit from this class.
    This enables pluggable LLM backends for various use cases.
    
    Attributes:
        name (str): Provider name (e.g., "phi3", "llama", "gpt4")
        model_name (str): Specific model identifier
        is_loaded (bool): Whether model is loaded and ready
    
    Provider Types:
        - Local: Model runs on local GPU (Phi-3, Llama, Mistral)
        - API: Remote API calls (OpenAI, Anthropic)
        - Hybrid: Local with API fallback
    """
    
    # =========================================================================
    # ABSTRACT PROPERTIES
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provider name identifier.
        
        Returns:
            str: Provider name (e.g., "phi3", "llama", "openai")
        
        Example:
            >>> provider.name
            'phi3'
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Specific model identifier.
        
        Returns:
            str: Full model name/path
        
        Example:
            >>> provider.model_name
            'microsoft/phi-3-mini-4k-instruct'
        """
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Check if model is loaded and ready for inference.
        
        Returns:
            bool: True if ready to generate
        """
        pass
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """
        Maximum context/token length supported.
        
        Returns:
            int: Maximum number of tokens
        
        Example:
            >>> provider.max_context_length
            4096
        """
        pass
    
    # =========================================================================
    # ABSTRACT METHODS - Core LLM Functions
    # =========================================================================
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the LLM provider.
        
        Loads the model and sets up inference parameters.
        
        Args:
            config: Provider configuration
                Common keys:
                - 'model_path': Model path or HuggingFace ID
                - 'device': 'cpu', 'cuda', or 'auto'
                - 'dtype': Data type ('float16', 'float32', 'bfloat16')
                - 'max_tokens': Maximum output tokens (default: 256)
                - 'temperature': Sampling temperature (default: 0.7)
                - 'top_p': Nucleus sampling threshold (default: 0.9)
                - 'top_k': Top-k sampling (default: 50)
                
                For API providers:
                - 'api_key': API key
                - 'api_base': API endpoint URL
                
                For local providers:
                - 'quantization': '4bit', '8bit', or None
                - 'load_in_8bit': Whether to use 8-bit quantization
                - 'trust_remote_code': Whether to trust remote code
        
        Raises:
            RuntimeError: If model cannot be loaded
            ValueError: If configuration is invalid
            
        Example:
            >>> llm.initialize({
            ...     'model_path': 'microsoft/phi-3-mini-4k-instruct',
            ...     'device': 'cuda',
            ...     'quantization': '4bit',
            ...     'max_tokens': 256
            ... })
        """
        pass
    
    @abstractmethod
    def generate(self,
                 prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            **kwargs: Additional generation parameters
                - 'top_p': Nucleus sampling
                - 'top_k': Top-k sampling
                - 'stop': Stop sequences
                - 'system_prompt': System instruction
        
        Returns:
            str: Generated text
        
        Example:
            >>> response = llm.generate(
            ...     "Explain why aggressive preprocessing was selected for SNR=5.2",
            ...     max_tokens=200,
            ...     temperature=0.7
            ... )
            >>> print(response)
        """
        pass
    
    @abstractmethod
    def generate_batch(self,
                       prompts: List[str],
                       max_tokens: Optional[int] = None,
                       **kwargs) -> List[str]:
        """
        Generate text for multiple prompts (batch processing).
        
        More efficient than calling generate() multiple times.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per response
            **kwargs: Additional generation parameters
        
        Returns:
            List[str]: Generated texts for each prompt
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, 
                       texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get text embeddings for semantic analysis.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            np.ndarray: Embedding vectors
                - Shape (embedding_dim,) for single text
                - Shape (n_texts, embedding_dim) for multiple texts
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get current provider parameters.
        
        Returns:
            Dict containing all parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'ILLMProvider':
        """
        Set provider parameters.
        
        Args:
            **params: Parameters to update
        
        Returns:
            Self for method chaining
        """
        pass
    
    # =========================================================================
    # OPTIONAL METHODS - BCI-Specific Functions
    # =========================================================================
    
    def explain_decision(self,
                         agent_name: str,
                         state: Dict[str, Any],
                         action: Any,
                         context: Optional[Dict[str, Any]] = None,
                         detail_level: str = 'detailed') -> str:
        """
        Generate explanation for an agent decision.
        
        Specialized method for BCI framework agent explanations.
        
        Args:
            agent_name: Name of agent ('apa' or 'dva')
            state: State when decision was made
            action: Action/decision taken
            context: Additional context
            detail_level: 'brief', 'detailed', or 'technical'
        
        Returns:
            str: Human-readable explanation
        
        Example:
            >>> explanation = llm.explain_decision(
            ...     agent_name='apa',
            ...     state={'snr': 5.2, 'artifact_ratio': 0.35},
            ...     action={'bandpass': (8, 30), 'notch': 50},
            ...     detail_level='detailed'
            ... )
        """
        # Default implementation using generate()
        prompt = self._build_explanation_prompt(agent_name, state, action, context, detail_level)
        return self.generate(prompt)
    
    def encode_features_semantically(self,
                                      features: Dict[str, float],
                                      feature_names: Optional[List[str]] = None) -> str:
        """
        Convert numerical features to semantic descriptions.
        
        Enables LLM-based reasoning about EEG features.
        
        Args:
            features: Dictionary of feature values
            feature_names: Optional list of feature names for context
        
        Returns:
            str: Semantic description of features
        
        Example:
            >>> description = llm.encode_features_semantically({
            ...     'alpha_power_C3': 0.85,
            ...     'alpha_power_C4': 0.72,
            ...     'mu_desync': 0.45
            ... })
            >>> print(description)
            "Strong alpha activity in left sensorimotor region (C3),
             moderate in right (C4), with significant mu desynchronization
             suggesting motor preparation."
        """
        prompt = self._build_feature_encoding_prompt(features, feature_names)
        return self.generate(prompt, max_tokens=150)
    
    def analyze_classification(self,
                                prediction: int,
                                probabilities: np.ndarray,
                                features: Optional[Dict[str, float]] = None,
                                class_names: Optional[List[str]] = None) -> str:
        """
        Analyze and explain a classification result.
        
        Args:
            prediction: Predicted class index
            probabilities: Class probability distribution
            features: Optional feature values used for classification
            class_names: Optional names for classes
        
        Returns:
            str: Analysis and explanation
        """
        class_names = class_names or [f"Class_{i}" for i in range(len(probabilities))]
        
        prompt = f"""Analyze this motor imagery classification result:

Prediction: {class_names[prediction]}
Confidence: {probabilities[prediction]:.1%}
Probability distribution: {dict(zip(class_names, probabilities.round(3)))}
"""
        if features:
            prompt += f"\nKey features: {features}"
        
        prompt += "\n\nProvide a brief analysis of this classification and its confidence."
        
        return self.generate(prompt, max_tokens=200)
    
    def generate_session_summary(self,
                                  session_stats: Dict[str, Any]) -> str:
        """
        Generate a summary report for a BCI session.
        
        Args:
            session_stats: Dictionary with session statistics
                - 'accuracy': Overall accuracy
                - 'n_trials': Number of trials
                - 'class_accuracies': Per-class accuracies
                - 'agent_stats': APA/DVA statistics
        
        Returns:
            str: Human-readable session summary
        """
        prompt = f"""Generate a concise summary of this BCI session:

Session Statistics:
{self._format_dict(session_stats)}

Provide a professional summary including:
1. Overall performance assessment
2. Key observations
3. Recommendations for improvement
"""
        return self.generate(prompt, max_tokens=300)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _build_explanation_prompt(self,
                                   agent_name: str,
                                   state: Dict[str, Any],
                                   action: Any,
                                   context: Optional[Dict[str, Any]],
                                   detail_level: str) -> str:
        """Build prompt for decision explanation."""
        if agent_name.lower() == 'apa':
            return f"""You are an expert in EEG signal processing for Brain-Computer Interfaces.

The Adaptive Preprocessing Agent (APA) made the following decision:

Signal Quality State:
{self._format_dict(state)}

Selected Preprocessing Action:
{action}

{f'Additional Context: {context}' if context else ''}

Provide a {detail_level} explanation of why this preprocessing strategy was chosen
based on the signal quality metrics. Focus on the relationship between the metrics
and the preprocessing parameters selected."""

        elif agent_name.lower() == 'dva':
            return f"""You are an expert in BCI classification validation.

The Decision Validation Agent (DVA) made the following decision:

Classification State:
{self._format_dict(state)}

Validation Decision: {action}

{f'Additional Context: {context}' if context else ''}

Provide a {detail_level} explanation of why this validation decision was made.
Consider confidence levels, consistency, and signal quality factors."""

        else:
            return f"""Explain the following agent decision:

Agent: {agent_name}
State: {state}
Action: {action}
Context: {context}

Detail level: {detail_level}"""
    
    def _build_feature_encoding_prompt(self,
                                        features: Dict[str, float],
                                        feature_names: Optional[List[str]]) -> str:
        """Build prompt for feature semantic encoding."""
        return f"""You are an expert in EEG feature interpretation for motor imagery BCIs.

Convert these numerical EEG features into a semantic description:

Features:
{self._format_dict(features)}

{f'Feature context: {feature_names}' if feature_names else ''}

Provide a concise, interpretable description of what these features indicate
about the brain state and motor imagery activity. Focus on patterns relevant
to motor imagery classification (left hand, right hand, feet, tongue)."""
    
    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format dictionary for prompt inclusion."""
        return "\n".join(f"- {k}: {v}" for k, v in d.items())
    
    def unload(self) -> None:
        """
        Unload model to free memory.
        
        Useful for switching between models or freeing GPU memory.
        """
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Dict with memory stats (GPU/CPU in GB)
        """
        return {'estimated_gb': 0.0}
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"model='{self.model_name}', "
            f"loaded={self.is_loaded})"
        )


# =============================================================================
# LLM PROVIDER TYPES (for reference and factory)
# =============================================================================

class LLMProviderType:
    """Enumeration of supported LLM providers."""
    
    # Local Models (Recommended for Colab)
    PHI3 = "phi3"           # microsoft/phi-3-mini-4k-instruct (recommended)
    PHI3_MEDIUM = "phi3_medium"  # microsoft/phi-3-medium-4k-instruct
    LLAMA3_8B = "llama3_8b"  # meta-llama/Meta-Llama-3.1-8B-Instruct
    MISTRAL_7B = "mistral_7b"  # mistralai/Mistral-7B-Instruct-v0.3
    GEMMA_2B = "gemma_2b"    # google/gemma-2b-it (lightweight)
    
    # API Models
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    
    # Mock/Testing
    MOCK = "mock"


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

PHI3_CONFIG = {
    'model_path': 'microsoft/phi-3-mini-4k-instruct',
    'device': 'cuda',
    'dtype': 'float16',
    'max_tokens': 256,
    'temperature': 0.7,
    'top_p': 0.9,
    'quantization': '4bit',  # Fits in 8GB VRAM
    'trust_remote_code': True
}

LLAMA3_CONFIG = {
    'model_path': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'device': 'cuda',
    'dtype': 'float16',
    'max_tokens': 256,
    'temperature': 0.7,
    'quantization': '4bit',  # Required for Colab
    'trust_remote_code': False
}
