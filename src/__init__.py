"""
LLM-EEG Framework
=================

A modular Brain-Computer Interface framework with LLM integration and AI Agents
for motor imagery EEG classification.

Repository: https://github.com/erlika/llm-eeg

Features:
---------
- Modular plugin architecture
- Support for BCI Competition IV-2a dataset
- Adaptive Preprocessing Agent (APA) with RL-based policy
- Decision Validation Agent (DVA) with 0.8 confidence threshold
- LLM integration (Phi-3) for explanations
- Cross-trial learning
- Google Colab deployment ready

Author: EEG-BCI Framework
Date: 2024
"""

# Version
__version__ = '1.0.0'

# Note: To avoid circular import issues in Google Colab,
# imports are done explicitly when needed rather than at module level.
# 
# Usage in Google Colab:
# ----------------------
# from src.core import get_config, EEGData, EventMarker
# from src.utils import setup_logging

__all__ = [
    '__version__',
]
