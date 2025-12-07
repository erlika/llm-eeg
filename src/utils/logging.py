"""
Logging Utilities
=================

This module provides centralized logging configuration for the EEG-BCI framework.

Features:
---------
- Consistent log formatting across all modules
- File and console logging
- Log level configuration
- Colored console output (optional)
- Performance logging decorators

Log Levels:
----------
- DEBUG: Detailed information for debugging
- INFO: General operational information
- WARNING: Something unexpected but not critical
- ERROR: Something went wrong, operation may continue
- CRITICAL: Severe error, program may not continue

Example Usage:
    ```python
    from src.utils.logging import get_logger, setup_logging
    
    # Setup logging (call once at startup)
    setup_logging(level='INFO', log_file='logs/bci.log')
    
    # Get logger for your module
    logger = get_logger(__name__)
    
    # Use it
    logger.info("Processing started")
    logger.debug(f"Data shape: {data.shape}")
    logger.warning("Low signal quality detected")
    logger.error("Failed to load model")
    ```

Author: EEG-BCI Framework
Date: 2024
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from functools import wraps
import time


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
SIMPLE_FORMAT = '%(levelname)s - %(message)s'

# Color codes for console output
COLORS = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Green
    'WARNING': '\033[33m',    # Yellow
    'ERROR': '\033[31m',      # Red
    'CRITICAL': '\033[35m',   # Magenta
    'RESET': '\033[0m'        # Reset
}


# =============================================================================
# CUSTOM FORMATTER WITH COLORS
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels for console output.
    """
    
    def __init__(self, fmt: str = DEFAULT_FORMAT, use_colors: bool = True):
        super().__init__(fmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        # Save original levelname
        original_levelname = record.levelname
        
        if self.use_colors:
            color = COLORS.get(record.levelname, '')
            reset = COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return result


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(
    level: Union[str, int] = 'INFO',
    log_file: Optional[str] = None,
    console: bool = True,
    use_colors: bool = True,
    format_string: str = DEFAULT_FORMAT,
    detailed: bool = False
) -> None:
    """
    Setup logging configuration for the framework.
    
    Should be called once at application startup.
    
    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path for logging
        console: Whether to log to console
        use_colors: Whether to use colored console output
        format_string: Log format string
        detailed: If True, use detailed format with file/line info
    
    Example:
        >>> setup_logging(level='DEBUG', log_file='logs/app.log')
    """
    # Use detailed format if requested
    if detailed:
        format_string = DETAILED_FORMAT
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(format_string, use_colors))
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)
    
    # Log setup complete
    logging.info(f"Logging configured: level={logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        logging.Logger: Configured logger
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


def set_level(level: Union[str, int], logger_name: Optional[str] = None) -> None:
    """
    Set log level for a specific logger or all loggers.
    
    Args:
        level: Log level
        logger_name: If None, sets root logger level
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    if logger_name:
        logging.getLogger(logger_name).setLevel(level)
    else:
        logging.getLogger().setLevel(level)


# =============================================================================
# PERFORMANCE LOGGING DECORATORS
# =============================================================================

def log_execution_time(logger: Optional[logging.Logger] = None, 
                       level: int = logging.DEBUG):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger to use (uses root logger if None)
        level: Log level for timing messages
    
    Example:
        >>> @log_execution_time()
        ... def slow_function():
        ...     time.sleep(1)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            log.log(level, f"{func.__name__} executed in {elapsed:.3f}s")
            return result
        
        return wrapper
    return decorator


def log_entry_exit(logger: Optional[logging.Logger] = None,
                   level: int = logging.DEBUG):
    """
    Decorator to log function entry and exit.
    
    Args:
        logger: Logger to use
        level: Log level for messages
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            
            log.log(level, f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                log.log(level, f"Exiting {func.__name__} (success)")
                return result
            except Exception as e:
                log.log(logging.ERROR, f"Exiting {func.__name__} (exception: {e})")
                raise
        
        return wrapper
    return decorator


# =============================================================================
# CONTEXT MANAGER FOR TEMPORARY LOG LEVEL
# =============================================================================

class LogLevel:
    """
    Context manager for temporarily changing log level.
    
    Example:
        >>> with LogLevel('DEBUG'):
        ...     logger.debug("This will be shown")
    """
    
    def __init__(self, level: Union[str, int], logger_name: Optional[str] = None):
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.logger_name = logger_name
        self.original_level = None
    
    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level
        logger.setLevel(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.original_level)
        return False


# =============================================================================
# PROGRESS LOGGING
# =============================================================================

class ProgressLogger:
    """
    Simple progress logger for long-running operations.
    
    Example:
        >>> progress = ProgressLogger(total=100, desc="Processing")
        >>> for i in range(100):
        ...     # do work
        ...     progress.update()
        >>> progress.finish()
    """
    
    def __init__(self, 
                 total: int, 
                 desc: str = 'Progress',
                 logger: Optional[logging.Logger] = None,
                 log_interval: int = 10):
        self.total = total
        self.desc = desc
        self.logger = logger or logging.getLogger(__name__)
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.time()
        self.last_log = 0
        
        self.logger.info(f"{desc}: Starting (total={total})")
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current += n
        
        # Log at intervals
        percent = (self.current / self.total) * 100
        if percent - self.last_log >= self.log_interval or self.current == self.total:
            elapsed = time.time() - self.start_time
            remaining = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
            
            self.logger.info(
                f"{self.desc}: {self.current}/{self.total} "
                f"({percent:.1f}%) - Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s"
            )
            self.last_log = percent
    
    def finish(self) -> None:
        """Mark progress as complete."""
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.desc}: Complete ({elapsed:.1f}s)")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_exception(logger: logging.Logger, 
                  exc: Exception,
                  message: str = "An error occurred") -> None:
    """
    Log an exception with traceback.
    
    Args:
        logger: Logger to use
        exc: Exception to log
        message: Additional message
    """
    logger.error(f"{message}: {exc}", exc_info=True)


def create_log_dir(base_path: str = 'logs') -> str:
    """
    Create a timestamped log directory.
    
    Args:
        base_path: Base directory for logs
    
    Returns:
        str: Path to created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(base_path) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)
