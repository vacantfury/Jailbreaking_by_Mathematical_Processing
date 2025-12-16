"""
Shared logger utility for the project.
"""
import logging
import sys

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with optional color formatting.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level)
    
    if COLORLOG_AVAILABLE:
        # Use colorlog for colored output
        handler = colorlog.StreamHandler(sys.stdout)
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)s: %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
    else:
        # Fallback to standard logging
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(levelname)s: %(message)s'
        ))
    
    logger.addHandler(handler)
    return logger

