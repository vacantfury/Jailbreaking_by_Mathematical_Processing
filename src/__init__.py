"""
Main package initialization for Jailbreaking by Mathematical Processing.

This module handles common initialization tasks that should run before
any other package code, including fixing platform-specific issues.
"""
import os
import sys

# =============================================================================
# Platform-specific Fixes
# =============================================================================

# Fix OpenMP duplicate library issue on macOS
# This MUST be set before importing torch/numpy/any library that uses OpenMP
# See: http://openmp.llvm.org/
if sys.platform == 'darwin':  # macOS only
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =============================================================================
# Package Metadata
# =============================================================================

__version__ = "0.1.0"
__author__ = "Jailbreaking Research Team"

