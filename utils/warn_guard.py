"""Capture the pristine ``warnings.warn`` before CrewAI patches it.

CrewAI replaces ``warnings.warn`` with a custom ``filtered_warn`` that does
not accept the ``skip_file_prefixes`` keyword argument that matplotlib 3.11+
passes. This breaks seaborn heatmap rendering.

This module must be imported before any crewai import. It stashes the
original ``warnings.warn`` so chart-rendering code can restore it.
"""

import warnings

ORIGINAL_WARN = warnings.warn


def restore_warnings():
    """Restore the original ``warnings.warn`` if it has been patched."""
    warnings.warn = ORIGINAL_WARN
