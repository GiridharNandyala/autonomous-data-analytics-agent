"""Shared execution context passed between agents and the Streamlit UI.

A single ``AnalysisContext`` instance carries the uploaded dataframe, the
filesystem paths the agents write artifacts to, and the intermediate results
each agent produces. Keeping everything in one object avoids global state and
makes the pipeline easy to reason about.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class AnalysisContext:
    """Mutable container shared by every agent in the pipeline."""

    # Raw upload
    raw_df: Optional[pd.DataFrame] = None
    filename: str = "upload.csv"

    # Working directory for artifacts (charts, cleaned csv, etc.)
    work_dir: str = "."

    # Intermediate results
    cleaned_df: Optional[pd.DataFrame] = None
    cleaning_report: Dict[str, Any] = field(default_factory=dict)
    eda_report: Dict[str, Any] = field(default_factory=dict)
    chart_paths: Dict[str, str] = field(default_factory=dict)
    insights: str = ""

    # Final output
    pdf_path: Optional[str] = None

    def path(self, *parts: str) -> str:
        """Join paths relative to the working directory."""
        return os.path.join(self.work_dir, *parts)

    def ensure_work_dir(self) -> None:
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.path("charts"), exist_ok=True)
