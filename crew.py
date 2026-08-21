"""Pipeline orchestration.

The ``AnalyticsCrew`` chains the four agents in order and yields progress
labels so the Streamlit UI can show step-by-step status. When an OpenAI API
key is present, the insights agent uses an LLM for richer narrative;
otherwise the pipeline runs fully deterministically with no external calls.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional, Tuple

# Must be imported before any crewai import so the pristine warnings.warn is
# captured before crewai patches it.
from utils.warn_guard import restore_warnings  # noqa: F401

from agents.cleaner import run_cleaning
from agents.eda import run_eda
from agents.insights import run_insights
from agents.reporter import run_report
from utils.context import AnalysisContext


def _maybe_llm():
    """Return a CrewAI LLM if an OpenAI key is configured, else None."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from crewai import LLM

        return LLM(model="gpt-4o", temperature=0.3)
    except Exception:
        return None


class AnalyticsCrew:
    """Runs the four-stage analytics pipeline."""

    def __init__(self, ctx: AnalysisContext):
        self.ctx = ctx
        self.llm: Optional[Any] = _maybe_llm()

    def run(self) -> Iterator[Tuple[str, AnalysisContext]]:
        """Execute the pipeline, yielding (step_label, context) after each stage."""
        steps = [
            ("Cleaning data", run_cleaning),
            ("Running exploratory data analysis", run_eda),
            ("Extracting business insights", lambda c: run_insights(c, self.llm)),
            ("Generating PDF report", run_report),
        ]
        for label, fn in steps:
            try:
                fn(self.ctx)
            except Exception as exc:
                raise RuntimeError(f"Step '{label}' failed: {exc}") from exc
            yield label, self.ctx
