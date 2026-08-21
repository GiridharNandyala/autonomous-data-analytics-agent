"""Report Generator Agent: compiles everything into a PDF."""

from __future__ import annotations

import os
from typing import Any

from utils.context import AnalysisContext
from utils.pdf_builder import build_pdf

try:
    from crewai import Agent
except Exception:
    Agent = None  # type: ignore


def run_report(ctx: AnalysisContext) -> AnalysisContext:
    if ctx.cleaned_df is None:
        raise ValueError("Cannot build report: cleaned dataframe is missing.")
    output_path = ctx.path("analytics_report.pdf")
    build_pdf(ctx, output_path)
    if not os.path.exists(output_path):
        raise RuntimeError("PDF generation failed: file was not written.")
    ctx.pdf_path = output_path
    return ctx


def build_agent(llm: Any = None):
    if Agent is None:
        return None
    return Agent(
        role="Report Generator",
        goal="Compile cleaned data, EDA metrics, charts, and insights into a professional PDF report.",
        backstory="A meticulous technical writer who produces polished, board-ready reports.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
