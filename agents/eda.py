"""EDA Agent: descriptive stats, correlation, and chart generation."""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict

# Capture the pristine warnings.warn before any crewai import patches it.
from utils.warn_guard import restore_warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.context import AnalysisContext

try:
    from crewai import Agent
except Exception:
    Agent = None  # type: ignore

sns.set_theme(style="whitegrid", palette="muted")


def _save_chart(ctx: AnalysisContext, name: str, fig) -> str:
    path = ctx.path("charts", f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def run_eda(ctx: AnalysisContext) -> AnalysisContext:
    """Compute statistics and render charts into the working directory."""
    if ctx.cleaned_df is None:
        raise ValueError("EDA requires a cleaned dataframe.")
    df = ctx.cleaned_df
    ctx.ensure_work_dir()

    # Restore the original warnings.warn — crewai patches it with a filtered_warn
    # that breaks matplotlib's deprecation-warning path.
    restore_warnings()
    return _run_eda_inner(ctx, df)


def _run_eda_inner(ctx: AnalysisContext, df):

    eda: Dict[str, Any] = {}
    chart_paths: Dict[str, str] = {}

    # Descriptive statistics
    desc = df.describe(include="all").transpose()
    eda["describe"] = desc

    # Correlation matrix (numeric only)
    numeric = df.select_dtypes(include=["number"])
    corr = numeric.corr(numeric_only=True) if numeric.shape[1] >= 2 else pd.DataFrame()
    eda["correlation"] = corr if not corr.empty else None
    eda["numeric_columns"] = list(numeric.columns)
    eda["categorical_columns"] = list(df.select_dtypes(exclude=["number"]).columns)
    eda["row_count"] = int(len(df))

    # Chart 1: missing-value heatmap (pre-cleaning proxy = current nulls)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.isna().transpose(), cbar=False, cmap="Blues", ax=ax)
    ax.set_title("Missing Values After Cleaning")
    chart_paths["missing_values"] = _save_chart(ctx, "missing_values", fig)

    # Chart 2: correlation heatmap
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        chart_paths["correlation_heatmap"] = _save_chart(ctx, "correlation_heatmap", fig)

    # Chart 3: distribution of first numeric column
    if not numeric.empty:
        col = numeric.columns[0]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(numeric[col].dropna(), kde=True, color="#1F7A8C", ax=ax)
        ax.set_title(f"Distribution of {col}")
        chart_paths[f"distribution_{col}"] = _save_chart(ctx, f"distribution_{col}", fig)

    # Chart 4: count plot of first categorical column
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    if len(cat_cols) > 0:
        col = cat_cols[0]
        top = df[col].astype(str).value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top.values, y=top.index, color="#E8A33D", ax=ax)
        ax.set_title(f"Top categories: {col}")
        ax.set_xlabel("Count")
        chart_paths[f"count_{col}"] = _save_chart(ctx, f"count_{col}", fig)

    ctx.eda_report = eda
    ctx.chart_paths = chart_paths
    return ctx


def build_agent(llm: Any = None):
    if Agent is None:
        return None
    return Agent(
        role="Exploratory Data Analyst",
        goal="Compute descriptive statistics, correlations, and render visualizations that reveal trends.",
        backstory=(
            "An analytical thinker who turns raw numbers into clear visual "
            "stories, spotting distributions and correlations others miss."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
