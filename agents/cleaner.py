"""Data Cleaning Agent.

Performs deterministic cleaning (missing values, duplicates, type coercion)
and exposes a CrewAI ``Agent`` definition so an LLM can add narrative notes
when an API key is available.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from utils.context import AnalysisContext

try:
    from crewai import Agent
except Exception:  # crewai not installed in some environments
    Agent = None  # type: ignore


def run_cleaning(ctx: AnalysisContext) -> AnalysisContext:
    """Clean the raw dataframe in place on the context."""
    if ctx.raw_df is None:
        raise ValueError("No dataframe has been loaded for cleaning.")

    df = ctx.raw_df.copy()
    notes: list[str] = []
    missing_filled = 0
    duplicates_removed = 0
    type_conversions = 0

    # 1. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)
    if duplicates_removed:
        notes.append(f"Removed {duplicates_removed} duplicate row(s).")

    # 2. Strip whitespace from column names and string cells
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    # 3. Infer better dtypes
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_numeric(df[col], errors="raise")
                df[col] = converted
                type_conversions += 1
                notes.append(f"Converted '{col}' to numeric.")
            except (ValueError, TypeError):
                pass

    # 4. Handle missing values
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
            notes.append(f"Filled {n_missing} missing value(s) in '{col}' with median ({median:.2f}).")
        else:
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill)
            notes.append(f"Filled {n_missing} missing value(s) in '{col}' with '{fill}'.")
        missing_filled += n_missing

    # 5. Drop columns that are entirely empty
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        notes.append(f"Dropped fully-empty column(s): {', '.join(empty_cols)}.")

    ctx.cleaned_df = df
    ctx.cleaning_report = {
        "missing_filled": missing_filled,
        "duplicates_removed": duplicates_removed,
        "type_conversions": type_conversions,
        "notes": "\n".join(f"- {n}" for n in notes) if notes else "- No cleaning actions were required.",
    }
    return ctx


def build_agent(llm: Any = None):
    """Return a CrewAI Agent for narrative cleaning notes (optional)."""
    if Agent is None:
        return None
    return Agent(
        role="Data Cleaning Specialist",
        goal="Ensure the dataset is free of missing values, duplicates, and type errors before analysis.",
        backstory=(
            "A meticulous data engineer who has cleaned thousands of messy "
            "datasets and knows exactly which imputation strategy fits each column."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
