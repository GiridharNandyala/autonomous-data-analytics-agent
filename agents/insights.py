"""Business Insights Agent.

Deterministically derives key takeaways from the EDA report and, when an LLM
is configured, augments them with strategic narrative recommendations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

from utils.context import AnalysisContext

try:
    from crewai import Agent, Task
except Exception:
    Agent = None  # type: ignore
    Task = None  # type: ignore


def _deterministic_insights(ctx: AnalysisContext) -> List[str]:
    eda = ctx.eda_report
    df = ctx.cleaned_df
    insights: List[str] = []

    insights.append(f"The dataset contains {eda.get('row_count', len(df)):,} rows across {df.shape[1]} columns.")

    numeric_cols = eda.get("numeric_columns", [])
    if numeric_cols:
        insights.append(f"Numeric columns analyzed: {', '.join(numeric_cols)}.")

    corr = eda.get("correlation")
    if corr is not None and not corr.empty:
        pairs = []
        for i in range(len(corr.index)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if pd.notna(val) and abs(val) >= 0.6:
                    pairs.append(f"{corr.index[i]} & {corr.columns[j]} ({val:+.2f})")
        if pairs:
            insights.append("Strong correlations detected: " + "; ".join(pairs[:5]) + ".")
        else:
            insights.append("No strong correlations (|r| >= 0.6) were found among numeric columns.")

    desc = eda.get("describe")
    if desc is not None and not desc.empty:
        for col in numeric_cols[:3]:
            if col in desc.index:
                row = desc.loc[col]
                mean = row.get("mean")
                std = row.get("std")
                if pd.notna(mean) and pd.notna(std) and std > 0:
                    cv = std / mean if mean != 0 else float("inf")
                    spread = "high" if cv > 0.5 else "moderate" if cv > 0.2 else "low"
                    insights.append(
                        f"'{col}' averages {mean:,.2f} with {spread} variability (std {std:,.2f})."
                    )

    cat_cols = eda.get("categorical_columns", [])
    if cat_cols:
        col = cat_cols[0]
        top = df[col].astype(str).value_counts().head(1)
        if not top.empty:
            insights.append(f"The most common '{col}' is '{top.index[0]}' ({top.iloc[0]:,} records).")

    insights.append("Recommendation: prioritize the strongest correlated variables for predictive modeling.")
    insights.append("Recommendation: investigate high-variability columns for data quality or segmentation opportunities.")

    return insights


def run_insights(ctx: AnalysisContext, llm: Any = None) -> AnalysisContext:
    base = _deterministic_insights(ctx)

    if llm is not None and Agent is not None and Task is not None:
        try:
            agent = build_agent(llm)
            summary = json.dumps(
                {
                    "describe": ctx.eda_report.get("describe").head(20).to_dict() if ctx.eda_report.get("describe") is not None else {},
                    "correlation": ctx.eda_report.get("correlation").to_dict() if ctx.eda_report.get("correlation") is not None else {},
                    "numeric_columns": ctx.eda_report.get("numeric_columns", []),
                    "categorical_columns": ctx.eda_report.get("categorical_columns", []),
                },
                default=str,
            )
            task = Task(
                description=(
                    "Given these EDA metrics, produce 5-7 concise business "
                    "insights and strategic recommendations as bullet points:\n" + summary
                ),
                expected_output="Bullet-point list of insights and recommendations.",
                agent=agent,
            )
            # Execute single task directly
            result = task.execute_sync() if hasattr(task, "execute_sync") else agent.execute_task(task)
            extra = str(result).strip()
            if extra:
                base.append("--- LLM Recommendations ---")
                base.append(extra)
        except Exception as exc:
            base.append(f"(LLM insight generation skipped: {exc})")

    ctx.insights = "\n".join(f"- {b}" for b in base)
    return ctx


def build_agent(llm: Any = None):
    if Agent is None:
        return None
    return Agent(
        role="Business Insights Strategist",
        goal="Translate EDA metrics into clear business takeaways, anomalies, and recommendations.",
        backstory=(
            "A senior strategy consultant who connects data patterns to "
            "business decisions and communicates recommendations in plain language."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
