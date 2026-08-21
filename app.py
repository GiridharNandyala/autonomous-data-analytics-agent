"""Streamlit UI for the Autonomous Data Analytics Agent System."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from utils.warn_guard import restore_warnings  # noqa: F401  must precede crewai
from crew import AnalyticsCrew
from utils.context import AnalysisContext
from utils.data_loader import load_csv_from_bytes, summarize_for_display

st.set_page_config(
    page_title="Autonomous Data Analytics Agent",
    page_icon="📊",
    layout="wide",
)

# --- Theme / styling ---------------------------------------------------------
st.markdown(
    """
    <style>
      .main .block-container {max-width: 1100px; padding-top: 2rem;}
      .stButton>button {border-radius: 8px; font-weight:600;}
      .step-label {font-size: 0.95rem; color: #1F7A8C; font-weight:600;}
      .metric-card {background:#F4F7FA; border:1px solid #E1E8EE; border-radius:10px; padding:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Autonomous Data Analytics Agent System")
st.caption(
    "Upload a CSV — the agent crew cleans the data, runs EDA, extracts "
    "business insights, and produces a downloadable PDF report."
)

has_openai_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
if not has_openai_key:
    st.info(
        "No `OPENAI_API_KEY` found in the environment. The pipeline will run "
        "with deterministic, rule-based insights. Add a key to `.env` to enable "
        "LLM-powered narrative recommendations."
    )

# --- Upload -----------------------------------------------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    try:
        raw_bytes = uploaded.getvalue()
        df = load_csv_from_bytes(raw_bytes, uploaded.name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"Loaded **{len(df):,}** rows and **{df.shape[1]}** columns from `{uploaded.name}`.")

    with st.expander("Preview raw data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # --- Run pipeline -------------------------------------------------------
    if st.button("Run Analytics Pipeline", type="primary"):
        work_dir = tempfile.mkdtemp(prefix="analytics_")
        ctx = AnalysisContext(raw_df=df, filename=uploaded.name, work_dir=work_dir)
        crew = AnalyticsCrew(ctx)

        progress = st.progress(0.0)
        status = st.empty()
        step_labels = [
            "Cleaning data",
            "Running exploratory data analysis",
            "Extracting business insights",
            "Generating PDF report",
        ]
        total = len(step_labels)

        try:
            for idx, (label, updated_ctx) in enumerate(crew.run()):
                status.markdown(f"<span class='step-label'>Step {idx + 1}/{total}: {label}…</span>", unsafe_allow_html=True)
                progress.progress((idx + 1) / total)
        except RuntimeError as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

        status.success("Pipeline complete.")
        progress.progress(1.0)

        # --- Results --------------------------------------------------------
        st.divider()
        st.header("Results")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Rows (cleaned)", f"{ctx.cleaned_df.shape[0]:,}")
        col_b.metric("Missing filled", ctx.cleaning_report.get("missing_filled", 0))
        col_c.metric("Duplicates removed", ctx.cleaning_report.get("duplicates_removed", 0))
        col_d.metric("Charts generated", len(ctx.chart_paths))

        with st.expander("Data Cleaning Summary", expanded=True):
            notes = ctx.cleaning_report.get("notes", "")
            st.markdown(notes)

        with st.expander("Cleaned Data Preview", expanded=False):
            st.dataframe(ctx.cleaned_df.head(25), use_container_width=True)

        with st.expander("Descriptive Statistics", expanded=False):
            desc = ctx.eda_report.get("describe")
            if desc is not None:
                st.dataframe(desc, use_container_width=True)

        with st.expander("Correlation Matrix", expanded=False):
            corr = ctx.eda_report.get("correlation")
            if corr is not None and not corr.empty:
                st.dataframe(corr, use_container_width=True)
            else:
                st.write("Not enough numeric columns for correlation.")

        if ctx.chart_paths:
            with st.expander("Visualizations", expanded=True):
                cols = st.columns(2)
                for i, (label, path) in enumerate(ctx.chart_paths.items()):
                    with cols[i % 2]:
                        st.image(path, caption=label.replace("_", " ").title(), use_container_width=True)

        with st.expander("Business Insights & Recommendations", expanded=True):
            st.markdown(ctx.insights)

        # --- Download -------------------------------------------------------
        st.divider()
        if ctx.pdf_path and os.path.exists(ctx.pdf_path):
            with open(ctx.pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name="analytics_report.pdf",
                    mime="application/pdf",
                    type="primary",
                )
        else:
            st.warning("PDF report was not generated.")
else:
    st.info("Awaiting a CSV upload to begin.")
