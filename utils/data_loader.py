"""Helpers for loading and validating uploaded CSV files."""

from __future__ import annotations

import io
from typing import Tuple

import pandas as pd

from utils.context import AnalysisContext


def load_csv_from_bytes(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded CSV bytes into a DataFrame with friendly error handling."""
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded file is empty or not a valid CSV.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("Could not parse the CSV. Please check the file format.") from exc
    if df.empty:
        raise ValueError("The CSV file contains no rows.")
    if df.shape[1] < 2:
        raise ValueError("The CSV must contain at least two columns to analyze.")
    return df


def summarize_for_display(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (head, dtypes) for quick UI display."""
    return df.head(10), df.dtypes.astype(str).rename("dtype").to_frame()
