"""Build a professional PDF report from the analysis context using ReportLab."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.context import AnalysisContext

# Brand palette
NAVY = colors.HexColor("#0F2A44")
TEAL = colors.HexColor("#1F7A8C")
LIGHT = colors.HexColor("#F4F7FA")
GREY = colors.HexColor("#5A6B7B")
ACCENT = colors.HexColor("#E8A33D")


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], textColor=NAVY, fontSize=22, spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], textColor=GREY, fontSize=11, spaceAfter=18),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], textColor=NAVY, fontSize=15, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], textColor=TEAL, fontSize=12, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"], textColor=colors.HexColor("#1B2733"), fontSize=10, leading=15, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], textColor=colors.HexColor("#1B2733"), fontSize=10, leading=14, leftIndent=14, bulletIndent=4, spaceAfter=3),
        "small": ParagraphStyle("small", parent=base["Normal"], textColor=GREY, fontSize=8),
    }


def _kv_table(rows: List[List[str]]) -> Table:
    t = Table(rows, colWidths=[1.9 * inch, 4.6 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D4DEE7")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E1E8EE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def _df_to_table(df, max_rows: int = 12, max_cols: int = 6) -> Table:
    cols = list(df.columns[:max_cols])
    head = df[cols].head(max_rows)
    data = [cols] + [
        [("" if pd_isna(v) else _fmt(v)) for v in row] for row in head.values.tolist()
    ]
    col_w = 6.5 * inch / len(cols)
    t = Table(data, colWidths=[col_w] * len(cols), repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D4DEE7")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E1E8EE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 0.01 else f"{v:.4g}"
    return str(v)


def pd_isna(v: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(v))
    except Exception:
        return v is None


def _bullets(text: str, style) -> List[Paragraph]:
    """Split a multi-line string into bullet paragraphs."""
    lines = [ln.strip("-•* ").strip() for ln in text.splitlines() if ln.strip()]
    return [Paragraph(f"• {ln}", style) for ln in lines]


def build_pdf(ctx: AnalysisContext, output_path: str) -> str:
    """Render the full report to ``output_path`` and return the path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Data Analytics Report",
    )
    s = _styles()
    story: List[Any] = []

    # Cover
    story.append(Paragraph("Data Analytics Report", s["title"]))
    story.append(Paragraph(f"Source file: {ctx.filename}", s["subtitle"]))
    story.append(Spacer(1, 6))

    # Overview
    story.append(Paragraph("1. Dataset Overview", s["h1"]))
    shape = ctx.cleaned_df.shape if ctx.cleaned_df is not None else (0, 0)
    overview = [
        ["Rows (after cleaning)", f"{shape[0]:,}"],
        ["Columns", f"{shape[1]:,}"],
        ["Missing values handled", str(ctx.cleaning_report.get("missing_filled", "N/A"))],
        ["Duplicates removed", str(ctx.cleaning_report.get("duplicates_removed", "N/A"))],
        ["Type conversions", str(ctx.cleaning_report.get("type_conversions", "N/A"))],
    ]
    story.append(_kv_table(overview))
    story.append(Spacer(1, 10))
    if ctx.cleaned_df is not None:
        story.append(Paragraph("Preview of cleaned data", s["h2"]))
        story.append(_df_to_table(ctx.cleaned_df))

    # Cleaning summary
    story.append(Paragraph("2. Data Cleaning Summary", s["h1"]))
    notes = ctx.cleaning_report.get("notes", "No cleaning notes recorded.")
    story.extend(_bullets(notes, s["bullet"]) if isinstance(notes, str) else [Paragraph(str(notes), s["body"])])

    # EDA
    story.append(PageBreak())
    story.append(Paragraph("3. Exploratory Data Analysis", s["h1"]))
    eda = ctx.eda_report
    if eda.get("describe") is not None:
        story.append(Paragraph("3.1 Descriptive Statistics", s["h2"]))
        story.append(_df_to_table(eda["describe"], max_rows=15, max_cols=6))

    corr = eda.get("correlation")
    if corr is not None and not corr.empty:
        story.append(Paragraph("3.2 Correlation Matrix", s["h2"]))
        story.append(_df_to_table(corr, max_rows=12, max_cols=8))

    # Charts
    if ctx.chart_paths:
        story.append(Paragraph("3.3 Visualizations", s["h2"]))
        for label, path in ctx.chart_paths.items():
            if os.path.exists(path):
                story.append(Paragraph(label.replace("_", " ").title(), s["small"]))
                story.append(Image(path, width=6.2 * inch, height=3.6 * inch))
                story.append(Spacer(1, 8))

    # Insights
    story.append(PageBreak())
    story.append(Paragraph("4. Business Insights & Recommendations", s["h1"]))
    if ctx.insights:
        story.extend(_bullets(ctx.insights, s["bullet"]))
    else:
        story.append(Paragraph("No insights were generated.", s["body"]))

    doc.build(story)
    return output_path
