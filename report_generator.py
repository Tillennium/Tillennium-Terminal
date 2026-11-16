from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
import html
import math
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Dark canvas and readable labels for every Matplotlib report chart.
plt.rcParams.update({
    "figure.facecolor": "#000000",
    "axes.facecolor": "#000000",
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether, HRFlowable
)


# =========================================================
# TILLENNIUM NETWORK — INSTITUTIONAL REPORT ENGINE
# =========================================================

NAVY = colors.HexColor("#10263F")
DARK_NAVY = colors.HexColor("#091827")
BLUE = colors.HexColor("#10202F")
LIGHT_BLUE = colors.HexColor("#EAF3FB")
TEAL = colors.HexColor("#10202F")
LIGHT_TEAL = colors.HexColor("#E5F5F3")
GREEN = colors.HexColor("#23845B")
LIGHT_GREEN = colors.HexColor("#E8F5ED")
AMBER = colors.HexColor("#D99522")
LIGHT_AMBER = colors.HexColor("#FFF4DD")
RED = colors.HexColor("#C44747")
LIGHT_RED = colors.HexColor("#FCEAEA")
PURPLE = colors.HexColor("#10202F")
LIGHT_PURPLE = colors.HexColor("#F0EBF8")
GREY = colors.HexColor("#64748B")
LIGHT_GREY = colors.HexColor("#F1F4F7")
DARK = colors.HexColor("#1E293B")
WHITE = colors.white
BORDER = colors.HexColor("#CBD5E1")

PAGE_W, PAGE_H = A4

HEADER_HEIGHT = 25 * mm
HEADER_LOGO_WIDTH = 65 * mm  # ലോഗോ കുറച്ചുകൂടി വലുതാക്കി
HEADER_LOGO_HEIGHT = 19 * mm


# =========================================================
# REPORT DEFINITIONS (UNIFORM TABLE HEADER ACCENT COLOR)
# =========================================================

UNIFORM_ACCENT = colors.HexColor("#1E293B")  # എല്ലാ PDF ടേബിൾ ഹെഡറുകൾക്കും ഒരേ പ്രൊഫഷണൽ കളർ

REPORTS = {
    "company_strategic_intelligence": {
        "title": "Company & Strategic Intelligence",
        "subtitle": "Corporate Profile, Business Model & Strategic Position",
        "accent": UNIFORM_ACCENT,
        "chapters": [
            "Executive Intelligence Summary",
            "Corporate Identity and Company Overview",
            "Corporate History and Development",
            "Business Model Architecture",
            "Products, Services and Solutions",
            "Revenue Model and Monetisation",
            "Revenue Segment Analysis",
            "Geographic and Regional Exposure",
            "Customer Base and End Markets",
            "Industry Structure and Value Chain",
            "Market Size and Industry Drivers",
            "Competitive Landscape",
            "Competitor Comparison Framework",
            "Competitive Positioning",
            "Economic Moat Assessment",
            "Technology and Intellectual Property",
            "Research and Development Profile",
            "Operating Model and Cost Structure",
            "Manufacturing and Operating Footprint",
            "Management and Leadership Overview",
            "Corporate Governance Framework",
            "Capital Allocation Framework",
            "Strategic Growth Priorities",
            "Mergers, Acquisitions and Partnerships",
            "Regulatory and Compliance Environment",
            "Supply Chain Dependency Framework",
            "Geopolitical and Country Exposure",
            "Business Model Resilience",
            "Strategic Catalysts and Watchpoints",
            "Institutional Research Conclusion",
        ],
    },

    "financial_valuation_analysis": {
        "title": "Financial & Valuation Analysis",
        "subtitle": "Financial Performance, Balance Sheet & Valuation Framework",
        "accent": UNIFORM_ACCENT,
        "chapters": [
            "Executive Financial Summary",
            "Financial Data Coverage and Periods",
            "Five-Year Financial Performance",
            "Revenue Growth and Quality",
            "Gross Profit and Gross Margin",
            "Operating Profitability",
            "EBITDA and Operating Cash Earnings",
            "Net Income and Earnings Growth",
            "Earnings Per Share Analysis",
            "Operating Cash Flow",
            "Free Cash Flow Framework",
            "Capital Expenditure Analysis",
            "Working Capital Assessment",
            "Receivables and Inventory Review",
            "Balance Sheet Structure",
            "Cash and Liquidity Position",
            "Debt and Leverage Assessment",
            "Interest Coverage and Financing Risk",
            "Shareholders' Equity and Returns",
            "Return on Invested Capital Framework",
            "Financial Quality Assessment",
            "Historical Valuation Multiples",
            "Comparable Company Framework",
            "Discounted Cash Flow Methodology",
            "Cost of Capital and WACC Framework",
            "Terminal Value Methodology",
            "Bull, Base and Bear Scenarios",
            "Valuation Sensitivity Framework",
            "Financial Risks and Key Watchpoints",
            "Financial Research Conclusion",
        ],
    },

    "market_risk_intelligence": {
        "title": "Market & Risk Intelligence",
        "subtitle": "Market Behaviour, Ownership, Risk & Stress Testing",
        "accent": UNIFORM_ACCENT,
        "chapters": [
            "Executive Market and Risk Summary",
            "Market Data Coverage and Methodology",
            "Historical Price Performance",
            "Daily and Period Return Analysis",
            "Price Trend and Market Structure",
            "Historical Trading Volume",
            "Volatility Assessment",
            "Maximum Drawdown Analysis",
            "Risk-Adjusted Performance Framework",
            "Beta and Market Sensitivity",
            "52-Week Price Range",
            "Market Capitalisation Profile",
            "Liquidity and Trading Activity",
            "Relative Performance Framework",
            "Sector and Industry Benchmarking",
            "Institutional Ownership Overview",
            "Major Disclosed Institutional Holders",
            "Institutional Ownership Concentration",
            "Ownership Change Monitoring",
            "Insider and Short Interest Data Availability",
            "ETF and Fund Ownership Framework",
            "Shareholder Structure Assessment",
            "Market and Macro Risk",
            "Interest Rate and Currency Sensitivity",
            "Hypothetical Downside Stress Scenario",
            "Severe Downside Stress Scenario",
            "Recovery Scenario Framework",
            "SEC Filing Intelligence and Disclosure Review",
            "Enterprise Risk Monitoring Framework",
            "Institutional Risk Conclusion",
        ],
    },
}


# =========================================================
# SAFE DATA HELPERS
# =========================================================

def val(value, fallback="Not available from current data source"):
    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    if isinstance(value, float):
        if not math.isfinite(value):
            return fallback
        return f"{value:,.2f}"

    if isinstance(value, int):
        return f"{value:,}"

    return str(value)


def esc(value):
    return html.escape(val(value)).replace("\n", "<br/>")


def pct(value):
    if value is None:
        return "Unavailable"

    try:
        number = float(value)
        if not math.isfinite(number):
            return "Unavailable"
        return f"{number:.2f}%"
    except (TypeError, ValueError):
        return "Unavailable"


def money(value, currency=""):
    if value is None:
        return "Unavailable"

    try:
        number = float(value)
        if not math.isfinite(number):
            return "Unavailable"
        return f"{currency} {number:,.2f}".strip()
    except (TypeError, ValueError):
        return val(value)


def frame_or_empty(value):
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def get_history(data):
    return frame_or_empty(data.get("history"))


def numeric_series(frame, possible_names):
    frame = frame_or_empty(frame)

    if frame.empty:
        return None

    for name in possible_names:
        for index in frame.index:
            if str(index).strip().lower() == name.lower():
                series = pd.to_numeric(
                    frame.loc[index], errors="coerce"
                ).dropna()

                if not series.empty:
                    return series

        for column in frame.columns:
            if str(column).strip().lower() == name.lower():
                series = pd.to_numeric(
                    frame[column], errors="coerce"
                ).dropna()

                if not series.empty:
                    return series

    return None


def get_close(history):
    history = frame_or_empty(history)

    if history.empty or "Close" not in history.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        history["Close"], errors="coerce"
    ).dropna()


def history_metrics(history):
    close = get_close(history)

    if close.empty:
        return {}

    returns = close.pct_change().dropna()
    drawdown = close / close.cummax() - 1

    volatility = None
    if len(returns) > 1:
        volatility = float(
            returns.std() * math.sqrt(252) * 100
        )

    period_return = None
    if len(close) > 1 and close.iloc[0] != 0:
        period_return = float(
            (close.iloc[-1] / close.iloc[0] - 1) * 100
        )

    average_volume = None

    if "Volume" in history.columns:
        volume = pd.to_numeric(
            history["Volume"], errors="coerce"
        ).dropna()

        if not volume.empty:
            average_volume = float(volume.mean())

    return {
        "observations": len(close),
        "start": str(close.index[0]),
        "end": str(close.index[-1]),
        "first_close": float(close.iloc[0]),
        "last_close": float(close.iloc[-1]),
        "period_return": period_return,
        "volatility": volatility,
        "max_drawdown": float(drawdown.min() * 100),
        "average_volume": average_volume,
    }


def safe_filename(value):
    value = str(value or "COMPANY").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return value.strip("_") or "COMPANY"


# =========================================================
# COLOUR AND SIGNAL HELPERS
# =========================================================

def signal_colour(label):
    text = str(label).strip().lower()

    if any(word in text for word in [
        "critical", "high risk", "severe", "danger", "red"
    ]):
        return RED, LIGHT_RED

    if any(word in text for word in [
        "caution", "moderate", "warning", "monitor", "amber"
    ]):
        return AMBER, LIGHT_AMBER

    if any(word in text for word in [
        "low risk", "strong", "positive", "healthy", "green"
    ]):
        return GREEN, LIGHT_GREEN

    return BLUE, LIGHT_BLUE


def signal_table(label, description=""):
    foreground, background = signal_colour(label)

    table = Table(
        [[Paragraph(
            f"<b>{html.escape(str(label).upper())}</b>",
            ParagraphStyle(
                "signal",
                fontName="Helvetica-Bold",
                fontSize=8,
                textColor=foreground,
                alignment=TA_CENTER,
            )
        ), Paragraph(
            html.escape(str(description)),
            styles()["small"]
        )]],
        colWidths=[34*mm, 123*mm]
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), background),
        ("BACKGROUND", (1, 0), (1, 0), LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    return table


# =========================================================
# CHART GENERATION
# =========================================================

def finish_chart(fig):
    fig.tight_layout()

    output = BytesIO()
    # Export report charts with a pure black background.
    fig.patch.set_facecolor("#000000")
    for axis in fig.axes:
        axis.set_facecolor("#000000")
        axis.tick_params(colors="white")
        axis.xaxis.label.set_color("white")
        axis.yaxis.label.set_color("white")
        axis.title.set_color("white")
        for spine in axis.spines.values():
            spine.set_color("#666666")
        legend = axis.get_legend()
        if legend is not None:
            for legend_text in legend.get_texts():
                legend_text.set_color("white")

    fig.savefig(
        output,
        format="png",
        dpi=220,
        bbox_inches="tight",
        facecolor="#000000"
    )

    plt.close(fig)
    output.seek(0)

    return output


def make_chart(history, chart_type="price"):
    history = frame_or_empty(history)

    if history.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.7))
    made = False

    if chart_type == "price" and "Close" in history.columns:
        series = get_close(history)

        if not series.empty:
            ax.plot(
                series.index,
                series.values,
                color="#246AA6",
                linewidth=1.8,
                label="Closing price"
            )

            ax.fill_between(
                series.index,
                series.values,
                min(series.values),
                color="#246AA6",
                alpha=0.08
            )

            ax.set_title(
                "Historical Closing Price",
                fontsize=12,
                fontweight="bold"
            )
            ax.set_ylabel("Price")
            ax.legend(loc="upper left", frameon=False)
            made = True

    elif chart_type == "volume" and "Volume" in history.columns:
        series = pd.to_numeric(
            history["Volume"], errors="coerce"
        ).dropna()

        if not series.empty:
            ax.bar(
                series.index,
                series.values,
                color="#168C91",
                width=1.0,
                alpha=0.8
            )

            ax.set_title(
                "Historical Trading Volume",
                fontsize=12,
                fontweight="bold"
            )
            ax.set_ylabel("Volume")
            made = True

    elif chart_type == "returns" and "Close" in history.columns:
        series = get_close(history)

        if len(series) > 1:
            returns = series.pct_change().dropna() * 100

            ax.plot(
                returns.index,
                returns.values,
                color="#7456A6",
                linewidth=0.9
            )

            ax.axhline(
                0,
                color="#64748B",
                linewidth=0.8,
                linestyle="--"
            )

            ax.fill_between(
                returns.index,
                returns.values,
                0,
                where=returns.values >= 0,
                color="#23845B",
                alpha=0.22
            )

            ax.fill_between(
                returns.index,
                returns.values,
                0,
                where=returns.values < 0,
                color="#C44747",
                alpha=0.22
            )

            ax.set_title(
                "Daily Historical Returns",
                fontsize=12,
                fontweight="bold"
            )
            ax.set_ylabel("Return (%)")
            made = True

    elif chart_type == "drawdown" and "Close" in history.columns:
        series = get_close(history)

        if len(series) > 1:
            drawdown = (series / series.cummax() - 1) * 100

            ax.fill_between(
                drawdown.index,
                drawdown.values,
                0,
                color="#C44747",
                alpha=0.22
            )

            ax.plot(
                drawdown.index,
                drawdown.values,
                color="#C44747",
                linewidth=1.1
            )

            ax.set_title(
                "Historical Drawdown Profile",
                fontsize=12,
                fontweight="bold"
            )
            ax.set_ylabel("Drawdown (%)")
            made = True

    if not made:
        plt.close(fig)
        return None

    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.autofmt_xdate()

    return finish_chart(fig)


def make_financial_chart(frame, chart_type):
    frame = frame_or_empty(frame)

    if frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.7))
    made = False

    if chart_type == "revenue":
        series = numeric_series(
            frame, ["Total Revenue", "Revenue"]
        )

        if series is not None and len(series) > 0:
            ax.bar(
                [str(x) for x in series.index],
                series.values,
                color="#246AA6",
                alpha=0.85
            )

            ax.set_title(
                "Revenue Across Available Reporting Periods",
                fontsize=12,
                fontweight="bold"
            )
            ax.set_ylabel("Reported value")
            made = True

    elif chart_type == "profitability":
        names = [
            ("Gross Profit", "#246AA6"),
            ("Operating Income", "#168C91"),
            ("Net Income", "#7456A6"),
        ]

        plotted = []

        for metric, colour in names:
            series = numeric_series(frame, [metric])

            if series is not None:
                ax.plot(
                    [str(x) for x in series.index],
                    series.values,
                    marker="o",
                    linewidth=1.6,
                    label=metric,
                    color=colour
                )
                plotted.append(metric)

        if plotted:
            ax.set_title(
                "Reported Profitability Indicators",
                fontsize=12,
                fontweight="bold"
            )
            ax.set_ylabel("Reported value")
            ax.legend(frameon=False)
            made = True

    elif chart_type == "cashflow":
        names = [
            ("Operating Cash Flow", "#168C91"),
            ("Capital Expenditure", "#D99522"),
            ("Free Cash Flow", "#246AA6"),
        ]

        plotted = []

        for metric, colour in names:
            series = numeric_series(frame, [metric])

            if series is not None:
                ax.plot(
                    [str(x) for x in series.index],
                    series.values,
                    marker="o",
                    linewidth=1.6,
                    label=metric,
                    color=colour
                )
                plotted.append(metric)

        if plotted:
            ax.set_title(
                "Cash Flow Indicators",
                fontsize=12,
                fontweight="bold"
            )
            ax.set_ylabel("Reported value")
            ax.legend(frameon=False)
            made = True

    elif chart_type == "balance":
        names = [
            ("Total Debt", "#C44747"),
            ("Cash And Cash Equivalents", "#168C91"),
            ("Total Assets", "#246AA6"),
        ]

        plotted = []

        for metric, colour in names:
            series = numeric_series(frame, [metric])

            if series is not None:
                ax.plot(
                    [str(x) for x in series.index],
                    series.values,
                    marker="o",
                    linewidth=1.6,
                    label=metric,
                    color=colour
                )
                plotted.append(metric)

        if plotted:
            ax.set_title(
                "Balance Sheet Indicators",
                fontsize=12,
                fontweight="bold"
            )
            ax.set_ylabel("Reported value")
            ax.legend(frameon=False)
            made = True

    if not made:
        plt.close(fig)
        return None

    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.autofmt_xdate()

    return finish_chart(fig)


def chart_flow(image_data, width=165*mm, height=68*mm):
    if image_data is None:
        return Paragraph(
            "Chart unavailable: the required source data was not returned.",
            styles()["small"]
        )

    return Image(
        image_data,
        width=width,
        height=height,
        kind="proportional"
    )


# =========================================================
# STYLES
# =========================================================

def styles():
    base = getSampleStyleSheet()

    return {
        "cover": ParagraphStyle(
            "TN_Cover",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=12
        ),

        "subtitle": ParagraphStyle(
            "TN_Subtitle",
            parent=base["Normal"],
            fontSize=10.5,
            leading=15,
            textColor=BLUE,
            spaceAfter=10
        ),

        "chapter": ParagraphStyle(
            "TN_Chapter",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=20,
            textColor=NAVY,
            spaceAfter=9
        ),

        "section": ParagraphStyle(
            "TN_Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=13,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=5
        ),

        "body": ParagraphStyle(
            "TN_Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=12,
            textColor=DARK,
            spaceAfter=6
        ),

        "small": ParagraphStyle(
            "TN_Small",
            parent=base["BodyText"],
            fontSize=7,
            leading=9,
            textColor=GREY
        ),

        "toc": ParagraphStyle(
            "TN_TOC",
            parent=base["BodyText"],
            fontSize=8,
            leading=11,
            textColor=DARK
        ),

        "kpi_label": ParagraphStyle(
            "TN_KPI_Label",
            fontName="Helvetica-Bold",
            fontSize=7,
            textColor=GREY,
            alignment=TA_CENTER
        ),

        "kpi_value": ParagraphStyle(
            "TN_KPI_Value",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=NAVY,
            alignment=TA_CENTER
        ),
    }


# =========================================================
# TABLE DESIGN
# =========================================================

def make_table(rows, widths=None, font_size=7, accent=None):
    accent = accent or NAVY

    converted = []

    for row in rows:
        converted_row = []

        for cell in row:
            if isinstance(cell, (Paragraph, Image)):
                converted_row.append(cell)
            else:
                converted_row.append(
                    Paragraph(
                        html.escape(str(cell)).replace("\n", "<br/>"),
                        styles()["small"]
                    )
                )

        converted.append(converted_row)

    table = Table(
        converted,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    return table


def kpi_table(items):
    cells = []

    for label, value in items:
        cells.append([
            Paragraph(
                html.escape(str(label)).upper(),
                styles()["kpi_label"]
            ),
            Paragraph(
                html.escape(str(value)),
                styles()["kpi_value"]
            )
        ])

    rows = []

    for i in range(0, len(cells), 2):
        row = []

        for cell in cells[i:i+2]:
            row.extend(cell)

        if len(row) == 2:
            row.extend(["", ""])

        rows.append(row)

    table = Table(
        rows,
        colWidths=[39*mm, 39*mm, 39*mm, 39*mm]
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("SPAN", (1, 0), (1, 0)),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    return table


# =========================================================
# LOGO HEADER AND FOOTER (UPDATED: LEFT-ALIGNED & LARGER LOGO)
# =========================================================

def draw_page(canvas, document, logo_path=None, accent=None):
    canvas.saveState()

    accent = accent or BLUE

    # Larger header strip
    canvas.setFillColor(DARK_NAVY)
    canvas.rect(
        0,
        PAGE_H - HEADER_HEIGHT,
        PAGE_W,
        HEADER_HEIGHT,
        fill=1,
        stroke=0
    )

    # Accent line below header
    canvas.setFillColor(accent)
    canvas.rect(
        0,
        PAGE_H - HEADER_HEIGHT - 1.2*mm,
        PAGE_W,
        1.2*mm,
        fill=1,
        stroke=0
    )

    # Logo positioned at left margin (18*mm) instead of center, with larger dimensions
    logo = Path(logo_path) if logo_path else None

    if logo and logo.exists():
        try:
            canvas.drawImage(
                str(logo),
                18*mm,  # പേജിന്റെ ഇടതുവശത്തെ മാർജിനുമായി കൃത്യമായി അലൈൻ ചെയ്തു
                PAGE_H - HEADER_HEIGHT + 3.0*mm,
                width=HEADER_LOGO_WIDTH,
                height=HEADER_LOGO_HEIGHT,
                preserveAspectRatio=True,
                anchor="w",  # Left/West anchor for clean alignment
                mask="auto"
            )
        except Exception:
            canvas.setFillColor(WHITE)
            canvas.setFont("Helvetica-Bold", 11)
            canvas.drawString(
                18*mm,
                PAGE_H - 15*mm,
                "TILLENNIUM NETWORK"
            )
    else:
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(
            18*mm,
            PAGE_H - 15*mm,
            "TILLENNIUM NETWORK"
        )

    # Header descriptor on right
    canvas.setFillColor(colors.HexColor("#D8E5F1"))
    canvas.setFont("Helvetica", 6.5)
    canvas.drawRightString(
        PAGE_W - 18*mm,
        PAGE_H - 14*mm,
        "INSTITUTIONAL RESEARCH"
    )

    # Footer
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(
        18*mm, 14*mm,
        PAGE_W - 18*mm, 14*mm
    )

    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 6.5)

    canvas.drawString(
        18*mm,
        8*mm,
        "TILLENNIUM NETWORK | CONFIDENTIAL RESEARCH"
    )

    canvas.drawRightString(
        PAGE_W - 18*mm,
        8*mm,
        f"PAGE {document.page}"
    )

    canvas.restoreState()


# =========================================================
# REPORT DATA TABLES
# =========================================================

def company_table(profile, data):
    return [
        ["Company field", "Retrieved information"],
        ["Company", val(profile.get("name"))],
        ["Ticker", val(profile.get("ticker"))],
        ["Sector", val(profile.get("sector"))],
        ["Industry", val(profile.get("industry"))],
        ["Country", val(profile.get("country"))],
        ["Website", val(profile.get("website"))],
        ["Employees", val(profile.get("employees"))],
        ["Currency", val(profile.get("currency"))],
        [
            "Market capitalisation",
            money(profile.get("market_cap"), profile.get("currency", ""))
        ],
        [
            "Current price",
            money(profile.get("current_price"), profile.get("currency", ""))
        ],
        [
            "52-week high",
            money(profile.get("fifty_two_week_high"), profile.get("currency", ""))
        ],
        [
            "52-week low",
            money(profile.get("fifty_two_week_low"), profile.get("currency", ""))
        ],
        ["Data provider", val(data.get("source"))],
        ["Retrieved at", val(data.get("retrieved_at"))],
    ]


def statement_rows(frame, max_rows=18):
    frame = frame_or_empty(frame)

    if frame.empty:
        return [
            ["Metric", "Connector status"],
            ["Financial statement", "Not available from current connector"],
        ]

    rows = [["Metric / Period", "Provider-returned data"]]

    sample = frame.iloc[:, :6].head(max_rows)

    for index, row in sample.iterrows():
        values = [val(x) for x in row.tolist()]

        rows.append([
            val(index),
            " | ".join(values)
        ])

    return rows


def holder_rows(frame):
    frame = frame_or_empty(frame)

    if frame.empty:
        return [
            ["Ownership data", "Availability"],
            ["Institutional holders", "Not available from current connector"],
            ["Ownership concentration", "Requires verified data"],
        ]

    rows = [["Holder field", "Provider-returned data"]]

    for index, row in frame.head(15).iterrows():
        rows.append([
            val(index),
            " | ".join(val(x) for x in row.tolist())
        ])

    return rows


# =========================================================
# RESEARCH COMMENTARY
# =========================================================

def chapter_text(report_id, chapter, profile, data, market, metrics):
    name = esc(profile.get("name"))
    ticker = esc(profile.get("ticker"))
    sector = esc(profile.get("sector"))
    industry = esc(profile.get("industry"))
    description = esc(profile.get("description"))

    history = get_history(data)
    hm = history_metrics(history)
    topic = chapter.lower()

    paragraphs = [
        (
            f"This section examines <b>{name} ({ticker})</b> using the "
            f"company, financial and market information returned by the "
            f"current data provider. The retrieved sector classification is "
            f"<b>{sector}</b> and the industry field is <b>{industry}</b>. "
            f"All conclusions remain subject to source coverage and "
            f"verification limitations."
        )
    ]

    if report_id == "company_strategic_intelligence":
        if any(x in topic for x in [
            "business", "product", "revenue", "customer",
            "industry", "competitive", "technology",
            "manufacturing", "corporate"
        ]):
            paragraphs.append(
                "<b>Provider company description:</b> " + description
            )

        paragraphs.extend([
            (
                "An institutional company review should distinguish "
                "reported operating facts from management commentary and "
                "independent analytical interpretation. Relevant areas "
                "include revenue drivers, customer end markets, geographic "
                "exposure, product economics, competitive intensity, "
                "operating dependencies and the durability of market position."
            ),
            (
                "A complete assessment requires primary company disclosures, "
                "annual reports, quarterly filings, investor presentations "
                "and independently verified industry sources. Segment-level, "
                "customer-level and supplier-level information may not be "
                "available through the current connector."
            ),
        ])

    elif report_id == "financial_valuation_analysis":
        paragraphs.extend([
            (
                "Financial analysis should examine revenue growth, gross "
                "profitability, operating margins, earnings, cash generation, "
                "capital expenditure, working capital and balance-sheet "
                "resilience. Reported earnings should be compared with "
                "operating cash flow to assess earnings conversion."
            ),
            (
                "Financial values are provider-returned fields. Fiscal "
                "periods, restatements, units and accounting classifications "
                "must be reconciled with the issuer's reported financial "
                "statements before formal investment use."
            ),
        ])

        if any(x in topic for x in [
            "valuation", "discounted", "wacc", "terminal", "scenario"
        ]):
            paragraphs.append(
                "No unsupported target price is asserted. A complete "
                "discounted cash flow model requires sourced forecasts, "
                "tax assumptions, capital expenditure, working capital, "
                "discount rate, cost of debt, capital structure and "
                "terminal growth assumptions. Inputs must be documented "
                "and sensitivity-tested."
            )

    else:
        paragraphs.extend([
            (
                f"The retrieved historical series contains "
                f"{val(hm.get('observations', 0))} closing-price observations. "
                f"The available period begins {val(hm.get('start'))} and ends "
                f"{val(hm.get('end'))}. Historical performance is not a "
                f"forecast of future returns."
            ),
            (
                "Risk analysis should consider price volatility, drawdown, "
                "liquidity, market sensitivity, financial leverage, "
                "operational dependencies and interactions between risk "
                "factors. The current application does not operate a "
                "validated predictive risk model."
            ),
            (
                "Institutional ownership information may be incomplete "
                "or delayed. Filing-specific conclusions require direct "
                "access to the relevant regulatory filings and their "
                "publication dates."
            ),
        ])

    paragraphs.extend([
        (
            f"<b>Research focus:</b> {html.escape(chapter)} should be "
            "assessed using traceable evidence. Record the source, "
            "publication date, reporting period, measurement unit, "
            "currency and methodology."
        ),
        (
            "<b>Evidence classification:</b> Separate confirmed provider "
            "data, company disclosures, third-party research, analyst "
            "assumptions and unverified information. Missing evidence "
            "must remain explicitly identified."
        ),
        (
            "<b>Institutional relevance:</b> This section supports "
            "structured research, identifies information gaps and "
            "defines further verification required before an investment "
            "or corporate decision."
        ),
    ])

    return paragraphs


# =========================================================
# CHAPTER DATA
# =========================================================

def chapter_data_table(report_id, chapter, profile, data, market, metrics):
    hm = history_metrics(get_history(data))
    currency = profile.get("currency", "")

    if report_id == "company_strategic_intelligence":
        rows = [
            ["Research field", "Available information"],
            ["Company", val(profile.get("name"))],
            ["Sector", val(profile.get("sector"))],
            ["Industry", val(profile.get("industry"))],
            ["Country", val(profile.get("country"))],
            ["Business description", val(profile.get("description"))[:350]],
            ["Supplier intelligence", "Requires separate source verification"],
            ["Competitor database", "Not connected in current build"],
        ]

    elif report_id == "financial_valuation_analysis":
        rows = [
            ["Financial indicator", "Available information"],
            ["Currency field", val(currency)],
            [
                "Market capitalisation",
                money(profile.get("market_cap"), currency)
            ],
            [
                "Current price field",
                money(profile.get("current_price"), currency)
            ],
            ["Income statement", "See financial statement appendix"],
            ["Balance sheet", "See balance sheet appendix"],
            ["Cash flow statement", "See cash flow appendix"],
            ["DCF fair value", "Not calculated without verified assumptions"],
            ["WACC", "Requires sourced market and capital structure inputs"],
        ]

    else:
        rows = [
            ["Market / risk indicator", "Retrieved value"],
            ["Price observations", val(hm.get("observations", 0))],
            ["First close", money(hm.get("first_close"), currency)],
            ["Last close", money(hm.get("last_close"), currency)],
            ["Period return", pct(hm.get("period_return"))],
            [
                "Annualised historical volatility",
                pct(hm.get("volatility"))
            ],
            [
                "Maximum observed drawdown",
                pct(hm.get("max_drawdown"))
            ],
            ["Average trading volume", val(hm.get("average_volume"))],
            ["SEC filings", "Not connected in current build"],
            ["Predictive risk model", "Not implemented"],
        ]

    return rows


# =========================================================
# FINANCIAL CHART APPENDIX
# =========================================================

def add_financial_charts(story, data, accent):
    financials = frame_or_empty(data.get("financials"))
    balance = frame_or_empty(data.get("balance_sheet"))
    cashflow = frame_or_empty(data.get("cashflow"))

    story.append(Paragraph(
        "Financial Data Visualisation",
        styles()["chapter"]
    ))

    story.append(Paragraph(
        "The following charts are generated only when the relevant "
        "financial line items are present in the provider-returned "
        "statements. They are not estimates.",
        styles()["body"]
    ))

    chart_specs = [
        (
            "Revenue Trend",
            make_financial_chart(financials, "revenue")
        ),
        (
            "Profitability Indicators",
            make_financial_chart(financials, "profitability")
        ),
        (
            "Cash Flow Indicators",
            make_financial_chart(cashflow, "cashflow")
        ),
        (
            "Balance Sheet Indicators",
            make_financial_chart(balance, "balance")
        ),
    ]

    for title, chart in chart_specs:
        story.append(Paragraph(title, styles()["section"]))
        story.append(chart_flow(chart))
        story.append(Spacer(1, 5*mm))

    story.append(PageBreak())


# =========================================================
# BUILD ONE PDF
# =========================================================

def build_one_report(
    report_id,
    profile,
    data,
    metrics=None,
    market=None,
    risks=None,
    logo_path=None
):
    spec = REPORTS[report_id]

    metrics = metrics or {}
    market = market or {}
    risks = risks or []

    accent = spec["accent"]

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18*mm,
        rightMargin=18*mm,
        topMargin=32*mm,
        bottomMargin=19*mm,
        title=spec["title"],
        author="Tillennium Network",
        subject=f"{profile.get('name', 'Company')} research report",
    )

    s = styles()
    story = []

    name = val(profile.get("name"))
    ticker = val(profile.get("ticker"))
    retrieved = val(data.get("retrieved_at"))
    history = get_history(data)
    hm = history_metrics(history)

    # -----------------------------------------------------
    # COVER
    # -----------------------------------------------------

    story.append(Spacer(1, 14*mm))

    story.append(Paragraph(
        "TILLENNIUM NETWORK",
        ParagraphStyle(
            "cover_brand",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=accent,
            spaceAfter=8
        )
    ))

    story.append(Paragraph(
        html.escape(spec["title"]),
        s["cover"]
    ))

    story.append(Paragraph(
        html.escape(spec["subtitle"]),
        s["subtitle"]
    ))

    story.append(HRFlowable(
        width="100%",
        thickness=2,
        color=accent,
        spaceBefore=5,
        spaceAfter=12
    ))

    cover = [
        ["COMPANY", name],
        ["TICKER", ticker],
        ["SECTOR", val(profile.get("sector"))],
        ["INDUSTRY", val(profile.get("industry"))],
        ["COUNTRY", val(profile.get("country"))],
        ["CURRENCY", val(profile.get("currency"))],
        ["SOURCE", val(data.get("source"))],
        ["RETRIEVED", retrieved],
        [
            "REPORT DATE",
            datetime.now(timezone.utc).strftime("%Y-%m-%d UTC")
        ],
    ]

    story.append(make_table(
        cover,
        [42*mm, 115*mm],
        8,
        accent=accent
    ))

    story.append(Spacer(1, 12*mm))

    story.append(Paragraph(
        "<b>CONFIDENTIAL INSTITUTIONAL RESEARCH</b><br/>"
        "Prepared for research and analytical discussion. This report "
        "is not investment advice, a recommendation, an offer or a solicitation.",
        s["body"]
    ))

    story.append(Spacer(1, 7*mm))

    story.append(Paragraph(
        "Research classification: Analytical research document",
        s["small"]
    ))

    story.append(PageBreak())

    # -----------------------------------------------------
    # EXECUTIVE SNAPSHOT
    # -----------------------------------------------------

    story.append(Paragraph(
        "Research Snapshot",
        s["chapter"]
    ))

    story.append(kpi_table([
        ("Ticker", ticker),
        ("Sector", val(profile.get("sector"))),
        ("Market Cap", money(
            profile.get("market_cap"),
            profile.get("currency", "")
        )),
        ("Period Return", pct(hm.get("period_return"))),
    ]))

    story.append(Spacer(1, 8*mm))

    story.append(Paragraph(
        "Company Profile",
        s["section"]
    ))

    story.append(make_table(
        company_table(profile, data),
        [48*mm, 109*mm],
        7,
        accent=accent
    ))

    story.append(Spacer(1, 6*mm))

    story.append(Paragraph(
        "Data quality and interpretation",
        s["section"]
    ))

    story.append(Paragraph(
        "The snapshot reflects fields returned by the connected data "
        "provider at the retrieval timestamp. It should not be treated "
        "as an audited company disclosure. Verify key figures against "
        "primary-source documents before relying on them.",
        s["body"]
    ))

    story.append(PageBreak())

    # -----------------------------------------------------
    # CONTENTS
    # -----------------------------------------------------

    story.append(Paragraph(
        "Contents",
        s["chapter"]
    ))

    toc = [["SECTION", "CHAPTER"]]

    for i, chapter in enumerate(spec["chapters"], 1):
        toc.append([f"{i:02d}", chapter])

    story.append(make_table(
        toc,
        [22*mm, 135*mm],
        7.5,
        accent=accent
    ))

    story.append(PageBreak())

    # -----------------------------------------------------
    # CHAPTERS
    # -----------------------------------------------------

    for i, chapter in enumerate(spec["chapters"], 1):
        story.append(Paragraph(
            html.escape(chapter),
            s["chapter"]
        ))

        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=accent,
            spaceBefore=0,
            spaceAfter=8
        ))

        story.append(Paragraph(
            "Research Commentary",
            s["section"]
        ))

        for paragraph in chapter_text(
            report_id,
            chapter,
            profile,
            data,
            market,
            metrics
        ):
            story.append(Paragraph(paragraph, s["body"]))

        story.append(Paragraph(
            "Company / Research Data",
            s["section"]
        ))

        story.append(make_table(
            chapter_data_table(
                report_id,
                chapter,
                profile,
                data,
                market,
                metrics
            ),
            [52*mm, 105*mm],
            7,
            accent=accent
        ))

        # Market and risk charts
        if report_id == "market_risk_intelligence":
            chart_map = {
                3: "price",
                4: "returns",
                5: "price",
                6: "volume",
                7: "returns",
                8: "drawdown",
            }

            if i in chart_map:
                story.append(Paragraph(
                    "Market Analytics",
                    s["section"]
                ))

                chart = make_chart(
                    history,
                    chart_map[i]
                )

                story.append(chart_flow(chart))

        # Financial market context
        if (
            report_id == "financial_valuation_analysis"
            and i == 3
        ):
            story.append(Paragraph(
                "Historical Market Price Context",
                s["section"]
            ))

            story.append(chart_flow(
                make_chart(history, "price")
            ))

        story.append(Paragraph(
            "Analyst Verification Checklist",
            s["section"]
        ))

        checklist = [
            "Verify the latest primary-source disclosure.",
            "Confirm reporting period, currency and units.",
            "Separate reported facts from analyst assumptions.",
            "Document missing information and unresolved inconsistencies.",
        ]

        for item in checklist:
            story.append(Paragraph(
                "• " + html.escape(item),
                s["body"]
            ))

        story.append(Paragraph(
            f"<font color='#64748B'>"
            f"Research section: {i:02d} | "
            f"Retrieved: {html.escape(retrieved)}"
            f"</font>",
            s["small"]
        ))

        story.append(PageBreak())

    # -----------------------------------------------------
    # FINANCIAL APPENDICES
    # -----------------------------------------------------

    if report_id == "financial_valuation_analysis":
        story.append(Paragraph(
            "Financial Statements — Income Statement",
            s["chapter"]
        ))

        story.append(make_table(
            statement_rows(data.get("financials")),
            [48*mm, 109*mm],
            6.5,
            accent=accent
        ))

        story.append(PageBreak())

        story.append(Paragraph(
            "Financial Statements — Balance Sheet",
            s["chapter"]
        ))

        story.append(make_table(
            statement_rows(data.get("balance_sheet")),
            [48*mm, 109*mm],
            6.5,
            accent=accent
        ))

        story.append(PageBreak())

        story.append(Paragraph(
            "Financial Statements — Cash Flow",
            s["chapter"]
        ))

        story.append(make_table(
            statement_rows(data.get("cashflow")),
            [48*mm, 109*mm],
            6.5,
            accent=accent
        ))

        story.append(PageBreak())

        add_financial_charts(
            story,
            data,
            accent
        )

    # -----------------------------------------------------
    # OWNERSHIP APPENDIX
    # -----------------------------------------------------

    if report_id == "market_risk_intelligence":
        story.append(Paragraph(
            "Institutional Ownership Data",
            s["chapter"]
        ))

        story.append(make_table(
            holder_rows(data.get("holders")),
            [50*mm, 107*mm],
            6.5,
            accent=accent
        ))

        story.append(PageBreak())

        # Risk indicator table
        story.append(Paragraph(
            "Risk Indicator Snapshot",
            s["chapter"]
        ))

        if risks:
            rows = [["Indicator", "Retrieved assessment"]]

            for item in risks[:25]:
                if isinstance(item, dict):
                    values = list(item.values())

                    rows.append([
                        val(values[0] if values else "Indicator"),
                        " | ".join(
                            val(x) for x in values[1:]
                        ) if len(values) > 1 else "No detail"
                    ])

            story.append(make_table(
                rows,
                [55*mm, 102*mm],
                7,
                accent=accent
            ))

            for item in risks[:10]:
                if not isinstance(item, dict):
                    continue

                severity = (
                    item.get("severity")
                    or item.get("status")
                    or item.get("level")
                )

                if severity:
                    story.append(Spacer(1, 3*mm))
                    story.append(signal_table(
                        severity,
                        item.get("description", item.get("detail", ""))
                    ))

        else:
            story.append(Paragraph(
                "No risk indicator records were provided.",
                s["body"]
            ))

        story.append(PageBreak())

    # -----------------------------------------------------
    # SOURCES AND LIMITATIONS
    # -----------------------------------------------------

    story.append(Paragraph(
        "Sources, Methodology & Limitations",
        s["chapter"]
    ))

    notices = [
        f"<b>Data provider:</b> {html.escape(val(data.get('source')))}",

        f"<b>Retrieval timestamp:</b> {html.escape(retrieved)}",

        (
            "<b>Market data:</b> Historical provider data may be delayed, "
            "incomplete or revised. It is not guaranteed to be real-time."
        ),

        (
            "<b>Financial data:</b> Reconcile all financial values and "
            "fiscal periods with audited company filings."
        ),

        (
            "<b>SEC data:</b> Direct SEC EDGAR filing analysis is not "
            "connected in this application version. No filing-specific "
            "conclusion is claimed unless filing data is separately supplied."
        ),

        (
            "<b>Supply chain:</b> Supplier names, factory locations, "
            "logistics routes and critical dependencies require "
            "independent verification."
        ),

        (
            "<b>Valuation:</b> No intrinsic value or target price is "
            "asserted without a documented and sourced valuation model."
        ),

        (
            "<b>Stress testing:</b> Scenarios are hypothetical sensitivities, "
            "not predictions or probability estimates."
        ),

        (
            "<b>Risk signals:</b> Colour signals are displayed only when "
            "a source-provided risk status or severity field is available. "
            "They are not generated from unsupported assumptions."
        ),

        (
            "<b>Important notice:</b> This report is for informational "
            "and research purposes only. It is not investment, legal "
            "or tax advice."
        ),
    ]

    for notice in notices:
        story.append(Paragraph(
            notice,
            s["body"]
        ))

    # -----------------------------------------------------
    # BUILD PDF
    # -----------------------------------------------------

    document.build(
        story,
        onFirstPage=lambda c, d: draw_page(
            c, d, logo_path, accent
        ),
        onLaterPages=lambda c, d: draw_page(
            c, d, logo_path, accent
        ),
    )

    return buffer.getvalue()


# =========================================================
# PUBLIC FUNCTION CALLED BY main.py
# =========================================================

def build_three_reports(
    profile,
    company_data,
    financial_metrics=None,
    market_metrics=None,
    risks=None,
    logo_path=None,
):
    results = {}

    for report_id, spec in REPORTS.items():
        pdf = build_one_report(
            report_id=report_id,
            profile=profile,
            data=company_data,
            metrics=financial_metrics,
            market=market_metrics,
            risks=risks,
            logo_path=logo_path,
        )

        ticker = safe_filename(
            profile.get("ticker", "COMPANY")
        )

        report_name = safe_filename(spec["title"])

        filename = f"{ticker}_{report_name}.pdf"

        results[filename] = pdf

    return results


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

def build_company_pdf(profile, metrics, source, retrieved_at):
    data = {
        "source": source,
        "retrieved_at": retrieved_at,
        "history": pd.DataFrame(),
        "financials": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "holders": pd.DataFrame(),
    }

    return build_one_report(
        "company_strategic_intelligence",
        profile,
        data,
        metrics=metrics
    )