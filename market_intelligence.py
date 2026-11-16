import pandas as pd
import plotly.graph_objects as go

LAYOUT = dict(
    paper_bgcolor="#000000", plot_bgcolor="#000000",
    font=dict(family="Inter, sans-serif", color="#AAB8CC", size=11),
    margin=dict(l=10,r=10,t=18,b=10),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(120,145,175,.12)", zeroline=False),
    hoverlabel=dict(bgcolor="#152338", font_color="white")
)

def price_figure(history: pd.DataFrame):
    fig = go.Figure()
    if history is not None and not history.empty and "Close" in history:
        fig.add_trace(go.Scatter(
            x=history.index, y=history["Close"], mode="lines",
            name="Adjusted close", line=dict(color="#4A9EFF", width=2),
            fill="tozeroy", fillcolor="rgba(74,158,255,.07)"
        ))
    fig.update_layout(**LAYOUT, height=330, showlegend=False)
    fig.update_yaxes(title_text="Price")
    return fig

def volume_figure(history: pd.DataFrame):
    fig = go.Figure()
    if history is not None and not history.empty and "Volume" in history:
        fig.add_trace(go.Bar(
            x=history.index, y=history["Volume"],
            marker_color="#397FC1", name="Volume"
        ))
    fig.update_layout(**LAYOUT, height=240, showlegend=False)
    return fig

def calculate_market_metrics(history: pd.DataFrame) -> dict:
    if history is None or history.empty or "Close" not in history:
        return {}
    close = history["Close"].dropna()
    if close.empty:
        return {}
    returns = close.pct_change().dropna()
    return {
        "latest_close": float(close.iloc[-1]),
        "period_change_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100) if len(close) > 1 else None,
        "annualized_volatility_pct": float(returns.std() * (252 ** 0.5) * 100) if len(returns) > 1 else None,
        "observations": int(len(close))
    }
