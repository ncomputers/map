"""BTCUSDT liquidity heatmap dashboard (version 11).

This script focuses purely on the frontend. Liquidity and candlestick data
are generated locally to mimic the look of CoinGlass without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ---------------------------
# Mock data
# ---------------------------
np.random.seed(1)
PRICE_LEVELS = np.arange(29900, 30101, 10)
TIMEFRAMES = {
    "1m": pd.date_range("2024-01-01", periods=40, freq="1min"),
    "5m": pd.date_range("2024-01-01", periods=40, freq="5min"),
}

HEATMAP: dict[str, np.ndarray] = {}
CANDLES: dict[str, pd.DataFrame] = {}

for tf, times in TIMEFRAMES.items():
    base = np.abs(np.random.randn(len(PRICE_LEVELS), len(times))) * 80
    # simple smoothing to avoid blocky look
    kernel = np.array([0.25, 0.5, 0.25])
    smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, base)
    price = 30000.0
    rows = []
    for t in times:
        open_p = price + np.random.randn() * 5
        close_p = open_p + np.random.randn() * 5
        high_p = max(open_p, close_p) + abs(np.random.randn()) * 2
        low_p = min(open_p, close_p) - abs(np.random.randn()) * 2
        rows.append((t, open_p, high_p, low_p, close_p))
        price = close_p
    HEATMAP[tf] = smooth
    CANDLES[tf] = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])


# ---------------------------
# Figure construction
# ---------------------------

def make_figure(tf: str) -> go.Figure:
    times = TIMEFRAMES[tf]
    heat = HEATMAP[tf]
    candles = CANDLES[tf]
    vol_profile = heat.mean(axis=1)
    strong_thresh = np.percentile(vol_profile, 85)
    last_price = candles["close"].iloc[-1]
    colors = ["red" if lvl >= last_price else "green" for lvl in PRICE_LEVELS]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.8, 0.2],
        horizontal_spacing=0.02,
        specs=[[{"type": "xy"}, {"type": "bar"}]],
    )

    fig.add_trace(
        go.Heatmap(
            z=heat,
            x=times,
            y=PRICE_LEVELS,
            colorscale=[(0, "#330066"), (0.5, "#9933cc"), (0.8, "#ffff66"), (1, "#ffffff")],
            zsmooth="best",
            zmin=0,
            zmax=heat.max(),
            showscale=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Candlestick(
            x=candles["time"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            increasing_line_color="white",
            decreasing_line_color="white",
            increasing_fillcolor="rgba(255,255,255,0.3)",
            decreasing_fillcolor="rgba(255,255,255,0.3)",
            line_width=1,
            showlegend=False,
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            y=PRICE_LEVELS,
            x=vol_profile,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # liquidity level lines
    for lvl, row in zip(PRICE_LEVELS, heat):
        mask = row >= strong_thresh
        if not mask.any():
            continue
        if mask.all():
            fig.add_shape(
                type="line",
                x0=times[0],
                x1=times[-1],
                y0=lvl,
                y1=lvl,
                line=dict(color="yellow", width=1, dash="solid"),
                xref="x1",
                yref="y1",
            )
        else:
            # dashed line across entire range
            fig.add_shape(
                type="line",
                x0=times[0],
                x1=times[-1],
                y0=lvl,
                y1=lvl,
                line=dict(color="yellow", width=1, dash="dash"),
                xref="x1",
                yref="y1",
            )

    fig.update_layout(
        height=700,
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="white"),
        margin=dict(l=60, r=40, t=40, b=40),
        hovermode="x",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis2_title="Volume",
        yaxis2_showticklabels=False,
        xaxis_rangeslider_visible=False,
    )
    return fig


# ---------------------------
# Dash app
# ---------------------------
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("BTCUSDT Liquidity Heatmap", style={"color": "white"}),
        dcc.Dropdown(
            id="timeframe",
            options=[{"label": tf, "value": tf} for tf in TIMEFRAMES],
            value="1m",
            clearable=False,
            style={"width": "120px", "color": "black"},
        ),
        dcc.Graph(id="chart", style={"height": "80vh", "width": "100%"}),
    ],
    style={"backgroundColor": "#000000", "color": "white", "padding": "10px"},
)


@app.callback(Output("chart", "figure"), Input("timeframe", "value"))
def update_chart(tf: str) -> go.Figure:
    return make_figure(tf)


if __name__ == "__main__":
    app.run(debug=False, port=8051)
