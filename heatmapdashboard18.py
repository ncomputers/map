"""BTCUSDT Liquidity Heatmap Dashboard (version 18)

Features:
- Fetch level 1-5 order book from Binance with ccxt
- Store snapshots in Redis and persist liquidity ≥ threshold until removed
- Render heatmap using four-stop color scale with optional Gaussian smoothing
- Overlay white-outlined semi-transparent candlesticks
- Right-hand volume profile (red above price, green below)
- Dashed current price line with annotation
- Dark theme with grid lines and right-side price labels
- Timeframe, color scale, and zoom controls
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Any, List

import ccxt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import redis

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None  # type: ignore

EXCHANGE = ccxt.binance()
REDIS_KEY = "btc_heatmap_history"
SNAPSHOT_LIMIT = 300
PRICE_STEP = 1.0
LIQ_THRESHOLD = 5.0

r = redis.Redis(host="localhost", port=6379, db=0)


def fetch_and_store_snapshot() -> None:
    """Fetch order book and save to redis."""
    try:
        ob = EXCHANGE.fetch_order_book("BTC/USDT", limit=5)
    except Exception as e:
        print("Failed to fetch order book:", e)
        return
    snap = {
        "time": datetime.now(timezone.utc).isoformat(),
        "bids": ob.get("bids", []),
        "asks": ob.get("asks", []),
    }
    r.rpush(REDIS_KEY, json.dumps(snap))
    r.ltrim(REDIS_KEY, -SNAPSHOT_LIMIT, -1)


def load_history() -> List[Dict[str, Any]]:
    data = r.lrange(REDIS_KEY, 0, -1)
    out: List[Dict[str, Any]] = []
    for item in data:
        try:
            rec = json.loads(item)
            if isinstance(rec, dict):
                out.append(rec)
        except json.JSONDecodeError:
            continue
    return out


def build_matrix(history: List[Dict[str, Any]]):
    if not history:
        return [], [], np.array([[]])

    times = [pd.to_datetime(h["time"]) for h in history]
    price_levels = sorted(
        {round(p / PRICE_STEP) * PRICE_STEP for h in history for p, _ in h.get("bids", []) + h.get("asks", [])}
    )
    idx = {p: i for i, p in enumerate(price_levels)}
    heat = np.zeros((len(price_levels), len(times)))

    active: Dict[float, float] = {}
    for j, snap in enumerate(history):
        # clear levels not present
        present = {round(p / PRICE_STEP) * PRICE_STEP for p, _ in snap.get("bids", []) + snap.get("asks", [])}
        for lvl in list(active.keys()):
            if lvl not in present:
                del active[lvl]
        # update with current snapshot
        for p, v in snap.get("bids", []) + snap.get("asks", []):
            lvl = round(p / PRICE_STEP) * PRICE_STEP
            if v >= LIQ_THRESHOLD:
                active[lvl] = v
            elif lvl in active:
                del active[lvl]
        for lvl, i in idx.items():
            heat[i, j] = active.get(lvl, 0.0)
    return price_levels, times, heat


def build_candles(history: List[Dict[str, Any]], rule: str) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = pd.DataFrame(
        {
            "time": [pd.to_datetime(h["time"]) for h in history],
            "mid": [((h.get("bids", [[0, 0]])[0][0] + h.get("asks", [[0, 0]])[0][0]) / 2) for h in history],
        }
    ).set_index("time")

    candles = df["mid"].resample(rule).ohlc()
    candles.dropna(inplace=True)
    return candles.reset_index()


FOUR_STOP = [
    (0.0, "#000000"),
    (0.4, "#00008b"),
    (0.7, "#ffff00"),
    (1.0, "#ffffff"),
]
PURPLE_SCALE = [
    (0.0, "#330066"),
    (0.6, "#6600cc"),
    (0.85, "#ffcc00"),
    (1.0, "#ffffff"),
]

TIMEFRAMES = {
    "1s": "1s",
    "5s": "5s",
    "15s": "15s",
    "30s": "30s",
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
ZOOM_PRESETS = {
    "All": None,
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
}
COLOR_SCALES = {
    "four-stop": FOUR_STOP,
    "purple": PURPLE_SCALE,
}

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H2("BTCUSDT Liquidity Heatmap", style={"color": "white"}),
        html.Div(
            [
                dcc.Dropdown(
                    id="tf",
                    options=[{"label": k, "value": v} for k, v in TIMEFRAMES.items()],
                    value="1min",
                    clearable=False,
                    style={"width": "120px"},
                ),
                dcc.Dropdown(
                    id="colorscale",
                    options=[{"label": k, "value": k} for k in COLOR_SCALES.keys()],
                    value="four-stop",
                    clearable=False,
                    style={"width": "120px", "marginLeft": "10px"},
                ),
                html.Div(
                    dcc.Slider(
                        id="smooth",
                        min=0,
                        max=2,
                        step=1,
                        value=0,
                        marks={0: "no smooth", 1: "1", 2: "2"},
                        tooltip={"placement": "bottom"},
                    ),
                    style={"width": "200px", "marginLeft": "10px"},
                ),
                dcc.RadioItems(
                    id="zoom",
                    options=[{"label": k, "value": k} for k in ZOOM_PRESETS.keys()],
                    value="All",
                    labelStyle={"display": "inline-block", "marginRight": "10px"},
                    style={"marginLeft": "10px"},
                ),
            ],
            style={"display": "flex", "alignItems": "center"},
        ),
        dcc.Interval(id="timer", interval=5000, n_intervals=0),
        dcc.Graph(id="chart", style={"height": "80vh"}),
    ],
    style={"backgroundColor": "#000", "color": "white", "padding": "10px"},
)


@app.callback(Output("chart", "figure"), [Input("timer", "n_intervals"), Input("tf", "value"), Input("colorscale", "value"), Input("smooth", "value"), Input("zoom", "value")])
def update_chart(n, tf, cs_name, smooth, zoom):
    fetch_and_store_snapshot()
    fig = make_figure(tf, COLOR_SCALES[cs_name], smooth)
    if zoom != "All" and fig.data:
        delta = ZOOM_PRESETS[zoom]
        end = fig.data[0].x[-1]
        start = end - delta
        fig.update_xaxes(range=[start, end])
    return fig


def make_figure(tf_rule: str, colorscale, smooth: int) -> go.Figure:
    history = load_history()
    levels, times, heat = build_matrix(history)
    if smooth > 0 and gaussian_filter is not None and heat.size > 0:
        heat = gaussian_filter(heat, sigma=smooth)
    candles = build_candles(history, tf_rule)

    if not levels or heat.size == 0:
        return go.Figure()

    vol_profile = np.sum(heat, axis=1)
    last_price = candles["close"].iloc[-1] if not candles.empty else levels[len(levels) // 2]
    colors = ["red" if lvl >= last_price else "green" for lvl in levels]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "bar"}]],
        column_widths=[0.8, 0.2],
        shared_yaxes=True,
        horizontal_spacing=0.02,
    )

    fig.add_trace(
        go.Heatmap(
            z=heat,
            x=times,
            y=levels,
            colorscale=colorscale,
            zmin=0,
            zmax=float(heat.max()) if heat.size > 0 else 1,
            showscale=False,
        ),
        row=1,
        col=1,
    )

    if not candles.empty:
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
                name="Price",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=[times[0], times[-1]],
            y=[last_price, last_price],
            mode="lines",
            line=dict(color="white", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            y=levels,
            x=vol_profile,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=700,
        plot_bgcolor="#000",
        paper_bgcolor="#000",
        font=dict(color="white"),
        margin=dict(l=60, r=60, t=40, b=40),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=True, gridcolor="#444", ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#444", side="right", ticks="outside")

    fig.add_annotation(
        x=times[-1],
        y=last_price,
        text=f"{last_price:.2f}",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="white"),
        bgcolor="rgba(0,0,0,0.5)",
    )

    return fig


if __name__ == "__main__":
    app.run(debug=False, port=8051)
