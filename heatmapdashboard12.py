"""BTCUSDT liquidity heatmap dashboard (version 12).

This script fetches real order book data from Binance using ccxt and stores
snapshots in Redis for historical visualization. Candlestick price data is
derived from the recorded mid prices so the heatmap and candles share the
same timeframe.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Dict, Any

import ccxt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import redis

# ---------------------------
# Configuration
# ---------------------------
EXCHANGE = ccxt.binance()
REDIS_KEY = "btc_heatmap_history"
SNAPSHOT_LIMIT = 200  # number of snapshots to keep
PRICE_STEP = 5.0  # USDT

r = redis.Redis(host="localhost", port=6379, db=0)

# ---------------------------
# Data functions
# ---------------------------

def fetch_and_store_snapshot() -> None:
    """Fetch order book from Binance and store in Redis."""
    try:
        ob = EXCHANGE.fetch_order_book("BTC/USDT", limit=100)
    except Exception as e:
        print("Failed to fetch order book:", e)
        return

    now = datetime.now(timezone.utc).isoformat()
    snap = {"time": now, "bids": ob.get("bids", []), "asks": ob.get("asks", [])}
    r.rpush(REDIS_KEY, json.dumps(snap))
    r.ltrim(REDIS_KEY, -SNAPSHOT_LIMIT, -1)


def load_history() -> List[Dict[str, Any]]:
    data = r.lrange(REDIS_KEY, 0, -1)
    return [json.loads(x) for x in data]


def build_heatmap(history: List[Dict[str, Any]]):
    if not history:
        return [], [], np.array([[]])
    times = [pd.to_datetime(s["time"]) for s in history]
    # collect all price levels rounded by PRICE_STEP
    price_set = set()
    for s in history:
        for p, v in s.get("bids", []) + s.get("asks", []):
            price_set.add(round(p / PRICE_STEP) * PRICE_STEP)
    levels = sorted(price_set)
    lvl_index = {p: i for i, p in enumerate(levels)}
    heat = np.zeros((len(levels), len(times)))
    for j, s in enumerate(history):
        for p, v in s.get("bids", []):
            idx = lvl_index.get(round(p / PRICE_STEP) * PRICE_STEP)
            if idx is not None:
                heat[idx, j] += v
        for p, v in s.get("asks", []):
            idx = lvl_index.get(round(p / PRICE_STEP) * PRICE_STEP)
            if idx is not None:
                heat[idx, j] += v
    return levels, times, heat


def build_candles(history: List[Dict[str, Any]], timeframe: str) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    df = pd.DataFrame(
        {
            "time": [pd.to_datetime(s["time"]) for s in history],
            "mid": [
                ((s.get("bids", [[0, 0]])[0][0] + s.get("asks", [[0, 0]])[0][0]) / 2)
                for s in history
            ],
        }
    ).set_index("time")
    ohlc = df["mid"].resample(timeframe).ohlc()
    ohlc.dropna(inplace=True)
    return ohlc.reset_index()


# ---------------------------
# Figure construction
# ---------------------------

def make_figure(tf: str) -> go.Figure:
    history = load_history()
    levels, times, heat = build_heatmap(history)
    candles = build_candles(history, tf)
    if len(levels) == 0 or heat.size == 0:
        return go.Figure()
    vol_profile = heat.mean(axis=1)
    last_price = candles["close"].iloc[-1] if not candles.empty else levels[len(levels)//2]
    colors = ["red" if lvl >= last_price else "green" for lvl in levels]
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
            y=levels,
            colorscale=[(0, "#330066"), (0.5, "#9933cc"), (0.8, "#ffff66"), (1, "#ffffff")],
            zsmooth="best",
            zmin=0,
            zmax=heat.max(),
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
                showlegend=False,
                name="Price",
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
    strong_thresh = np.percentile(vol_profile, 85)
    for lvl, row in zip(levels, heat):
        mask = row >= strong_thresh
        if not mask.any():
            continue
        dash_style = "solid" if mask.all() else "dash"
        fig.add_shape(
            type="line",
            x0=times[0],
            x1=times[-1],
            y0=lvl,
            y1=lvl,
            line=dict(color="yellow", width=1, dash=dash_style),
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

TIMEFRAMES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "5s": "5s",
    "1s": "1s",
    "15s": "15s",
    "30s": "30s",
}

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("BTCUSDT Liquidity Heatmap", style={"color": "white"}),
        dcc.Dropdown(
            id="timeframe",
            options=[{"label": k, "value": v} for k, v in TIMEFRAMES.items()],
            value="1min",
            clearable=False,
            style={"width": "120px", "color": "black"},
        ),
        dcc.Interval(id="interval", interval=5000, n_intervals=0),
        dcc.Graph(id="chart", style={"height": "80vh", "width": "100%"}),
    ],
    style={"backgroundColor": "#000000", "color": "white", "padding": "10px"},
)


@app.callback(Output("chart", "figure"), [Input("interval", "n_intervals"), Input("timeframe", "value")])
def update_chart(n: int, tf: str) -> go.Figure:
    fetch_and_store_snapshot()
    return make_figure(tf)


if __name__ == "__main__":
    app.run(debug=False, port=8051)
