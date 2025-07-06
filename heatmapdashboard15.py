"""BTCUSDT liquidity heatmap dashboard (version 15).

This script fetches level 1-5 order book data from Binance using ccxt,
stores snapshots in Redis, and displays a heatmap with candlestick
overlay and volume profile. Liquidity lines persist only while an
order above the threshold remains in the book, replicating CoinGlass
behavior.
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

# Exchange and redis configuration
EXCHANGE = ccxt.binance()
REDIS_KEY = "btc_heatmap_history"
SNAPSHOT_LIMIT = 300  # keep latest snapshots
PRICE_STEP = 1.0  # USDT step for y axis
LIQ_THRESHOLD = 5.0  # minimum size to draw liquidity

r = redis.Redis(host="localhost", port=6379, db=0)


def fetch_and_store_snapshot() -> None:
    """Fetch top 5 levels and store snapshot in redis."""
    try:
        ob = EXCHANGE.fetch_order_book("BTC/USDT", limit=5)
    except Exception as e:
        print("Failed to fetch order book:", e)
        return
    now = datetime.now(timezone.utc).isoformat()
    snap = {"time": now, "bids": ob.get("bids", []), "asks": ob.get("asks", [])}
    r.rpush(REDIS_KEY, json.dumps(snap))
    r.ltrim(REDIS_KEY, -SNAPSHOT_LIMIT, -1)


def load_history() -> List[Dict[str, Any]]:
    """Load snapshots from redis."""
    data = r.lrange(REDIS_KEY, 0, -1)
    history: List[Dict[str, Any]] = []
    for item in data:
        try:
            rec = json.loads(item)
            if isinstance(rec, dict):
                history.append(rec)
        except json.JSONDecodeError:
            continue
    return history


def build_matrix(history: List[Dict[str, Any]]):
    """Create price levels, timestamps and heat matrix without smoothing."""
    if not history:
        return [], [], np.array([[]])

    times = [pd.to_datetime(s.get("time")) for s in history]
    price_levels = set()
    for snap in history:
        for p, _ in snap.get("bids", []) + snap.get("asks", []):
            price_levels.add(round(p / PRICE_STEP) * PRICE_STEP)
    levels = sorted(price_levels)
    idx = {p: i for i, p in enumerate(levels)}
    heat = np.zeros((len(levels), len(times)))

    for j, snap in enumerate(history):
        for p, v in snap.get("bids", []) + snap.get("asks", []):
            lvl = round(p / PRICE_STEP) * PRICE_STEP
            if v >= LIQ_THRESHOLD:
                i = idx.get(lvl)
                if i is not None:
                    heat[i, j] = v
    return levels, times, heat


def build_candles(history: List[Dict[str, Any]], rule: str) -> pd.DataFrame:
    """Create OHLC candles from mid price of snapshots."""
    if not history:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = pd.DataFrame({
        "time": [pd.to_datetime(s.get("time")) for s in history],
        "mid": [((s.get("bids", [[0,0]])[0][0] + s.get("asks", [[0,0]])[0][0]) / 2) for s in history]
    }).set_index("time")

    candles = df["mid"].resample(rule).ohlc()
    candles.dropna(inplace=True)
    return candles.reset_index()


def make_figure(tf_rule: str) -> go.Figure:
    history = load_history()
    levels, times, heat = build_matrix(history)
    candles = build_candles(history, tf_rule)

    if len(levels) == 0 or heat.size == 0:
        return go.Figure()

    vol_profile = np.sum(heat, axis=1)
    last_price = candles["close"].iloc[-1] if not candles.empty else levels[len(levels)//2]
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
            colorscale=[(0.0, "#330066"), (0.6, "#6600cc"), (0.85, "#ffcc00"), (1.0, "#ffffff")],
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
                increasing_fillcolor="rgba(255,255,255,0.2)",
                decreasing_fillcolor="rgba(255,255,255,0.2)",
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

    fig.update_layout(
        height=700,
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="white"),
        margin=dict(l=60, r=40, t=40, b=40),
        hovermode="x",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis=dict(tickformat="%H:%M:%S"),
        yaxis=dict(tickformat=".0f", fixedrange=True),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(showgrid=False)
    return fig


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

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("BTCUSDT Liquidity Heatmap", style={"color": "white"}),
        dcc.Dropdown(
            id="tf",
            options=[{"label": k, "value": v} for k, v in TIMEFRAMES.items()],
            value="1min",
            clearable=False,
            style={"width": "120px", "color": "black"},
        ),
        dcc.Interval(id="timer", interval=5000, n_intervals=0),
        dcc.Graph(id="chart", style={"height": "80vh", "width": "100%"}),
    ],
    style={"backgroundColor": "#000000", "color": "white", "padding": "10px"},
)


@app.callback(Output("chart", "figure"), [Input("timer", "n_intervals"), Input("tf", "value")])
def update_chart(n: int, tf: str) -> go.Figure:
    fetch_and_store_snapshot()
    return make_figure(tf)


if __name__ == "__main__":
    app.run(debug=False, port=8051)
