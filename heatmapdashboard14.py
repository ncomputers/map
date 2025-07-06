"""BTCUSDT liquidity heatmap dashboard (version 14).

This script fetches BTCUSDT order book data from Binance using ccxt,
stores snapshots in Redis, and displays a heatmap with candlestick
overlay and a volume profile. The UI mimics CoinGlass with bright
liquidity lines and detailed candles.
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

# Configuration
EXCHANGE = ccxt.binance()
REDIS_KEY = "btc_heatmap_history"
SNAPSHOT_LIMIT = 200
PRICE_STEP = 5.0  # price increment for heatmap levels

r = redis.Redis(host="localhost", port=6379, db=0)


def fetch_and_store_snapshot() -> None:
    """Fetch order book from Binance and store the snapshot in Redis."""
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
    """Load all snapshots from Redis."""
    data = r.lrange(REDIS_KEY, 0, -1)
    history: List[Dict[str, Any]] = []
    for item in data:
        try:
            record = json.loads(item)
            if isinstance(record, dict):
                history.append(record)
        except json.JSONDecodeError:
            continue
    return history


def build_heatmap(history: List[Dict[str, Any]]):
    """Return price levels, timestamps, and heat matrix."""
    if not history:
        return [], [], np.array([[]])

    times = [pd.to_datetime(s.get("time")) for s in history]
    price_set = set()
    for s in history:
        for p, _ in s.get("bids", []) + s.get("asks", []):
            price_set.add(round(p / PRICE_STEP) * PRICE_STEP)
    levels = sorted(price_set)
    idx = {p: i for i, p in enumerate(levels)}
    heat = np.zeros((len(levels), len(times)))
    for j, s in enumerate(history):
        for p, v in s.get("bids", []):
            i = idx.get(round(p / PRICE_STEP) * PRICE_STEP)
            if i is not None:
                heat[i, j] += v
        for p, v in s.get("asks", []):
            i = idx.get(round(p / PRICE_STEP) * PRICE_STEP)
            if i is not None:
                heat[i, j] += v
    return levels, times, heat


def build_candles(history: List[Dict[str, Any]], rule: str) -> pd.DataFrame:
    """Build OHLC candles from mid-price of snapshots."""
    if not history:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = pd.DataFrame(
        {
            "time": [pd.to_datetime(s.get("time")) for s in history],
            "mid": [
                ((s.get("bids", [[0, 0]])[0][0] + s.get("asks", [[0, 0]])[0][0]) / 2)
                for s in history
            ],
        }
    ).set_index("time")

    ohlc = df["mid"].resample(rule).ohlc()
    ohlc.dropna(inplace=True)
    return ohlc.reset_index()


def make_figure(tf_rule: str) -> go.Figure:
    history = load_history()
    levels, times, heat = build_heatmap(history)
    candles = build_candles(history, tf_rule)

    if len(levels) == 0 or heat.size == 0:
        return go.Figure()

    vol_profile = heat.mean(axis=1)
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
            colorscale=[
                (0.0, "#330066"),
                (0.6, "#6600cc"),
                (0.85, "#ffcc00"),
                (1.0, "#ffffff"),
            ],
            zmin=0,
            zmax=float(heat.max()),
            zsmooth="best",
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

    strong_thresh = np.percentile(vol_profile, 90)
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
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickformat="%H:%M:%S"),
        yaxis=dict(tickformat=".0f"),
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
def update(n: int, tf: str) -> go.Figure:
    fetch_and_store_snapshot()
    return make_figure(tf)


if __name__ == "__main__":
    app.run(debug=False, port=8051)
