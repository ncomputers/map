"""Dash XAU/USDT liquidity heatmap with candlestick overlay.

This script fetches the Binance order book for XAU/USDT using ``ccxt``. Each
snapshot is stored in Redis so that historical liquidity can be visualised as a
heatmap. A candlestick trace of the mid price is overlaid on top of the
heatmap. A dropdown allows selection of resampling timeframes.

Running this file launches a Dash application with ``app.run`` on port 8051.
"""

from __future__ import annotations

import json
import random
import threading
import time
from datetime import datetime, timezone

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import redis

try:  # pragma: no cover - ccxt may be missing
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dep
    ccxt = None

# Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Heatmap parameters
STEP = 0.5  # price increment in USD
RANGE = 100  # number of steps above/below mid price
MAX_HISTORY = 500
SYMBOL = "XAU/USDT"

price_levels: list[float] | None = None

TIMEFRAMES = ["1s", "5s", "15s", "30s", "1m", "5m", "15m", "1h", "4h", "1d"]
TF_RULES = {
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


def fetch_orderbook_loop() -> None:
    """Continuously fetch order book data and store snapshots in Redis."""
    exchange = ccxt.binance() if ccxt else None
    global price_levels
    while True:
        try:
            if exchange:
                ob = exchange.fetch_order_book(SYMBOL)
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if not bids or not asks:
                    raise ValueError("empty order book")
            else:  # pragma: no cover - network may be blocked
                raise RuntimeError("ccxt not available")
        except Exception as exc:  # noqa: PERF203
            print("Fetch error:", exc)
            mid = float(redis_client.get("xau_last_price") or 2000)
            bids = [[mid - STEP * i, random.uniform(0.1, 1)] for i in range(1, RANGE)]
            asks = [[mid + STEP * i, random.uniform(0.1, 1)] for i in range(1, RANGE)]

        bid_price = bids[0][0]
        ask_price = asks[0][0]
        mid_price = (bid_price + ask_price) / 2

        if price_levels is None:
            price_levels = [mid_price + STEP * i for i in range(-RANGE, RANGE + 1)]
            redis_client.set("xau_price_levels", json.dumps(price_levels))
        levels = price_levels

        volume = [0.0 for _ in levels]
        for price, amount in bids + asks:
            idx = int(round((price - levels[0]) / STEP))
            if 0 <= idx < len(levels):
                volume[idx] += amount

        ts = datetime.now(timezone.utc).isoformat()
        snap = [ts, mid_price] + volume
        redis_client.rpush("xau_heatmap_history", json.dumps(snap))
        redis_client.ltrim("xau_heatmap_history", -MAX_HISTORY, -1)
        redis_client.set("xau_last_bid", bid_price)
        redis_client.set("xau_last_ask", ask_price)
        redis_client.set("xau_last_price", mid_price)
        time.sleep(5)


def fetch_history() -> tuple[list[list], list[float]]:
    rows = [json.loads(x) for x in redis_client.lrange("xau_heatmap_history", 0, -1)]
    levels_raw = redis_client.get("xau_price_levels")
    levels = json.loads(levels_raw) if levels_raw else []
    return rows, levels


def resample_history(history: list[list], levels: list[float], tf: str) -> pd.DataFrame:
    if not history or not levels:
        return pd.DataFrame()
    df = pd.DataFrame(history, columns=["time", "price"] + levels)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    rule = TF_RULES.get(tf, "1min")
    price_ohlc = df.set_index("time")["price"].resample(rule).ohlc()
    vol = df.set_index("time").iloc[:, 1:].resample(rule).mean()
    out = pd.concat([price_ohlc, vol], axis=1).dropna()
    return out.reset_index()


def make_figure(tf: str) -> go.Figure:
    history, levels = fetch_history()
    resampled = resample_history(history, levels, tf)
    if resampled.empty:
        return go.Figure()
    heat = resampled.iloc[:, 5:].values.T
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=heat,
            x=resampled["time"],
            y=levels,
            colorscale="Viridis",
            colorbar=dict(title="Volume"),
        )
    )
    fig.add_trace(
        go.Candlestick(
            x=resampled["time"],
            open=resampled["open"],
            high=resampled["high"],
            low=resampled["low"],
            close=resampled["close"],
            name="Price",
            increasing_line_color="white",
            decreasing_line_color="white",
        )
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Price (USDT)")
    return fig


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("XAU/USDT Liquidity Heatmap"),
        dcc.Dropdown(id="timeframe", options=[{"label": tf, "value": tf} for tf in TIMEFRAMES], value="1m", clearable=False),
        dcc.Graph(id="heatmap"),
        html.Div(id="bid-ask"),
        dcc.Interval(id="interval", interval=5000, n_intervals=0),
    ]
)


@app.callback(Output("heatmap", "figure"), Output("bid-ask", "children"), Input("interval", "n_intervals"), Input("timeframe", "value"))
def update_dashboard(_, timeframe):
    fig = make_figure(timeframe)
    bid = redis_client.get("xau_last_bid")
    ask = redis_client.get("xau_last_ask")
    price = redis_client.get("xau_last_price")
    if bid and ask and price:
        info = f"Bid: {float(bid):.2f}  Ask: {float(ask):.2f}  Price: {float(price):.2f}"
    else:
        info = "No data"
    return fig, info


if __name__ == "__main__":
    thread = threading.Thread(target=fetch_orderbook_loop, daemon=True)
    thread.start()
    app.run(debug=False, port=8051)
