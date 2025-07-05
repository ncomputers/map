"""BTCUSDT liquidity heatmap with candlesticks and volume profile.

This version adds price overlay, highlighted liquidity bands, a darker theme,
and a right-side volume profile similar to CoinGlass. Data updates every few
seconds from Binance (with a fallback if requests fail).
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import redis

try:  # pragma: no cover - ccxt may be missing or network blocked
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dep
    ccxt = None

# Redis setup
r = redis.Redis(host="localhost", port=6379, db=0)

STEP = 0.5  # price increment in USD
RANGE = 100  # number of steps above and below
MAX_HISTORY = 500
SYMBOL = "BTC/USDT"

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

price_levels: list[float] | None = None


def fetch_loop() -> None:
    """Fetch order book data periodically and store snapshots in Redis."""
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
        except Exception as exc:
            print("Fetch error:", exc)
            mid = float(r.get("btc_last_price") or 20000)
            bids = [[mid - STEP * i, 1.0] for i in range(1, RANGE)]
            asks = [[mid + STEP * i, 1.0] for i in range(1, RANGE)]

        bid_price = bids[0][0]
        ask_price = asks[0][0]
        mid_price = (bid_price + ask_price) / 2

        if price_levels is None:
            price_levels = [mid_price + STEP * i for i in range(-RANGE, RANGE + 1)]
            r.set("btc_price_levels", json.dumps(price_levels))
        levels = price_levels

        volume = [0.0 for _ in levels]
        for price, amount in bids + asks:
            idx = int(round((price - levels[0]) / STEP))
            if 0 <= idx < len(levels):
                volume[idx] += amount

        ts = datetime.now(timezone.utc).isoformat()
        snap = [ts, mid_price] + volume
        r.rpush("btc_heatmap_history", json.dumps(snap))
        r.ltrim("btc_heatmap_history", -MAX_HISTORY, -1)
        r.set("btc_last_bid", bid_price)
        r.set("btc_last_ask", ask_price)
        r.set("btc_last_price", mid_price)
        time.sleep(5)


def fetch_history() -> tuple[list[list], list[float]]:
    rows = [json.loads(x) for x in r.lrange("btc_heatmap_history", 0, -1)]
    levels_raw = r.get("btc_price_levels")
    levels = json.loads(levels_raw) if levels_raw else []
    return rows, levels


def resample_history(history: list[list], levels: list[float], tf: str) -> pd.DataFrame:
    if not history or not levels:
        return pd.DataFrame()
    cols = ["time", "price"] + [f"v{i}" for i in range(len(levels))]
    df = pd.DataFrame(history, columns=cols)
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
    vol_profile = heat.mean(axis=1)
    mid = resampled["close"].iloc[-1]
    colors = ["red" if lvl > mid else "green" for lvl in levels]
    strong_thresh = np.percentile(vol_profile, 90) if len(vol_profile) else 0
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.8, 0.2],
        shared_yaxes=True,
        specs=[[{"type": "xy"}, {"type": "bar"}]],
        horizontal_spacing=0.05,
    )
    fig.add_trace(
        go.Heatmap(
            z=heat,
            x=resampled["time"],
            y=levels,
            colorscale=[[0, "rgb(0,0,60)"], [0.5, "purple"], [1, "yellow"]],
            showscale=False,
        ),
        row=1,
        col=1,
    )
    # highlight strong liquidity bands
    for lvl, vol in zip(levels, vol_profile):
        if vol >= strong_thresh:
            fig.add_shape(
                type="line",
                x0=resampled["time"].iloc[0],
                x1=resampled["time"].iloc[-1],
                y0=lvl,
                y1=lvl,
                line=dict(color="yellow", width=1),
                xref="x",
                yref="y",
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
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis2_title="Volume",
        yaxis2_showticklabels=False,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=60, r=60, t=40, b=40),
    )
    return fig


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("BTCUSDT Liquidity Heatmap"),
        dcc.Dropdown(id="timeframe", options=[{"label": tf, "value": tf} for tf in TIMEFRAMES], value="1m", clearable=False),
        dcc.Graph(id="heatmap", style={"width": "100%", "height": "80vh"}, config={"scrollZoom": True}),
        html.Div(id="bid-ask"),
        dcc.Interval(id="interval", interval=5000, n_intervals=0),
    ]
)


@app.callback(
    Output("heatmap", "figure"),
    Output("bid-ask", "children"),
    Input("interval", "n_intervals"),
    Input("timeframe", "value"),
)
def update_dashboard(_, timeframe):
    fig = make_figure(timeframe)
    bid = r.get("btc_last_bid")
    ask = r.get("btc_last_ask")
    price = r.get("btc_last_price")
    if bid and ask and price:
        info = f"Bid: {float(bid):.2f}  Ask: {float(ask):.2f}  Price: {float(price):.2f}"
    else:
        info = "No data"
    return fig, info


if __name__ == "__main__":
    thread = threading.Thread(target=fetch_loop, daemon=True)
    thread.start()
    app.run(debug=False, port=8051)
