"""Dash BTCUSDT liquidity heatmap with candlestick overlay and timeframe selector."""

import json
import threading
import time
from datetime import datetime, timezone

import ccxt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import redis

# Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Heatmap parameters
STEP = 10
RANGE = 50
MAX_HISTORY = 500
SYMBOL = "BTC/USDT"

price_levels = None

TIMEFRAMES = [
    "1s",
    "5s",
    "15s",
    "30s",
    "1m",
    "5m",
    "15m",
    "1h",
    "4h",
    "1d",
]

TF_ALIAS = {
    "1s": "1S",
    "5s": "5S",
    "15s": "15S",
    "30s": "30S",
    "1m": "1T",
    "5m": "5T",
    "15m": "15T",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}


def fetch_orderbook_loop():
    """Fetch order book and store snapshots in Redis."""
    exchange = ccxt.binance()
    global price_levels
    while True:
        try:
            ob = exchange.fetch_order_book(SYMBOL.replace("/", ""))
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if not bids or not asks:
                raise ValueError("empty order book")
            bid_price = bids[0][0]
            ask_price = asks[0][0]
            mid_price = (bid_price + ask_price) / 2

            if price_levels is None:
                levels = [mid_price + STEP * i for i in range(-RANGE, RANGE + 1)]
                price_levels = levels
                redis_client.set("btc_price_levels", json.dumps(levels))
            else:
                levels = price_levels

            volume = [0.0 for _ in levels]
            for price, amount in bids + asks:
                idx = int(round((price - levels[0]) / STEP))
                if 0 <= idx < len(levels):
                    volume[idx] += amount

            timestamp = datetime.now(timezone.utc).isoformat()
            snap = [timestamp, mid_price] + volume
            redis_client.rpush("btc_heatmap_history", json.dumps(snap))
            redis_client.ltrim("btc_heatmap_history", -MAX_HISTORY, -1)
            redis_client.set("btc_last_bid", bid_price)
            redis_client.set("btc_last_ask", ask_price)
            redis_client.set("btc_last_price", mid_price)
        except Exception as exc:  # pragma: no cover - network errors not tested
            print("Fetch error:", exc)
        time.sleep(5)


def fetch_history():
    rows = [json.loads(x) for x in redis_client.lrange("btc_heatmap_history", 0, -1)]
    levels_raw = redis_client.get("btc_price_levels")
    levels = json.loads(levels_raw) if levels_raw else []
    return rows, levels


def resample_history(history, levels, tf: str):
    if not history or not levels:
        return pd.DataFrame()
    df = pd.DataFrame(history, columns=["time", "price"] + levels)
    df["time"] = pd.to_datetime(df["time"])
    rule = TF_ALIAS.get(tf, "1T")
    price_ohlc = df.set_index("time")["price"].resample(rule).ohlc()
    vol = df.set_index("time").iloc[:, 1:].resample(rule).mean()
    out = pd.concat([price_ohlc, vol], axis=1).dropna()
    return out.reset_index()


def make_figure(tf: str):
    history, levels = fetch_history()
    resampled = resample_history(history, levels, tf)
    if resampled.empty:
        return go.Figure()
    heat = resampled.iloc[:, 5:].values.T
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=heat, x=resampled["time"], y=levels, colorscale="Viridis", colorbar=dict(title="Volume"))
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
        html.H1("BTCUSDT Liquidity Heatmap"),
        dcc.Dropdown(
            id="timeframe",
            options=[{"label": tf, "value": tf} for tf in TIMEFRAMES],
            value="1m",
            clearable=False,
        ),
        dcc.Graph(id="heatmap"),
        html.Div(id="bid-ask"),
        dcc.Interval(id="interval", interval=5000, n_intervals=0),
    ]
)


@app.callback(
    [Output("heatmap", "figure"), Output("bid-ask", "children")],
    [Input("interval", "n_intervals"), Input("timeframe", "value")],
)
def update_dashboard(_, timeframe):
    fig = make_figure(timeframe)
    bid = redis_client.get("btc_last_bid")
    ask = redis_client.get("btc_last_ask")
    price = redis_client.get("btc_last_price")
    if bid and ask and price:
        info = f"Bid: {float(bid):.2f} Ask: {float(ask):.2f} Price: {float(price):.2f}"
    else:
        info = "No data"
    return fig, info


if __name__ == "__main__":
    thread = threading.Thread(target=fetch_orderbook_loop, daemon=True)
    thread.start()
    app.run(debug=False, port=8051)
