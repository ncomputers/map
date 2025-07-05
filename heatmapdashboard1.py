"""Dash application to display an XAUUSDT order book liquidity heatmap.

The app fetches order book data from Binance via ccxt and stores
snapshots in Redis. Each snapshot keeps aggregated volume at price
levels around the mid price. A Dash interval component updates the
chart to visualize liquidity over time with a heatmap and mid-price line.
"""

import json
import threading
import time
from datetime import datetime

import ccxt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import redis

# Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Parameters for heatmap
STEP = 0.5  # price step in USD
RANGE = 50  # number of steps on each side of mid price
MAX_HISTORY = 120  # number of snapshots to keep

symbol = "XAU/USDT"

price_levels = None


def fetch_orderbook_loop():
    """Fetch order book from Binance and store snapshots in Redis."""
    exchange = ccxt.binance()
    global price_levels
    while True:
        try:
            order_book = exchange.fetch_order_book(symbol.replace("/", ""))
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if not bids or not asks:
                raise ValueError("empty order book")
            bid_price = bids[0][0]
            ask_price = asks[0][0]
            mid_price = (bid_price + ask_price) / 2

            if price_levels is None:
                levels = [mid_price + STEP * i for i in range(-RANGE, RANGE + 1)]
                price_levels = levels
                redis_client.set("price_levels", json.dumps(levels))
            else:
                levels = price_levels

            volume = [0.0 for _ in levels]
            for price, amount in bids + asks:
                idx = int(round((price - levels[0]) / STEP))
                if 0 <= idx < len(levels):
                    volume[idx] += amount

            timestamp = datetime.utcnow().strftime("%H:%M:%S")
            snapshot = [timestamp, mid_price] + volume
            redis_client.rpush("heatmap_history", json.dumps(snapshot))
            redis_client.ltrim("heatmap_history", -MAX_HISTORY, -1)
            redis_client.set("last_bid", bid_price)
            redis_client.set("last_ask", ask_price)
            redis_client.set("last_price", mid_price)
        except Exception as exc:  # pragma: no cover - network errors not tested
            print("Fetch error:", exc)
        time.sleep(5)


def fetch_history():
    rows = [json.loads(x) for x in redis_client.lrange("heatmap_history", 0, -1)]
    levels_raw = redis_client.get("price_levels")
    levels = json.loads(levels_raw) if levels_raw else []
    return rows, levels


def make_figure():
    history, levels = fetch_history()
    if not history or not levels:
        return go.Figure()
    df = pd.DataFrame(history, columns=["time", "price"] + levels)
    z = df.iloc[:, 2:].values.T
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=z, x=df["time"], y=levels, colorscale="Viridis", colorbar=dict(title="Volume"))
    )
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["price"], mode="lines", line=dict(color="white"), name="Price")
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Price (USDT)")
    return fig


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("XAUUSDT Liquidity Heatmap"),
        dcc.Graph(id="heatmap"),
        html.Div(id="bid-ask"),
        dcc.Interval(id="interval", interval=5000, n_intervals=0),
    ]
)


@app.callback(
    [Output("heatmap", "figure"), Output("bid-ask", "children")],
    [Input("interval", "n_intervals")],
)
def update_dashboard(_):
    fig = make_figure()
    bid = redis_client.get("last_bid")
    ask = redis_client.get("last_ask")
    price = redis_client.get("last_price")
    if bid and ask and price:
        bid_ask_text = f"Bid: {float(bid):.2f} Ask: {float(ask):.2f} Price: {float(price):.2f}"
    else:
        bid_ask_text = "No data"
    return fig, bid_ask_text


if __name__ == "__main__":
    thread = threading.Thread(target=fetch_orderbook_loop, daemon=True)
    thread.start()
    app.run(debug=False, port=8051)
