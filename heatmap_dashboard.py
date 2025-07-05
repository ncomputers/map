# Python script to fetch order book data from Binance and display a liquidity heatmap using Dash.


"""Dashboard to show a BTCUSDT liquidity heatmap using real Binance data."""

import asyncio
import json
import threading
from datetime import datetime

import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import pandas as pd
import plotly.graph_objs as go
import redis
import websockets


symbol = "btcusdt"
stream_url = f"wss://stream.binance.com:9443/ws/{symbol}@depth@100ms"
redis_client = redis.Redis(host="localhost", port=6379, db=0)

price_levels = None
MAX_HISTORY = 120  # last 120 snapshots
STEP = 10.0        # $10 price bins for BTCUSDT


async def listen_order_book():
    """Listen to Binance websocket and store snapshots in Redis."""
    global price_levels
    async with websockets.connect(stream_url) as ws:
        async for message in ws:
            data = json.loads(message)
            bids = [(float(p), float(a)) for p, a in data.get("b", [])]
            asks = [(float(p), float(a)) for p, a in data.get("a", [])]
            if not bids or not asks:
                continue

            bid_price = bids[0][0]
            ask_price = asks[0][0]

            if price_levels is None:
                mid = (bid_price + ask_price) / 2
                levels = [mid + STEP * i for i in range(-40, 41)]
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
            snapshot = [timestamp] + volume
            redis_client.rpush("heatmap_history", json.dumps(snapshot))
            redis_client.ltrim("heatmap_history", -MAX_HISTORY, -1)
            redis_client.set("last_bid", bid_price)
            redis_client.set("last_ask", ask_price)


def fetch_history():
    data = [json.loads(x) for x in redis_client.lrange("heatmap_history", 0, -1)]
    levels_raw = redis_client.get("price_levels")
    levels = json.loads(levels_raw) if levels_raw else []
    return data, levels


def make_figure():
    history, levels = fetch_history()
    if not history or not levels:
        return go.Figure()
    df = pd.DataFrame(history, columns=["time"] + levels)
    z = df.iloc[:, 1:].values.T
    fig = go.Figure(
        data=go.Heatmap(z=z, x=df["time"], y=levels, colorscale="Viridis")
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Price")
    return fig


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("BTCUSDT Liquidity Heatmap"),
        dcc.Graph(id="heatmap"),
        html.Div(id="bid-ask"),
        dcc.Interval(id="interval", interval=5000, n_intervals=0),
    ]
)


@app.callback(
    [Output("heatmap", "figure"), Output("bid-ask", "children")],
    [Input("interval", "n_intervals")],
)
def update_dashboard(n):
    fig = make_figure()
    bid = redis_client.get("last_bid")
    ask = redis_client.get("last_ask")
    if bid and ask:
        bid_ask_text = f"Bid: {float(bid):.2f} Ask: {float(ask):.2f}"
    else:
        bid_ask_text = "No data"
    return fig, bid_ask_text


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    asyncio.run_coroutine_threadsafe(listen_order_book(), loop)
    try:
        app.run(debug=False)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
