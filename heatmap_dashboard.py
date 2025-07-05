# Python script to fetch order book data from Binance and display a liquidity heatmap using Dash.

import asyncio
import random
import time
from datetime import datetime

import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import numpy as np
import pandas as pd

import ccxt


# Attempt to create a Binance exchange instance
try:
    binance = ccxt.binance()
except Exception:
    binance = None

symbol = "XAU/USDT"
price_levels = None
history = []
MAX_HISTORY = 120  # keep last 120 snapshots
STEP = 0.5         # price increment for bins


def simulate_order_book():
    """Generate simulated order book data when Binance is unreachable."""
    mid = random.uniform(2300, 2400)
    bids = [[mid - i * STEP, random.uniform(1, 20)] for i in range(50)]
    asks = [[mid + i * STEP, random.uniform(1, 20)] for i in range(50)]
    return bids, asks


def fetch_order_book():
    """Fetch order book from Binance or return simulated data on failure."""
    if binance is None:
        return simulate_order_book()
    try:
        ob = binance.fetch_order_book(symbol)
        return ob['bids'], ob['asks']
    except Exception:
        return simulate_order_book()


def update_history():
    global price_levels
    bids, asks = fetch_order_book()
    if not bids or not asks:
        return 0, 0

    bid_price, _ = bids[0]
    ask_price, _ = asks[0]

    if price_levels is None:
        mid = (bid_price + ask_price) / 2
        levels = [mid + STEP * i for i in range(-40, 41)]
        price_levels = levels
    else:
        levels = price_levels

    volume_at_levels = [0.0 for _ in levels]
    for price, amount in bids + asks:
        idx = int(round((price - levels[0]) / STEP))
        if 0 <= idx < len(levels):
            volume_at_levels[idx] += amount

    timestamp = datetime.utcnow().strftime('%H:%M:%S')
    history.append([timestamp] + volume_at_levels)
    if len(history) > MAX_HISTORY:
        history.pop(0)

    return bid_price, ask_price


def make_figure():
    if not history:
        return go.Figure()
    df = pd.DataFrame(history, columns=['time'] + price_levels)
    z = df.iloc[:, 1:].values.T
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=df['time'],
        y=price_levels,
        colorscale='Viridis'))
    fig.update_layout(xaxis_title='Time', yaxis_title='Price')
    return fig


app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("XAUUSD Liquidity Heatmap"),
    dcc.Graph(id='heatmap'),
    html.Div(id='bid-ask'),
    dcc.Interval(id='interval', interval=5000, n_intervals=0)
])


@app.callback(
    [Output('heatmap', 'figure'), Output('bid-ask', 'children')],
    [Input('interval', 'n_intervals')]
)
def update_dashboard(n):
    bid, ask = update_history()
    fig = make_figure()
    bid_ask_text = f"Bid: {bid:.2f} Ask: {ask:.2f}"
    return fig, bid_ask_text


if __name__ == '__main__':
    # Dash 3 uses app.run instead of the deprecated run_server
    app.run(debug=False)
