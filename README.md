# uchicagotrading

Async trading bot for the UChicago Midwest Trading Competition.

## Architecture

One bot process connects to the exchange. Each asset class runs as an independent async task within that single event loop.

```
base_sniper.py          — entry point; connects to exchange, spawns asset tasks
a_sniper.py             — async strategy for asset class A
b_sniper.py             — async strategy for asset class B  (future)
...
```

`base_sniper.py` owns the connection and event callbacks (`bot_handle_order_fill`, `bot_handle_trade_msg`, `bot_handle_book_update`). Each `x_sniper.py` module exposes a `trade(client)` coroutine that receives the shared client and implements that asset's strategy.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
server=<exchange host>
user=<username>
password=<password>
```

## Running

```bash
cd order-sniper
python base_sniper.py
```
