import asyncio

SYMBOL = "A"  # update with actual symbol name from the exchange


async def trade(client):
    await asyncio.sleep(5)  # allow order books to populate after connect
    book = client.order_books.get(SYMBOL)
    if book:
        bids = sorted((k, v) for k, v in book.bids.items() if v != 0)
        asks = sorted((k, v) for k, v in book.asks.items() if v != 0)
        print(f"\n[{SYMBOL}] Order Book")
        print(f"  Bids: {bids}")
        print(f"  Asks: {asks}")
    else:
        print(f"[{SYMBOL}] No order book data received yet")
