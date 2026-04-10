import os
from dotenv import load_dotenv
load_dotenv()
print()


from typing import Optional

from utcxchangelib import XChangeClient, Side
import asyncio


class MyXchangeClient(XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        pass

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        pass


    async def trade(self):
        """This is a simple example bot that places orders and prints updates."""
        await asyncio.sleep(5)

        # You can also look at order books like this
        for security, book in self.order_books.items():
            if book.bids or book.asks:
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

    async def start(self):
        asyncio.create_task(self.trade())
        await self.connect()

def require_env(name: str) -> str:
    """Return the environment variable `name` or raise a clear error.

    This keeps environment access centralized and delays failure until the
    value is actually required.
    """
    val = os.environ.get(name)
    if val is None:
        raise RuntimeError(f"Environment variable '{name}' is required")
    return val


async def main():
    SERVER = require_env('server')
    USER = require_env('user')
    PASSWORD = require_env('password')
    my_client = MyXchangeClient(SERVER, USER, PASSWORD)
    print(f"Connecting to {SERVER} as {USER}")
    await my_client.start()


if __name__ == "__main__":
    asyncio.run(main())