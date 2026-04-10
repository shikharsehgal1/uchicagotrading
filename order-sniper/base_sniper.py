import os
import asyncio
from dotenv import load_dotenv
from utcxchangelib import XChangeClient

import a_sniper

load_dotenv()


class MyXchangeClient(XChangeClient):

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        pass

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        pass

    async def start(self):
        asyncio.create_task(a_sniper.trade(self))
        await self.connect()


async def main():
    server = os.environ["server"]
    user = os.environ["user"]
    password = os.environ["password"]
    while True:
        try:
            print(f"Connecting to {server} as {user}...")
            client = MyXchangeClient(server, user, password)
            await client.start()
        except Exception as e:
            print(f"Connection lost: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
