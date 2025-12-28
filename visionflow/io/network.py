import asyncio
import aiohttp

async def fetch(session: aiohttp.ClientSession, url: str, retries: int = 3):
    delay = 1
    for i in range(retries + 1):
        try:
            resp = await session.get(url, timeout=15)
            resp.raise_for_status()
            return resp
        except Exception:
            if i == retries: return None
            await asyncio.sleep(delay)
            delay *= 2
    return None