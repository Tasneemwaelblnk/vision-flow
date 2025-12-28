import os
import aiohttp

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def exists(path: str) -> bool:
    return os.path.exists(path)

async def save_stream(response: aiohttp.ClientResponse, path: str):
    ensure_dir(path)
    with open(path, 'wb') as f:
        async for chunk in response.content.iter_chunked(1024):
            f.write(chunk)