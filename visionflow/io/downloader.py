import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from . import files, network

class AsyncDownloader:
    def __init__(self, concurrency=50, retries=3):
        self.sem = asyncio.Semaphore(concurrency)
        self.retries = retries
        ############ We cannot share one connector across multiple sessions.

    async def download_batch(self, tasks, description="Downloading"):
        if not tasks:
            print("No tasks provided.")
            return 0
            
        # --- Create a NEW connector for every batch ---
        connector = aiohttp.TCPConnector(limit_per_host=None)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            jobs = [self._worker(session, url, path) for url, path in tasks]
            results = [await f for f in tqdm.as_completed(jobs, desc=description)]
            
        return sum(results)

    async def _worker(self, session, url, path):
        if files.exists(path): return True
        
        async with self.sem:
            # 2. Fetch
            resp = await network.fetch(session, url, self.retries)
            if not resp: 
                # print(f"Failed to fetch: {url}") 
                return False
            
            try:
                await files.save_stream(resp, path)
                return True
            except Exception as e:
                print(f"Error saving {path}: {e}")
                return False
            finally:
                resp.close()