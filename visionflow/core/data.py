import pandas as pd
from typing import List, Callable

class DataManager:
    def __init__(self, csv_path: str):
        print(f"Loading CSV: {csv_path}...")
        self.df = pd.read_csv(csv_path)

    def slice(self, start: int = 0, end: int = -1):
        actual_end = end if end != -1 else len(self.df)
        self.df = self.df.iloc[start:actual_end].copy()
        return self

    def filter(self, condition_func: Callable):
        self.df = self.df[self.df.apply(condition_func, axis=1)]
        return self

    def get_records(self) -> List[dict]:
        return self.df.to_dict('records')