import os
import pandas as pd
import functools

@functools.lru_cache(maxsize=16)
def load_cached_csv(file_path, index_col=None):
    """
    Loads a CSV file and caches the result in memory to avoid multiple
    redundant copies of the same dataset.
    """
    print(f"Loading CSV (uncached first time): {os.path.basename(file_path)}")
    return pd.read_csv(file_path, index_col=index_col)
