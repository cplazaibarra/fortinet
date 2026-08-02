"""
src/debug_naive_slice.py
"""
import pandas as pd
import numpy as np

print("1. Loading dataset...")
df = pd.read_parquet("data/master/master_dataset.parquet")

# simulate entry_idx
entry_idx = np.arange(0, len(df), 32)
print("entry_idx length:", len(entry_idx))

print("2. Converting whole series to tz-naive...")
ts_naive = df['timestamp'].dt.tz_localize(None)
print("  ts_naive type:", type(ts_naive), ts_naive.dtype)

print("3. Slicing tz-naive series...")
ts_naive_sliced = ts_naive.iloc[entry_idx]
print("  Sliced success! length:", len(ts_naive_sliced), "type:", type(ts_naive_sliced), ts_naive_sliced.dtype)
print("  Min:", ts_naive_sliced.min(), "Max:", ts_naive_sliced.max())
