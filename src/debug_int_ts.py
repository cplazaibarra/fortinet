"""
src/debug_int_ts.py
"""
import pandas as pd
import numpy as np

print("1. Loading dataset...")
df = pd.read_parquet("data/master/master_dataset.parquet")

# simulate entry_idx
entry_idx = np.arange(0, len(df), 32)
print("entry_idx length:", len(entry_idx))

print("2. Converting to int64 values...")
ts_int = df['timestamp'].astype('int64').values
print("  ts_int type:", type(ts_int), ts_int.dtype)

print("3. Slicing int64 values...")
ts_int_sliced = ts_int[entry_idx]
print("  Sliced success! length:", len(ts_int_sliced))

print("4. Converting back to datetime...")
ts_datetime = pd.to_datetime(ts_int_sliced)
print("  Datetime success! type:", type(ts_datetime), ts_datetime.dtype)

print("5. Localize to UTC then remove timezone (naive)...")
ts_naive = ts_datetime.tz_localize('UTC').tz_localize(None)
print("  Naive success! min:", ts_naive.min(), "max:", ts_naive.max())
