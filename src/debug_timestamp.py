"""
src/debug_timestamp.py
"""
import pandas as pd
import numpy as np

print("1. Loading dataset...")
df = pd.read_parquet("data/master/master_dataset.parquet")

# simulate entry_idx
entry_idx = np.arange(0, len(df), 32)
print("entry_idx length:", len(entry_idx))

print("Method 1: df['timestamp'].iloc[entry_idx]")
try:
    ts_iloc = df['timestamp'].iloc[entry_idx]
    print("  Success! type:", type(ts_iloc), ts_iloc.dtype)
except Exception as e:
    print("  Failed Method 1:", e)

print("Method 2: df['timestamp'].values[entry_idx]")
try:
    ts_val_slice = df['timestamp'].values[entry_idx]
    print("  Success! type:", type(ts_val_slice), ts_val_slice.dtype)
except Exception as e:
    print("  Failed Method 2:", e)

print("Method 3: df.loc[entry_idx, 'timestamp']")
try:
    ts_loc = df.loc[entry_idx, 'timestamp']
    print("  Success! type:", type(ts_loc), ts_loc.dtype)
except Exception as e:
    print("  Failed Method 3:", e)

print("Method 4: df['timestamp'].to_numpy()[entry_idx]")
try:
    ts_numpy = df['timestamp'].to_numpy()[entry_idx]
    print("  Success! type:", type(ts_numpy), ts_numpy.dtype)
except Exception as e:
    print("  Failed Method 4:", e)

print("Method 5: df['timestamp'].dt.tz_localize(None).iloc[entry_idx]")
try:
    ts_naive = df['timestamp'].dt.tz_localize(None).iloc[entry_idx]
    print("  Success! type:", type(ts_naive), ts_naive.dtype)
except Exception as e:
    print("  Failed Method 5:", e)
