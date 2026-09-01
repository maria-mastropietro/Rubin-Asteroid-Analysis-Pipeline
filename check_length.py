import glob
import os
import pyarrow.parquet as pq

base_dir = "../rubin_mpc/parquet"
pattern = os.path.join(base_dir, "*", "parquet", "obs_sbn_X05_202*.parquet")

paths = sorted(glob.glob(pattern))
print(f"Found {len(paths)} parquet files to check.")

if not paths:
    print("No files matched your pattern!")

for path in paths:
    pf = pq.ParquetFile(path)
    print(f"{path}: {pf.metadata.num_rows}")
