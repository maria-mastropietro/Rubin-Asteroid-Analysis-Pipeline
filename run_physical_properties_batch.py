"""
Run Asteroid_physical_properties.py for all objects in choose_physical.txt.

This script:
1. Reads objects from choose_physical.txt
2. Finds the corresponding *_ALL.csv files for each object
3. Runs Asteroid_physical_properties.py for each object
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# =========================
# Configuration
# =========================
# Path to choose_physical.txt
CHOOSE_FILE = Path(__file__).resolve().parent / "choose_physical.txt"

# Path to Asteroid_physical_properties.py
PHYSICAL_PROPERTIES_SCRIPT = Path(__file__).resolve().parent / "Asteroid_physical_properties.py"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = (SCRIPT_DIR / ".." / "rubin_mpc" / "parquet").resolve()

# Directories to search for *_ALL.csv files
PROVID_DIR = Path(__file__).resolve().parent / "parquet"
PERMID_DIR = Path(__file__).resolve().parent / "parquet"

# Default HORIZONS location
HORIZONS_LOCATION = os.environ.get("HORIZONS_LOCATION", "X05")


# =========================
# Helpers
# =========================
def read_choose_objects(input_file: Path) -> dict[str, str]:
    """Read object names and parquet files from choose file."""
    objects = {}

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return objects

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                match = re.match(r"^(.*?)\s+(\S+\.parquet)\s*$", line)
                if match:
                    object_name = match.group(1).strip()
                    parquet_file = match.group(2).strip()
                    objects[object_name] = parquet_file

    return objects


def extract_date_from_parquet(parquet_file: str) -> str:
    """Extract date from parquet file name (e.g., obs_sbn_X05_2026-04-28.parquet -> 2026-04-28)."""
    match = re.search(r"obs_sbn_[^_]+_(\d{4}-\d{2}-\d{2})\.parquet", parquet_file)
    if match:
        return match.group(1)
    return ""


def find_all_csv_for_object(save_dir: Path, object_name: str) -> Optional[Path]:
    """Find the *_ALL.csv file in a directory that matches the object name."""
    # First try to find a CSV that contains the object name in the filename (recursive)
    matching_csvs = sorted(save_dir.rglob(f"*{object_name.replace(' ', '_')}*_ALL.csv"))
    if matching_csvs:
        return matching_csvs[0]
    
    # Fallback to any *_ALL.csv file (recursive)
    all_csvs = sorted(save_dir.rglob("*_ALL.csv"))
    if not all_csvs:
        return None
    return all_csvs[0]


def safe_prefix(object_name: str) -> str:
    """Create a safe filename prefix from object name."""
    s = str(object_name).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._")
    return s


def run_physical_properties_for_object(
    object_name: str,
    csv_path: Path,
    output_dir: Path,
    location: str,
) -> bool:
    """
    Run Asteroid_physical_properties.py for a single object.
    Returns True if successful, False otherwise.
    """
    out_prefix = safe_prefix(object_name)

    cmd = [
        sys.executable,
        str(PHYSICAL_PROPERTIES_SCRIPT),
        "--csv", str(csv_path),
        "--target", str(object_name),
        "--location", location,
        "--out-prefix", out_prefix,
    ]

    print(f"  -> Running: {object_name}")
    print(f"     CSV: {csv_path}")

    try:
        subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent),  # Save results in Asteroid_code folder
            check=True,
        )
        print(f"     [OK] {object_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"     [FAILED] {object_name}: exit code {e.returncode}")
        return False


def main():
    print("=" * 80)
    print("Run Asteroid_physical_properties.py for objects in choose_physical.txt")
    print("=" * 80)

    # Check if Asteroid_physical_properties.py exists
    if not PHYSICAL_PROPERTIES_SCRIPT.exists():
        print(f"[ERROR] Asteroid_physical_properties.py not found: {PHYSICAL_PROPERTIES_SCRIPT}")
        return

    # Read objects from choose_physical.txt
    print(f"\nReading objects from: {CHOOSE_FILE}")
    objects = read_choose_objects(CHOOSE_FILE)

    if not objects:
        print("No objects found in choose_physical.txt. Exiting.")
        return

    print(f"Found {len(objects)} objects in choose_physical.txt")

    # Process each object
    ok = 0
    failed = 0
    skipped = 0

    for object_name, parquet_file in objects.items():
        date = extract_date_from_parquet(parquet_file)
        if not date:
            print(f"[SKIP] {object_name}: could not extract date from {parquet_file}")
            skipped += 1
            continue

        save_dir = BASE_DIR / date
        if not save_dir.exists():
            print(f"[SKIP] {object_name}: directory not found {save_dir}")
            skipped += 1
            continue

        csv_path = find_all_csv_for_object(save_dir, object_name)
        if csv_path is None:
            print(f"[SKIP] {object_name}: *_ALL.csv not found in {save_dir}")
            skipped += 1
            continue

        if run_physical_properties_for_object(
            object_name,
            csv_path,
            save_dir,
            HORIZONS_LOCATION,
        ):
            ok += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  OK: {ok}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(objects)}")
    print("=" * 80)


if __name__ == "__main__":
    main()


# python run_physical_properties_batch.py
