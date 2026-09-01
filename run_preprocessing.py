import glob
import os
import re
import subprocess
import sys
from pathlib import Path
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = (SCRIPT_DIR / ".." / "rubin_mpc" / "parquet").resolve()

# Leave REQUESTED empty to automatically discover all parquet files under ../rubin_mpc/parquet.
# Use this list only if you want to explicitly limit the set of parquet files to process.
REQUESTED = [
    ("2026-02-27", "obs_sbn_X05_2026-02-27.parquet"),

]

PROVISIONAL_RE = re.compile(r"^\d{4}\s+[A-Z]{1,2}\d*$")


def get_parquet_paths() -> list[str]:
    if REQUESTED:
        paths = [BASE_DIR / d / "parquet" / f for d, f in REQUESTED]
        missing_paths = [p for p in paths if not p.exists()]
        if missing_paths:
            print("Warning: missing requested parquet files:")
            for p in missing_paths:
                print("  ", p)
        return [str(p) for p in paths if p.exists()]

    # Auto-discover all parquet files under date subdirectories.
    candidate = sorted(BASE_DIR.glob("*/parquet/obs_sbn_*.parquet"))
    return [str(p) for p in candidate]


paths = get_parquet_paths()

NUMBERED_SCRIPT = SCRIPT_DIR / "Select_numbered_split_plot.py"
PROVISIONAL_SCRIPT = SCRIPT_DIR / "Select_multiopp_split_plot.py"


def parquet_num_rows(path: str) -> int:
    pf = pq.ParquetFile(path)
    if pf.metadata is None:
        return 0
    return int(pf.metadata.num_rows)


def run_one(script_path: str, parquet_path: str) -> None:
    cmd = [sys.executable, script_path, parquet_path]
    subprocess.run(cmd, check=True)


def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def parquet_expected_numbered_ids(path: str) -> set[str]:
    table = pq.read_table(path, columns=["permid"])
    ids = set()
    if "permid" not in table.column_names:
        return ids
    for value in table.column("permid").to_pylist():
        if value is None:
            continue
        permid = str(value).strip()
        if permid:
            ids.add(permid)
    return ids


def parquet_expected_provisional_ids(path: str) -> set[str]:
    table = pq.read_table(path, columns=["provid"])
    ids = set()
    if "provid" not in table.column_names:
        return ids
    for value in table.column("provid").to_pylist():
        if value is None:
            continue
        provid = str(value).strip()
        if provid and PROVISIONAL_RE.match(provid):
            ids.add(provid)
    return ids


def parquet_incomplete_numbered_ids(path: str) -> set[str]:
    expected = parquet_expected_numbered_ids(path)
    if not expected:
        return set()

    input_dir = os.path.dirname(os.path.abspath(path))
    numbered_dir = os.path.join(input_dir, "numbered_asteroids_outputs")
    missing = set()

    for permid in expected:
        asteroid_dir = os.path.join(numbered_dir, f"permid_{safe_name(permid)}")
        if not os.path.isdir(asteroid_dir) or not _has_required_master_files(asteroid_dir, f"permid_{safe_name(permid)}_ALL"):
            missing.add(permid)

    return missing


def parquet_incomplete_provisional_ids(path: str) -> set[str]:
    expected = parquet_expected_provisional_ids(path)
    if not expected:
        return set()

    input_dir = os.path.dirname(os.path.abspath(path))
    provisional_dir = os.path.join(input_dir, "provisional_designation_outputs")
    missing = set()

    for provid in expected:
        obj_dir = os.path.join(provisional_dir, f"provid_{safe_name(provid)}")
        if not os.path.isdir(obj_dir) or not _has_required_master_files(obj_dir, f"provid_{safe_name(provid)}_ALL"):
            missing.add(provid)

    return missing


def _has_required_master_files(obj_dir: str, prefix: str) -> bool:
    has_master = any(
        os.path.exists(os.path.join(obj_dir, prefix + ext))
        for ext in [".parquet", ".csv"]
    )
    has_png = os.path.exists(os.path.join(obj_dir, prefix + "_lightcurve.png"))
    return has_master and has_png


def output_complete_for_parquet(parquet_path: str) -> bool:
    input_dir = os.path.dirname(os.path.abspath(parquet_path))
    numbered_dir = os.path.join(input_dir, "numbered_asteroids_outputs")
    provisional_dir = os.path.join(input_dir, "provisional_designation_outputs")

    expected_numbered = parquet_expected_numbered_ids(parquet_path)
    expected_provisional = parquet_expected_provisional_ids(parquet_path)

    if expected_numbered and not os.path.isdir(numbered_dir):
        return False
    if expected_provisional and not os.path.isdir(provisional_dir):
        return False

    for permid in expected_numbered:
        asteroid_dir = os.path.join(numbered_dir, f"permid_{safe_name(permid)}")
        if not os.path.isdir(asteroid_dir):
            return False
        if not _has_required_master_files(asteroid_dir, f"permid_{safe_name(permid)}_ALL"):
            return False

    for provid in expected_provisional:
        obj_dir = os.path.join(provisional_dir, f"provid_{safe_name(provid)}")
        if not os.path.isdir(obj_dir):
            return False
        if not _has_required_master_files(obj_dir, f"provid_{safe_name(provid)}_ALL"):
            return False

    return True


def main() -> None:
    if not paths:
        print("No files matched your pattern!")
        return

    nonzero = []
    for path in paths:
        n = parquet_num_rows(path)
        print(f"{path}: {n}")
        if n > 0:
            nonzero.append(path)

    if not nonzero:
        print("No non-zero parquet files found. Nothing to run.")
        return

    ans = input(f"\nRun processing for {len(nonzero)} non-zero parquet file(s)? [Y/N] ").strip().lower()
    if ans not in {"y", "yes"}:
        print("Not running processing.")
        return

    if not os.path.exists(NUMBERED_SCRIPT):
        raise FileNotFoundError(NUMBERED_SCRIPT)
    if not os.path.exists(PROVISIONAL_SCRIPT):
        raise FileNotFoundError(PROVISIONAL_SCRIPT)

    for p in nonzero:
        missing_numbered = parquet_incomplete_numbered_ids(p)
        missing_provisional = parquet_incomplete_provisional_ids(p)

        if not missing_numbered and not missing_provisional:
            print(f"\n=== Skipping (outputs already complete): {p} ===")
            continue

        print(f"\n=== Processing: {p} ===")
        if missing_numbered:
            print(f"Found {len(missing_numbered)} numbered object(s) needing output")
            run_one(NUMBERED_SCRIPT, p)
        else:
            print("No missing numbered outputs. Skipping numbered processing.")

        if missing_provisional:
            print(f"Found {len(missing_provisional)} provisional object(s) needing output")
            run_one(PROVISIONAL_SCRIPT, p)
        else:
            print("No missing provisional outputs. Skipping provisional processing.")


if __name__ == "__main__":
    main()
