# extract_numbered_asteroids_split_with_lightcurves.py
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(SCRIPT_DIR, "rubin_mpc", "parquet")

# Output root folder (will be created)
OUT_SUBDIR = "numbered_asteroids_outputs"

# Output formats
WRITE_PARQUET = True
WRITE_CSV = True

# Plot settings
SAVE_LIGHTCURVES = True
LIGHTCURVE_USE_MJD = True   # if False, uses obs_time on x-axis
DPI = 150

# The six filters you mentioned (the file appears to use these exact strings)
BANDS_ORDER = ["Lu", "Lg", "Lr", "Li", "Lz", "Ly"]

BAND_MAP = {
    "u": "Lu",
    "lu": "Lu",
    "g": "Lg",
    "lg": "Lg",
    "r": "Lr",
    "lr": "Lr",
    "i": "Li",
    "li": "Li",
    "z": "Lz",
    "lz": "Lz",
    "y": "Ly",
    "ly": "Ly",
}

def normalize_band_series(band: pd.Series) -> pd.Series:
    b = band.astype(str).where(band.notna(), "").str.strip()
    key = b.str.lower()
    mapped = key.map(BAND_MAP)
    return mapped.where(mapped.notna(), b)

# Columns to keep from the input
KEEP_COLS = [
    "permid", "provid", "trkid",
    "mag", "rmsmag", "band",
    "ra", "dec",
    "obstime_text",
]

def get_input_parquet_path() -> str:
    if len(sys.argv) >= 2 and str(sys.argv[1]).strip():
        return str(sys.argv[1]).strip()
    matches = sorted(glob.glob(os.path.join(IN_DIR, "*", "parquet", "obs_sbn_*.parquet")))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Input parquet not found under {IN_DIR}. "
        f"Found {len(matches)} matches: {matches}"
    )

def safe_name(s: str) -> str:
    """Make a safe filename fragment."""
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s

def utc_to_mjd(dt_utc: pd.Series) -> pd.Series:
    """
    Convert UTC datetime series to Modified Julian Date (MJD).

    JD = unix_seconds / 86400 + 2440587.5
    MJD = JD - 2400000.5
    """
    dt = pd.to_datetime(dt_utc, errors="coerce", utc=True)
    mask_nat = dt.isna()

    dt_naive_utc = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    ns = dt_naive_utc.astype("datetime64[ns]").astype("int64")
    seconds = ns.astype("float64") / 1e9
    seconds[mask_nat.to_numpy()] = np.nan

    jd = seconds / 86400.0 + 2440587.5
    mjd = jd - 2400000.5
    return pd.Series(mjd, index=dt_utc.index, dtype="float64")

def write_df(df: pd.DataFrame, basepath_no_ext: str) -> None:
    """Write df to configured formats."""
    if WRITE_CSV:
        df.to_csv(basepath_no_ext + ".csv", index=False)
    if WRITE_PARQUET:
        df.to_parquet(basepath_no_ext + ".parquet", index=False)


def object_already_analyzed(asteroid_dir: str, permid: str) -> bool:
    permid_safe = safe_name(permid)
    base_master = os.path.join(asteroid_dir, f"permid_{permid_safe}_ALL")
    expected_csv = base_master + ".csv"
    expected_parquet = base_master + ".parquet"
    expected_png = os.path.join(asteroid_dir, f"permid_{permid_safe}_lightcurve.png")

    if WRITE_CSV and not os.path.exists(expected_csv):
        return False
    if WRITE_PARQUET and not os.path.exists(expected_parquet):
        return False
    if SAVE_LIGHTCURVES and not os.path.exists(expected_png):
        return False
    return True

def plot_lightcurve_per_asteroid(g: pd.DataFrame, permid: str, out_dir: str) -> None:
    """
    Create a single figure with 6 subplots (one per band) for this asteroid,
    and save it as a PNG.
    """
    # Ensure sorted by time
    g = g.sort_values("obs_time")

    # Decide x-axis
    if LIGHTCURVE_USE_MJD:
        xcol = "mjd"
        xlabel = "MJD"
    else:
        xcol = "obs_time"
        xlabel = "UTC time"

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=False, sharey=False)
    axes = axes.ravel()

    fig.suptitle(f"Light curves for permid={permid}", fontsize=14)

    for i, band in enumerate(BANDS_ORDER):
        ax = axes[i]
        gb = g[g["band"] == band]

        ax.set_title(band)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mag")

        if len(gb) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Drop rows without required values
        gb = gb.dropna(subset=[xcol, "mag"])

        if len(gb) == 0:
            ax.text(0.5, 0.5, "No plottable points", ha="center", va="center", transform=ax.transAxes)
            continue

        # Error bars if rmsmag is present
        if "rmsmag" in gb.columns and gb["rmsmag"].notna().any():
            ax.errorbar(
                gb[xcol],
                gb["mag"],
                yerr=gb["rmsmag"],
                fmt="o",
                markersize=3,
                capsize=2,
                linewidth=1,
            )
        else:
            ax.plot(gb[xcol], gb["mag"], "o", markersize=3)

        # Invert magnitude axis (brighter = up)
        ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    png_name = os.path.join(out_dir, f"permid_{safe_name(permid)}_lightcurve.png")
    fig.savefig(png_name, dpi=DPI)
    plt.close(fig)

def main():
    input_path = get_input_parquet_path()
    input_dir = os.path.dirname(os.path.abspath(input_path))
    out_dir = os.path.join(input_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    # Read
    table = pq.read_table(input_path)
    df = table.to_pandas()

    # Validate columns
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in input: {missing}")

    # Keep only numbered asteroids (permid present + non-empty)
    numbered = df[df["permid"].notna() & (df["permid"].astype(str).str.strip() != "")].copy()

    # Select columns
    out = numbered[KEEP_COLS].copy()

    # Clean / types
    out["permid"] = out["permid"].astype(str).str.strip()
    out["provid"] = out["provid"].astype(str).where(out["provid"].notna(), None)
    out["trkid"] = out["trkid"].astype(str).where(out["trkid"].notna(), None)

    for c in ["mag", "rmsmag", "ra", "dec"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["band"] = normalize_band_series(out["band"])

    # Parse observation time (UTC) and compute MJD
    out["obs_time"] = pd.to_datetime(out["obstime_text"], errors="coerce", utc=True)
    out["mjd"] = utc_to_mjd(out["obs_time"])

    # Drop original time string
    out = out.drop(columns=["obstime_text"])

    # Sort by asteroid and time
    out = out.sort_values(["permid", "obs_time"], ascending=[True, True]).reset_index(drop=True)

    all_bands = sorted(out["band"].dropna().unique().tolist())
    print(f"Total numbered rows: {len(out)}")
    print(f"Total unique permid: {out['permid'].nunique()}")
    print(f"Bands present overall ({len(all_bands)}): {all_bands}")

    # Group and write one file per asteroid (+ per-band splits + lightcurve png)
    for permid, g in out.groupby("permid", sort=False):
        permid_safe = safe_name(permid)
        asteroid_dir = os.path.join(out_dir, f"permid_{permid_safe}")
        if object_already_analyzed(asteroid_dir, permid):
            print(f"Skipping already analyzed numbered object: {permid}")
            continue
        os.makedirs(asteroid_dir, exist_ok=True)

        # 1) Master file per asteroid (all bands)
        base_master = os.path.join(asteroid_dir, f"permid_{permid_safe}_ALL")
        write_df(g, base_master)

        # 2) Split automatically by band (one file per band for this asteroid)
        for band, gb in g.groupby("band", sort=False):
            band_safe = safe_name(band)
            base_band = os.path.join(asteroid_dir, f"permid_{permid_safe}_band_{band_safe}")
            write_df(gb, base_band)

        # 3) Lightcurve plot (single figure, 6 panels)
        if SAVE_LIGHTCURVES:
            plot_lightcurve_per_asteroid(g, permid=permid, out_dir=asteroid_dir)

    print(f"Done. Wrote outputs (and PNGs) under: {out_dir}")

if __name__ == "__main__":
    main()
