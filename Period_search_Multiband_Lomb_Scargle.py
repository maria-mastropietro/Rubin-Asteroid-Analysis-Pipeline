#!/usr/bin/env python3
"""
Estimate the rotational period of asteroid 2026 DO14 from multiband photometry,
using a continuous JPL Horizons ephemeris range plus interpolation.

Workflow
--------
1) Read photometry CSV.
2) Query JPL Horizons over one continuous time range covering the observations.
3) Interpolate ephemeris quantities back onto each observation MJD.
4) Build corrected lightcurve:
       corr_mag = observed_mag - predicted_V
   then median-center and 3-sigma clip.
5) Run multiband Lomb-Scargle with a 2nd-order Fourier model.
6) Keep candidate periods whose fitted phased lightcurve has two minima.
7) Save merged table and plots.

Notes
-----
- This is a JPL-based approximation to the paper's Section 3.2 method.
- The paper uses MPC-predicted V magnitudes; this script uses Horizons V.
- If rmsmag is missing, the period search is run unweighted.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astroquery.jplhorizons import Horizons
from astropy.timeseries import LombScargleMultiband
from astropy.time import Time

import re

from period_error_montgomery import compute_period_error


# ---------- Plot styling ----------
PLOT_FONT_SIZE = 14
MEDIUM_SIZE = 15
BIGGER_SIZE = 17

plt.rc('font', size=PLOT_FONT_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=PLOT_FONT_SIZE)

FIGSIZE = (7.2, 4.8)


MJDREF = 2400000.5

AU_KM = 149597870.700
C_KM_S = 299792.458
SECONDS_PER_DAY = 86400.0
LIGHT_TIME_DAYS_PER_AU = AU_KM / C_KM_S / SECONDS_PER_DAY


def normalize_band_label(x: str) -> str:
    if pd.isna(x):
        return str(x)
    x = str(x).strip()
    mapping = {
        "Lu": "u", "Lg": "g", "Lr": "r", "Li": "i", "Lz": "z", "Ly": "y",
        "u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "y": "y",
    }
    return mapping.get(x, x)


def sigma_clip_mask(y: np.ndarray, nsig: float = 3.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return finite
    mu = np.nanmean(y[finite])
    sd = np.nanstd(y[finite], ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return finite
    out = np.zeros_like(finite, dtype=bool)
    out[finite] = np.abs(y[finite] - mu) <= nsig * sd
    return out


def phase_fold(t: np.ndarray, period: float, t0: Optional[float] = None) -> np.ndarray:
    if t0 is None:
        t0 = np.nanmin(t)
    return ((t - t0) / period) % 1.0


def count_local_minima(y: np.ndarray) -> int:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return 0
    c = 0
    for i in range(n):
        ym1 = y[(i - 1) % n]
        y0 = y[i]
        yp1 = y[(i + 1) % n]
        if y0 < ym1 and y0 < yp1:
            c += 1
    return c

def make_period_grid(
    t: np.ndarray,
    min_period_days: float = 0.00065,
    max_period_days: float = 3.0,
    oversample: int = 100,
) -> np.ndarray:
    """
    Follow the paper's Section 3.2 recipe more closely:
        n_periods = (1/Pmin - 1/Pmax) * (tmax - tmin) * oversample
    and sample UNIFORMLY IN PERIOD.
    """
    t = np.asarray(t, dtype=float)
    tspan = float(np.nanmax(t) - np.nanmin(t))
    n_periods = int(np.ceil((1.0 / min_period_days - 1.0 / max_period_days) * tspan * oversample))
    n_periods = max(n_periods, 5000)
    return np.linspace(min_period_days, max_period_days, n_periods)

def load_and_prepare_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"mag", "band"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns at least {sorted(required)}")

    if "mjd" not in df.columns:
        if "obs_time" in df.columns:
            obs_time = pd.to_datetime(df["obs_time"], utc=True, errors="coerce")
            unix_sec = obs_time.astype("int64") / 1e9
            jd = unix_sec / 86400.0 + 2440587.5
            df["mjd"] = jd - MJDREF
        else:
            raise ValueError("CSV must contain either 'mjd' or 'obs_time'.")

    df["mjd"] = pd.to_numeric(df["mjd"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["rmsmag"] = pd.to_numeric(df["rmsmag"], errors="coerce") if "rmsmag" in df.columns else np.nan
    df["band"] = df["band"].map(normalize_band_label)

    df = df.dropna(subset=["mjd", "mag", "band"]).copy()
    df = df.sort_values("mjd").reset_index(drop=True)
    return df


def mjd_to_iso_utc(mjd: float) -> str:
    return Time(mjd, format="mjd", scale="utc").isot

def safe_slug(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._")
    return s if s else "target"


def make_horizons_cache_path(
    cache_dir: Path,
    target_id: str,
    location: str,
    step_minutes: int,
    pad_minutes: int,
    start_mjd: float,
    stop_mjd: float,
) -> Path:
    target_slug = safe_slug(target_id)
    loc_slug = safe_slug(location)
    fname = (
        f"{target_slug}__loc_{loc_slug}__step_{step_minutes}m"
        f"__pad_{pad_minutes}m__{start_mjd:.6f}_{stop_mjd:.6f}.csv"
    )
    return cache_dir / fname

def query_horizons_range(
    target_id: str,
    mjd_min: float,
    mjd_max: float,
    location: str = "500",
    id_type: Optional[str] = None,
    step_minutes: int = 1,
    pad_minutes: int = 10,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Query Horizons over one continuous time range and return a dense ephemeris table.

    If caching is enabled, reuse a local CSV cache when the same target/location/
    step/pad/time range has already been queried.
    """
    start_mjd = mjd_min - pad_minutes / 1440.0
    stop_mjd = mjd_max + pad_minutes / 1440.0

    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = make_horizons_cache_path(
            cache_dir=cache_dir,
            target_id=target_id,
            location=location,
            step_minutes=step_minutes,
            pad_minutes=pad_minutes,
            start_mjd=start_mjd,
            stop_mjd=stop_mjd,
        )

    # ---------- try cache first ----------
    if use_cache and cache_path is not None and cache_path.exists():
        print(f"Loading Horizons cache: {cache_path.resolve()}")
        out = pd.read_csv(cache_path)

        # Safety checks
        if "mjd" not in out.columns and "jd" in out.columns:
            out["jd"] = pd.to_numeric(out["jd"], errors="coerce")
            out["mjd"] = out["jd"] - MJDREF

        out = out.sort_values("mjd").reset_index(drop=True)

        if len(out) >= 2:
            return out

        print("Cache file was invalid or too short; querying Horizons again...")

    # ---------- otherwise query Horizons ----------
    epochs = {
        "start": mjd_to_iso_utc(start_mjd),
        "stop": mjd_to_iso_utc(stop_mjd),
        "step": f"{int(step_minutes)}m",
    }

    if id_type is None:
        obj = Horizons(id=target_id, location=location, epochs=epochs)
    else:
        obj = Horizons(id=target_id, id_type=id_type, location=location, epochs=epochs)

    eph = obj.ephemerides()
    eph_df = eph.to_pandas()

    out = pd.DataFrame({
        "jd": pd.to_numeric(eph_df.get("datetime_jd", np.nan), errors="coerce"),
        "pred_V": pd.to_numeric(eph_df.get("V", np.nan), errors="coerce"),
        "r_au": pd.to_numeric(eph_df.get("r", np.nan), errors="coerce"),
        "delta_au": pd.to_numeric(eph_df.get("delta", np.nan), errors="coerce"),
        "alpha_deg": pd.to_numeric(eph_df.get("alpha", np.nan), errors="coerce"),
        "RA_deg": pd.to_numeric(eph_df.get("RA", np.nan), errors="coerce"),
        "DEC_deg": pd.to_numeric(eph_df.get("DEC", np.nan), errors="coerce"),
    })

    out = out.dropna(subset=["jd"]).copy()
    out["mjd"] = out["jd"] - MJDREF
    out = out.sort_values("mjd").reset_index(drop=True)

    if len(out) < 2:
        raise RuntimeError("Horizons range query returned too few points for interpolation.")

    # ---------- save cache ----------
    if use_cache and cache_path is not None:
        out.to_csv(cache_path, index=False)
        print(f"Saved Horizons cache: {cache_path.resolve()}")

    return out


def interpolate_ephemerides(obs_df: pd.DataFrame, eph_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate selected Horizons quantities onto observation MJDs.
    """
    x = eph_df["mjd"].to_numpy(dtype=float)
    ycols = ["pred_V", "r_au", "delta_au", "alpha_deg", "RA_deg", "DEC_deg"]

    out = obs_df.copy()
    xo = out["mjd"].to_numpy(dtype=float)

    xmin = np.nanmin(x)
#    xmax = np.nanmax(xo)
    if np.nanmin(xo) < xmin or np.nanmax(xo) > np.nanmax(x):
        raise RuntimeError("Observation times fall outside the queried Horizons range.")

    for col in ycols:
        y = eph_df[col].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            out[col] = np.nan
            continue
        out[col] = np.interp(xo, x[finite], y[finite])

    out = out.rename(columns={
        "RA_deg": "RA_horizons_deg",
        "DEC_deg": "DEC_horizons_deg",
    })
    return out


def build_corrected_lightcurve(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if out["pred_V"].notna().sum() == 0:
        raise RuntimeError("No interpolated Horizons V magnitudes were available.")

    # Light-time correction (mirrors HOF's add_section31_preprocessing)
    out["light_time_days"] = out["delta_au"] * LIGHT_TIME_DAYS_PER_AU
    out["t_corr_mjd"] = out["mjd"] - out["light_time_days"]

    # Raw correction against predicted V
    out["corr_mag"] = out["mag"] - out["pred_V"]

    # Per-band median subtraction to remove filter-dependent offsets
    band_medians = (
        out.groupby("band", dropna=False)["corr_mag"]
        .median()
        .to_dict()
    )
    out["band_median_offset"] = out["band"].map(band_medians)
    out["corr_mag_band_centered"] = out["corr_mag"] - out["band_median_offset"]

    # No final global median subtraction
    out["corr_mag_centered"] = out["corr_mag_band_centered"]

    # Sigma clipping on the band-corrected values
    out["is_inlier"] = sigma_clip_mask(
        out["corr_mag_centered"].to_numpy(dtype=float),
        nsig=3.0
    )

    return out


@dataclass
class CandidatePeriod:
    period_days: float
    power: float
    n_minima: int
    is_local_max: bool = True
    
def find_local_maxima(y: np.ndarray) -> np.ndarray:
    """
    Simple local-max finder.
    """
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=bool)
    if len(y) < 3:
        return out
    out[1:-1] = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    return out

def fit_multiband_periodogram(
    df,
    min_period_days: float = 0.00065,
    max_period_days: float = 3.0,
    oversample: int = 100,
) -> Tuple[np.ndarray, np.ndarray, LombScargleMultiband]:
    use = df[df["is_inlier"]].copy()
    if len(use) < 10:
        raise RuntimeError("Too few inlier points for a period search.")

    t = use["t_corr_mjd"].to_numpy(dtype=float)
    y = use["corr_mag_centered"].to_numpy(dtype=float)
    bands = use["band"].astype(str).to_numpy()

    dy = use["rmsmag"].to_numpy(dtype=float) if "rmsmag" in use.columns else None
    if dy is not None:
        bad = ~np.isfinite(dy) | (dy <= 0)
        if np.all(bad):
            dy = None
        else:
            med = np.nanmedian(dy[~bad])
            dy = dy.copy()
            dy[bad] = med

    model = LombScargleMultiband(
        t=t,
        y=y,
        dy=dy,
        bands=bands,
        nterms_base=2,
        nterms_band=1,
        reg_base=None,
        reg_band=1e-6,
    )

    periods = make_period_grid(
        t=t,
        min_period_days=min_period_days,
        max_period_days=max_period_days,
        oversample=oversample,
    )
    freq = 1.0 / periods
    power = model.power(freq)
    return periods, power, model

def fit_fourier_series(
    phase: np.ndarray,
    y: np.ndarray,
    order: int = 2,
    dy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fit y(phase) with a Fourier series:
        c0 + sum_k [a_k cos(2πkφ) + b_k sin(2πkφ)]
    Returns coefficient vector beta and covariance matrix (or None if not computable).
    """
    phase = np.asarray(phase, dtype=float) % 1.0
    y = np.asarray(y, dtype=float)

    cols = [np.ones_like(phase)]
    for k in range(1, order + 1):
        cols.append(np.cos(2.0 * np.pi * k * phase))
        cols.append(np.sin(2.0 * np.pi * k * phase))
    X = np.column_stack(cols)

    if dy is not None:
        dy = np.asarray(dy, dtype=float)
        good = np.isfinite(dy) & (dy > 0)
        if np.any(good):
            w = np.ones_like(y)
            w[good] = 1.0 / dy[good]
            Xw = X * w[:, None]
            yw = y * w
            beta, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
            
            # Compute covariance matrix
            w2 = w ** 2
            n_params = X.shape[1]
            dof = len(y) - n_params
            if dof > 0 and rank == n_params:
                # Compute MSE in original data space: Σ w_i (y_i - ŷ_i)^2 / dof
                yhat = X @ beta
                mse = np.sum(w2 * (y - yhat)**2) / dof
                if np.isfinite(mse) and mse > 0:
                    # cov(beta) = mse * (X^T W X)^(-1)
                    XtWX = X.T @ (w2[:, None] * X)
                    try:
                        cov = mse * np.linalg.inv(XtWX)
                        return beta, cov
                    except np.linalg.LinAlgError:
                        pass
            return beta, None

    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    # Compute covariance matrix for unweighted fit
    n_params = X.shape[1]
    dof = len(y) - n_params
    if dof > 0 and rank == n_params:
        if len(residuals) > 0:
            mse = residuals[0] / dof if residuals[0] > 0 else np.nan
        else:
            yhat = X @ beta
            mse = np.sum((y - yhat)**2) / dof
        if np.isfinite(mse) and mse > 0:
            try:
                cov = mse * np.linalg.inv(X.T @ X)
                return beta, cov
            except np.linalg.LinAlgError:
                pass
    return beta, None


def eval_fourier_series(
    phase: np.ndarray,
    beta: np.ndarray,
    order: int = 2,
) -> np.ndarray:
    phase = np.asarray(phase, dtype=float) % 1.0
    y = np.full_like(phase, beta[0], dtype=float)

    j = 1
    for k in range(1, order + 1):
        y += beta[j] * np.cos(2.0 * np.pi * k * phase)
        j += 1
        y += beta[j] * np.sin(2.0 * np.pi * k * phase)
        j += 1

    return y

def compute_lightcurve_amplitude_from_beta(
    beta: np.ndarray,
    order: int = 2,
    n_grid: int = 1000,
) -> float:
    """
    Compute peak-to-peak amplitude of a fitted Fourier model.
    """
    phase = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    y = eval_fourier_series(phase, beta, order=order)
    return float(np.nanmax(y) - np.nanmin(y))


def compute_amplitude_error_from_covariance(
    beta: np.ndarray,
    cov: np.ndarray,
    order: int = 2,
    n_grid: int = 1000,
) -> Optional[float]:
    """
    Compute amplitude error using error propagation from the covariance matrix.
    
    The amplitude A = max(y) - min(y) where y = f(phase, beta).
    Using error propagation: var(A) ≈ ∇A^T cov(β) ∇A
    where ∇A is the gradient of A with respect to β, computed numerically.
    """
    if cov is None or not np.isfinite(cov).all():
        return None
    
    beta = np.asarray(beta, dtype=float)
    n_params = len(beta)
    
    # Compute amplitude at current beta
    phase = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    y = eval_fourier_series(phase, beta, order=order)
    amplitude = np.nanmax(y) - np.nanmin(y)
    
    # Find phases of max and min
    idx_max = np.nanargmax(y)
    idx_min = np.nanargmin(y)
    phase_max = phase[idx_max]
    phase_min = phase[idx_min]
    
    # Compute gradient of amplitude with respect to beta numerically using central differences
    epsilon = 1e-8
    gradient = np.zeros(n_params)
    
    for i in range(n_params):
        beta_plus = beta.copy()
        beta_plus[i] += epsilon
        y_plus = eval_fourier_series(phase, beta_plus, order=order)
        amplitude_plus = np.nanmax(y_plus) - np.nanmin(y_plus)
        
        beta_minus = beta.copy()
        beta_minus[i] -= epsilon
        y_minus = eval_fourier_series(phase, beta_minus, order=order)
        amplitude_minus = np.nanmax(y_minus) - np.nanmin(y_minus)
        
        gradient[i] = (amplitude_plus - amplitude_minus) / (2 * epsilon)
    
    # Compute variance using error propagation
    try:
        variance = gradient.T @ cov @ gradient
        if variance > 0:
            return float(np.sqrt(variance))
    except (np.linalg.LinAlgError, ValueError):
        pass
    
    return None

def evaluate_candidate_minima(
    df: pd.DataFrame,
    period_days: float,
    n_phase: int = 500,
    order: int = 2,
) -> int:
    """
    Phase the inlier corrected lightcurve at a trial period, fit a 2nd-order
    Fourier series directly to the data, and count minima on the fitted curve.
    """
    use = df[df["is_inlier"]].copy()

    t = use["t_corr_mjd"].to_numpy(dtype=float)
    y = use["corr_mag_centered"].to_numpy(dtype=float)

    if "rmsmag" in use.columns:
        dy = use["rmsmag"].to_numpy(dtype=float)
        if not np.isfinite(dy).any():
            dy = None
    else:
        dy = None

    phase = phase_fold(t, period_days, t0=np.nanmin(t))
    beta, _ = fit_fourier_series(phase, y, order=order, dy=dy)

    phase_grid = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    y_fit = eval_fourier_series(phase_grid, beta, order=order)

    return count_local_minima(y_fit)

def select_candidate_periods(
    df,
    periods: np.ndarray,
    power: np.ndarray,
    top_n_local: int = 80,
) -> Tuple[List[CandidatePeriod], CandidatePeriod]:
    local_max = find_local_maxima(power)
    idx = np.where(local_max)[0]

    if len(idx) == 0:
        idx = np.argsort(power)[::-1][:top_n_local]
    else:
        idx = idx[np.argsort(power[idx])[::-1][:top_n_local]]

    cands = []
    for i in idx:
        p = float(periods[i])
        n_minima = evaluate_candidate_minima(df=df, period_days=p, n_phase=500, order=2)
        cands.append(
            CandidatePeriod(
                period_days=p,
                power=float(power[i]),
                n_minima=n_minima,
                is_local_max=True,
            )
        )

    cands = sorted(cands, key=lambda c: c.power, reverse=True)

    two_min = [c for c in cands if c.n_minima == 2]
    best = two_min[0] if two_min else cands[0]
    return cands, best

def plot_periodogram(
    periods: np.ndarray,
    power: np.ndarray,
    candidates: List[CandidatePeriod],
    best: CandidatePeriod,
    out_png: str,
    max_plot_hours: Optional[float] = None,
    target_name: Optional[str] = None,
    auto_zoom: bool = False,
) -> None:
    """
    Make the plot look more like the paper's LSM panels:
    - x axis is PERIOD in hours, linear scale
    - all power shown as a thin gray curve
    - local-max peaks with 2 minima highlighted
    - best period annotated
    """
    periods_hr = 24.0 * np.asarray(periods, dtype=float)
    power = np.asarray(power, dtype=float)

    plt.figure(figsize=FIGSIZE)

    # Lighter, thinner background spectrum
    plt.plot(periods_hr, power, lw=1.0, alpha=0.45, color="0.5", zorder=1)

    elim_x, elim_y = [], []
    cand_x, cand_y = [], []

    for c in candidates:
        x = c.period_days * 24.0
        y = c.power
        if c.n_minima == 2:
            cand_x.append(x)
            cand_y.append(y)
        else:
            elim_x.append(x)
            elim_y.append(y)

    # Smaller markers to reduce clutter
    if elim_x:
        plt.scatter(
            elim_x, elim_y,
            s=20, marker="o",
            label="Eliminated (!= 2 minima)",
            zorder=3
        )
    if cand_x:
        plt.scatter(
            cand_x, cand_y,
            s=20, marker="o",
            label="Candidate (2 minima)",
            zorder=4
        )

    if best is not None:
        best_hr = best.period_days * 24.0
        plt.axvline(best_hr, lw=1.2, linestyle="--", zorder=2, label=f"Best period: {best_hr:.4f} h")

    plt.xlabel("Period (hours)")
    plt.ylabel("Power")
    plt.title(
        "Lomb-Scargle Periodogram"
        if target_name is None
        else f"Lomb-Scargle periodogram: {target_name}",
    )

    # X limits: by default, only show a bit more than the best solution.
    if max_plot_hours is not None:
        plt.xlim(0.0, max_plot_hours)
    elif best is not None:
        xmax = max(6.0, best_hr * 1.5)
        plt.xlim(0.0, min(xmax, float(periods_hr.max())))
    elif auto_zoom and candidates:
        all_x = [c.period_days * 24.0 for c in candidates]
        xmax = max(all_x)
        plt.xlim(0.0, max(6.0, min(xmax * 1.25, float(periods_hr.max()))))
    else:
        plt.xlim(0.0, float(periods_hr.max()))

    # Add a little headroom so text does not crowd the top edge
    y_max = float(np.nanmax(power))
    plt.ylim(bottom=min(-0.02, float(np.nanmin(power)) - 0.01), top=1.05 * y_max)

    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()



def fit_shared_fourier_with_band_offsets(
    phase: np.ndarray,
    y: np.ndarray,
    bands: np.ndarray,
    order: int = 2,
    dy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict, str, Optional[np.ndarray]]:
    """
    Fit a common Fourier lightcurve shape plus one constant offset per band.

    Returns
    -------
    common_beta : ndarray
        Fourier coefficients for the shared lightcurve shape only.
    band_offsets : dict
        Per-band fitted offsets. Reference band has offset 0.
    ref_band : str
        The band used as the zero-offset reference.
    cov_common : ndarray or None
        Covariance matrix for the common Fourier coefficients only.
    """
    phase = np.asarray(phase, dtype=float) % 1.0
    y = np.asarray(y, dtype=float)
    bands = np.asarray(bands).astype(str)

    # Use the most populated band as reference
    ref_band = pd.Series(bands).value_counts().idxmax()
    unique_bands = sorted(pd.unique(bands))
    offset_bands = [band for band in unique_bands if band != ref_band]

    cols = [np.ones_like(phase)]
    for k in range(1, order + 1):
        cols.append(np.cos(2.0 * np.pi * k * phase))
        cols.append(np.sin(2.0 * np.pi * k * phase))

    for band in offset_bands:
        cols.append((bands == band).astype(float))

    X = np.column_stack(cols)

    if dy is not None:
        dy = np.asarray(dy, dtype=float)
        good = np.isfinite(dy) & (dy > 0)
        if np.any(good):
            w = np.ones_like(y)
            w[good] = 1.0 / dy[good] 
            Xw = X * w[:, None]
            yw = y * w
            beta_all, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
            
            # Compute covariance matrix
            w2 = w ** 2
            n_params = X.shape[1]
            dof = len(y) - n_params
            if dof > 0 and rank == n_params:
                # Compute MSE in original data space: Σ w_i (y_i - ŷ_i)^2 / dof
                yhat = X @ beta_all
                mse = np.sum(w2 * (y - yhat)**2) / dof
                if np.isfinite(mse) and mse > 0:
                    XtWX = X.T @ (w2[:, None] * X)
                    try:
                        cov_all = mse * np.linalg.inv(XtWX)
                        n_common = 1 + 2 * order
                        cov_common = cov_all[:n_common, :n_common]
                    except np.linalg.LinAlgError:
                        cov_common = None
                else:
                    cov_common = None
            else:
                cov_common = None
        else:
            beta_all, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            cov_common = None
    else:
        beta_all, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        # Compute covariance matrix for unweighted fit
        n_params = X.shape[1]
        dof = len(y) - n_params
        if dof > 0 and rank == n_params:
            if len(residuals) > 0:
                mse = residuals[0] / dof if residuals[0] > 0 else np.nan
            else:
                yhat = X @ beta_all
                mse = np.sum((y - yhat)**2) / dof
            if np.isfinite(mse) and mse > 0:
                try:
                    cov_all = mse * np.linalg.inv(X.T @ X)
                    n_common = 1 + 2 * order
                    cov_common = cov_all[:n_common, :n_common]
                except np.linalg.LinAlgError:
                    cov_common = None
            else:
                cov_common = None
        else:
            cov_common = None

    n_common = 1 + 2 * order
    common_beta = beta_all[:n_common]

    band_offsets = {ref_band: 0.0}
    j = n_common
    for band in offset_bands:
        band_offsets[band] = float(beta_all[j])
        j += 1

    return common_beta, band_offsets, ref_band, cov_common


def compute_colors_from_offsets(band_offsets: dict) -> dict:
    def color(a: str, b: str) -> float:
        oa = band_offsets.get(a, np.nan)
        ob = band_offsets.get(b, np.nan)
        if not (np.isfinite(oa) and np.isfinite(ob)):
            return np.nan
        return float(oa - ob)

    return {
        "g-r": color("g", "r"),
        "g-i": color("g", "i"),
        "r-i": color("r", "i"),
        "i-z": color("i", "z"),
    }


def plot_phased_lightcurve(
    df: pd.DataFrame,
    best_period_days: float,
    out_png: str,
    order: int = 2,
    target_name: Optional[str] = None,
) -> None:
    use = df[df["is_inlier"]].copy()
    t = use["t_corr_mjd"].to_numpy(dtype=float)
    y = use["corr_mag_centered"].to_numpy(dtype=float)
    b = use["band"].astype(str).to_numpy()

    if "rmsmag" in use.columns:
        dy = use["rmsmag"].to_numpy(dtype=float)
        if not np.isfinite(dy).any():
            dy = None
        else:
            bad = ~np.isfinite(dy) | (dy <= 0)
            if np.all(bad):
                dy = None
            else:
                med = np.nanmedian(dy[~bad])
                dy = dy.copy()
                dy[bad] = med
    else:
        dy = None

    phase = phase_fold(t, best_period_days, t0=np.nanmin(t))

    # Fit common shape + per-band offsets
    common_beta, band_offsets, ref_band, cov_common = fit_shared_fourier_with_band_offsets(
        phase=phase,
        y=y,
        bands=b,
        order=order,
        dy=dy,
    )

    # Shift each point by its fitted band offset so all bands lie on one curve
    y_aligned = y - np.array([band_offsets[band] for band in b], dtype=float)

    phase_grid = np.linspace(0.0, 1.0, 800, endpoint=True)
    y_fit = eval_fourier_series(phase_grid, common_beta, order=order)
    phase_grid_2 = np.concatenate([phase_grid, phase_grid + 1.0])
    y_fit_2 = np.concatenate([y_fit, y_fit])


    plt.figure(figsize=FIGSIZE)

    band_style = {
        "g": {"color": "blue",   "marker": "s"},
        "i": {"color": "orange", "marker": "D"},
        "r": {"color": "green",  "marker": "^"},
        "z": {"color": "red",    "marker": "v"},
        "u": {"color": "purple", "marker": "o"},
    }

    for band in sorted(pd.unique(b)):
        m = (b == band)
        style = band_style.get(band, {"color": "gray", "marker": "o"})

        phase_plot = np.concatenate([phase[m], phase[m] + 1.0])
        y_plot = np.concatenate([y_aligned[m], y_aligned[m]])

        if dy is not None:
            dy_plot = np.concatenate([dy[m], dy[m]])
            plt.errorbar(
                phase_plot,
                y_plot,
                yerr=dy_plot,
                fmt=style["marker"],
                ms=5,
                capsize=0,
                alpha=0.85,
                label=band,
                color=style["color"],
                markerfacecolor=style["color"],
                markeredgecolor="black",
                markeredgewidth=0.5,
                ecolor=style["color"],
                linestyle="None",
                zorder=2,
            )
        else:
            plt.scatter(
                phase_plot,
                y_plot,
                s=30,
                label=band,
                color=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )

    plt.plot(
        phase_grid_2,
        y_fit_2,
        lw=3.0,
        color="black",
        label="Shared Fourier fit",
        zorder=10,
    )

    plt.xlim(0.0, 1.0)  # phase: choose 1.0 or 2.0
    plt.xlabel("Phase")
    plt.ylabel("Corrected magnitude")

    if target_name is None:
        title = f"LSM: phased lightcurve, P = {best_period_days * 24.0:.4f} h"
    else:
        title = f"LSM: {target_name}, phased lightcurve, P = {best_period_days * 24.0:.4f} h"

    plt.title(title)
    plt.gca().invert_yaxis()   
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input photometry CSV")
    parser.add_argument("--target", default="2026 DO14", help="Horizons target designation")
    parser.add_argument("--location", default="500", help='Horizons observer code; e.g. "X05" or "500"')
    parser.add_argument("--id-type", default=None, help="Optional Horizons id_type")
    parser.add_argument("--step-minutes", type=int, default=1, help="Horizons ephemeris step size in minutes")
    parser.add_argument("--pad-minutes", type=int, default=10, help="Padding around observation window")
    parser.add_argument("--min-period-days", type=float, default=0.00065)
    parser.add_argument("--max-period-days", type=float, default=3.0)
    parser.add_argument("--oversample", type=int, default=100)
    parser.add_argument(
        "--max-plot-hours",
        type=float,
        default=None,
        help="Optional fixed x-axis max (hours) for the periodogram. If omitted, scales to ~1.5x best period.",
    )
    parser.add_argument("--out-prefix", default="2026_DO14")
    parser.add_argument(
        "--auto-zoom-periodogram",
        action="store_true",
        help="Automatically zoom periodogram x-axis around the candidate region",
    )
    parser.add_argument("--cache-dir", default="HORIZONS_CACHE", help="Directory for cached Horizons queries")
    parser.add_argument("--no-cache", action="store_true", help="Disable Horizons cache")

    args = parser.parse_args()

    mls_dir = Path("MULTIBAND_LOMB_SCARGLE")
    figures_dir = Path("FIGURES")
    mls_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    base_prefix = f"{args.out_prefix}_Multiband_Lomb_Scargle"

    print("Loading photometry...")
    df = load_and_prepare_csv(args.csv)
    print(f"Loaded {len(df)} rows")
    print(f"Bands: {sorted(df['band'].unique().tolist())}")
    print(f"MJD span: {df['mjd'].min():.8f} to {df['mjd'].max():.8f}")

    print("Querying JPL Horizons over a continuous time range...")
    eph_df = query_horizons_range(
        target_id=args.target,
        mjd_min=float(df["mjd"].min()),
        mjd_max=float(df["mjd"].max()),
        location=args.location,
        id_type=args.id_type,
        step_minutes=args.step_minutes,
        pad_minutes=args.pad_minutes,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
    )

    eph_csv = mls_dir / f"{base_prefix}_horizons_range.csv"
    eph_df.to_csv(eph_csv, index=False)
    print(f"Wrote dense Horizons table: {eph_csv.resolve()}")

    merged = interpolate_ephemerides(df, eph_df)
    merged = build_corrected_lightcurve(merged)

    merged_csv = mls_dir / f"{base_prefix}_with_horizons_interp.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"Wrote merged/interpolated table: {merged_csv.resolve()}")

    n_in = int(merged["is_inlier"].sum())
    n_out = int((~merged["is_inlier"]).sum())
    print(f"Inliers: {n_in}, outliers: {n_out}")

    print("Running multiband Lomb-Scargle...")
    periods, power, model = fit_multiband_periodogram(
        merged,
        min_period_days=args.min_period_days,
        max_period_days=args.max_period_days,
        oversample=args.oversample,
    )

    candidates, best = select_candidate_periods(
        df=merged,
        periods=periods,
        power=power,
        top_n_local=80,
    )

    if not candidates:
        raise RuntimeError("No candidate periods were found.")

    best_hours = best.period_days * 24.0
    use = merged[merged["is_inlier"]].copy()

    t = use["t_corr_mjd"].to_numpy(dtype=float)
    phase = phase_fold(t, best.period_days, t0=np.nanmin(t))
    bands_arr = use["band"].astype(str).to_numpy()

    if "rmsmag" in use.columns:
        dy = use["rmsmag"].to_numpy(dtype=float)
        bad = ~np.isfinite(dy) | (dy <= 0)
        if np.all(bad):
            dy = None
        else:
            med = np.nanmedian(dy[~bad])
            dy = dy.copy()
            dy[bad] = med
    else:
        dy = None

    # ------------------------------------------------------------
    # 1) COLOR FIT: use raw corrected magnitudes (corr_mag)
    # ------------------------------------------------------------
    y_color = use["corr_mag"].to_numpy(dtype=float)
    common_beta_color, band_offsets, ref_band, _ = fit_shared_fourier_with_band_offsets(
        phase=phase,
        y=y_color,
        bands=bands_arr,
        order=2,
        dy=dy,
    )
    colors = compute_colors_from_offsets(band_offsets)

    # ------------------------------------------------------------
    # 2) AMPLITUDE FIT: use band-median-corrected magnitudes
    #    (corr_mag_centered), consistent with period search / plot
    # ------------------------------------------------------------
    y_amp = use["corr_mag_centered"].to_numpy(dtype=float)
    common_beta_amp, band_offsets_amp, _, cov_amp = fit_shared_fourier_with_band_offsets(
        phase=phase,
        y=y_amp,
        bands=bands_arr,
        order=2,
        dy=dy,
    )
    amplitude_mag = compute_lightcurve_amplitude_from_beta(common_beta_amp, order=2)
    amplitude_error = compute_amplitude_error_from_covariance(common_beta_amp, cov_amp, order=2)



    # ------------------------------------------------------------
    # 3) FORMAL PERIOD ERROR: Montgomery & O'Donoghue (1999)
    # ------------------------------------------------------------
    # Full fitted model = common Fourier shape + fitted band offset
    yhat_amp = eval_fourier_series(phase, common_beta_amp, order=2)
    yhat_amp = yhat_amp + np.array(
        [band_offsets_amp[band] for band in bands_arr],
        dtype=float,
    )

    residuals_amp = y_amp - yhat_amp

    n_amp_params = len(common_beta_amp) + max(0, len(set(bands_arr)) - 1)
    dof_amp = max(1, len(y_amp) - n_amp_params)

    residual_rms_amp = float(
        np.sqrt(np.sum(residuals_amp**2) / dof_amp)
    )

    time_span_days_amp = float(np.nanmax(t) - np.nanmin(t))
    semi_amplitude_mag = 0.5 * amplitude_mag

    period_error_montgomery_days = np.nan
    period_error_montgomery_hours = np.nan

    if (
        np.isfinite(semi_amplitude_mag)
        and semi_amplitude_mag > 0
        and np.isfinite(residual_rms_amp)
        and residual_rms_amp >= 0
    ):
        period_error_montgomery_days = compute_period_error(
            period_days=best.period_days,
            amplitude=semi_amplitude_mag,
            n_data_points=len(y_amp),
            time_span_days=time_span_days_amp,
            residual_rms=residual_rms_amp,
        )
        period_error_montgomery_hours = 24.0 * period_error_montgomery_days




    print(f"Lightcurve amplitude: {amplitude_mag:.4f} mag")
    if amplitude_error is not None and np.isfinite(amplitude_error):
        print(f"Lightcurve amplitude error: {amplitude_error:.4f} mag")

    if np.isfinite(period_error_montgomery_days):
        print(
            f"Formal Montgomery period error: "
            f"{period_error_montgomery_days:.8f} d = "
            f"{period_error_montgomery_hours:.5f} h"
        )
        print(f"Residual RMS used for period error: {residual_rms_amp:.5f} mag")
        
    print(f"Reference band for offsets: {ref_band}")

    print("\nDerived colors from fitted offsets:")
    for name, value in colors.items():
        if np.isfinite(value):
            print(f"  {name} = {value:.6f}")
        else:
            print(f"  {name} = NaN")

    print("\nTop candidates:")
    for c in candidates[:10]:
        print(
            f"  P = {c.period_days:.8f} d "
            f"({c.period_days * 24.0:.5f} h), "
            f"power = {c.power:.6f}, minima = {c.n_minima}"
        )

    print(f"\nBest period: {best.period_days:.8f} d = {best_hours:.5f} h")

    summary_txt = mls_dir / f"{base_prefix}_best_period.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"target={args.target}\n")
        f.write(f"location={args.location}\n")
        f.write(f"step_minutes={args.step_minutes}\n")
        f.write(f"best_period_days={best.period_days:.10f}\n")
        f.write(f"best_period_hours={best_hours:.10f}\n")
        f.write(f"period_error_montgomery_days={period_error_montgomery_days:.10f}\n")
        f.write(f"period_error_montgomery_hours={period_error_montgomery_hours:.10f}\n")
        f.write(f"residual_rms_montgomery_mag={residual_rms_amp:.10f}\n")
        f.write(f"semi_amplitude_mag={semi_amplitude_mag:.10f}\n")
        f.write(f"power={best.power:.10f}\n")
        f.write(f"n_minima={best.n_minima}\n")
        f.write(f"amplitude_mag={amplitude_mag:.10f}\n")
        if amplitude_error is not None and np.isfinite(amplitude_error):
            f.write(f"amplitude_error_mag={amplitude_error:.10f}\n")
        else:
            f.write(f"amplitude_error_mag=NaN\n")
        f.write(f"reference_band={ref_band}\n")
        for band in sorted(band_offsets):
            f.write(f"offset_{band}_minus_{ref_band}={band_offsets[band]:.10f}\n")
        for name in ["g-r", "g-i", "r-i", "i-z"]:
            val = colors.get(name, np.nan)
            f.write(f"color_{name.replace('-', '_')}={val:.10f}\n")
    print(f"Wrote summary: {summary_txt.resolve()}")

    periodogram_png = figures_dir / f"{base_prefix}_periodogram.png"
    phased_png = figures_dir / f"{base_prefix}_phased.png"

    plot_periodogram(
        periods=periods,
        power=power,
        candidates=candidates,
        best=best,
        out_png=periodogram_png,
        max_plot_hours=args.max_plot_hours,
        target_name=args.target,
        auto_zoom=args.auto_zoom_periodogram,
    )

    plot_phased_lightcurve(
        merged,
        best.period_days,
        phased_png,
        order=2,
        target_name=args.target,
    )

    print(f"Wrote plots: {periodogram_png.resolve()}, {phased_png.resolve()}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

# python Period_search_Multiband_Lomb_Scargle.py --csv MU8.csv --target "2025 MU8" --out-prefix 2025_MU8 --min-period-days 0.03333 --max-period-days 0.03458
# python Period_search_Multiband_Lomb_Scargle.py --csv DO14.csv --target "2026 DO14" --out-prefix 2026_DO14

