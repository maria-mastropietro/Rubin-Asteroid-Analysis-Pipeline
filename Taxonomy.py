#!/usr/bin/env python3
"""
Estimate the rotational period of an asteroid from multiband photometry,
using a continuous JPL Horizons ephemeris range plus interpolation.

This version also derives color differences following the paper's LSM-style
approach: fit one shared periodic signal across all bands while allowing each
band to have an independent magnitude offset, set the r-band offset to 0, and
use the fitted band offsets to compute colors.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from astroquery.jplhorizons import Horizons
from astropy.timeseries import LombScargleMultiband
from astropy.time import Time


def format_float_safe(x: object, fmt: str = ".10f") -> str:
    try:
        if x is None:
            return "nan"
        val = float(x)
        if not np.isfinite(val):
            return "nan"
        return format(val, fmt)
    except (TypeError, ValueError):
        return "nan"
    
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

SUN_APP_MAGS = {
    "g": -26.34,
    "r": -27.04,
    "i": -27.38,
    "z": -27.56,
}

FILTER_WAVELENGTH_NM = {
    "g": 467.3,
    "r": 614.2,
    "i": 745.9,
    "z": 892.5,
}


TAXONOMY_BOXES = {
    "A": (21.5, 28.0, -0.265, -0.115, (0.4, 1.0, 0.1), 22.0, -0.175),
    "B": (-5.0,  0.0, -0.200,  0.000, "c",             -4.5, -0.060),
    "C": (-5.0,  6.0, -0.200,  0.185, "b",             -4.5,  0.125),
    "D": ( 6.0, 25.0,  0.085,  0.335, (0.5, 0.2, 0.0),  6.5,  0.275),
    "X": ( 2.5,  9.0, -0.005,  0.185, "g",              3.0,  0.125),
    "S": ( 6.0, 25.0, -0.265, -0.005, "r",              6.5, -0.065),
    "L": ( 9.0, 25.0, -0.005,  0.085, "m",              9.5,  0.025),
    "Q": ( 5.0,  9.5, -0.265, -0.165, (1.0, 0.5, 0.2),  8.3, -0.225),
    "V": ( 5.0, 25.0, -0.665, -0.265, (0.7, 0.0, 0.2),  5.5, -0.325),
}


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


def make_frequency_grid(
    min_period_days: float = 0.00065,
    max_period_days: float = 3.0,
    n_freq: int = 50000,
) -> np.ndarray:
    fmin = 1.0 / max_period_days
    fmax = 1.0 / min_period_days
    return np.linspace(fmin, fmax, n_freq)


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


def query_horizons_range(
    target_id: str,
    mjd_min: float,
    mjd_max: float,
    location: str = "500",
    id_type: Optional[str] = None,
    step_minutes: int = 1,
    pad_minutes: int = 10,
) -> pd.DataFrame:
    start_mjd = mjd_min - pad_minutes / 1440.0
    stop_mjd = mjd_max + pad_minutes / 1440.0

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

    return out


def interpolate_ephemerides(obs_df: pd.DataFrame, eph_df: pd.DataFrame) -> pd.DataFrame:
    x = eph_df["mjd"].to_numpy(dtype=float)
    ycols = ["pred_V", "r_au", "delta_au", "alpha_deg", "RA_deg", "DEC_deg"]

    out = obs_df.copy()
    xo = out["mjd"].to_numpy(dtype=float)

    xmin = np.nanmin(x)
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

    out["corr_mag"] = out["mag"] - out["pred_V"]
    med = np.nanmedian(out["corr_mag"].to_numpy(dtype=float))
    out["corr_mag_centered"] = out["corr_mag"] - med
    out["is_inlier"] = sigma_clip_mask(out["corr_mag_centered"].to_numpy(dtype=float), nsig=3.0)

    return out


@dataclass
class CandidatePeriod:
    period_days: float
    power: float
    n_minima: int


@dataclass
class ColorFitResult:
    period_days: float
    t0_mjd: float
    base_beta: np.ndarray
    band_offsets_relative_to_r: Dict[str, float]
    colors: Dict[str, float]
    chi2: float
    dof: int
    rms_resid: float
    cov: Optional[np.ndarray] = None
    color_errors: Optional[Dict[str, Optional[float]]] = None
    gri_slope_error: Optional[float] = None


@dataclass
class WeightedSeries:
    values: np.ndarray
    errors: Optional[np.ndarray]


def compute_gri_slope_from_colors(gr: float, ri: float) -> float:
    if not (np.isfinite(gr) and np.isfinite(ri)):
        return np.nan

    rg = -gr
    ig = -(gr + ri)
    r_ig = np.power(10.0, -0.4 * (ig - (SUN_APP_MAGS["i"] - SUN_APP_MAGS["g"])))
    r_rg = np.power(10.0, -0.4 * (rg - (SUN_APP_MAGS["r"] - SUN_APP_MAGS["g"])))
    return float(10000.0 * (r_rg - r_ig) / abs(FILTER_WAVELENGTH_NM["r"] - FILTER_WAVELENGTH_NM["i"]))


def compute_gri_slope_error_from_covariance(
    colors: Dict[str, float],
    fitted_offset_bands: List[str],
    cov: Optional[np.ndarray],
    reference_band: str = "r",
    order: int = 2,
) -> Optional[float]:
    """
    Propagate the covariance matrix of fitted band offsets into the gri-slope error.

    gri slope depends on:
        g-r = offset_g - offset_r
        r-i = offset_r - offset_i

    Since r is the reference band, offset_r = 0.
    Therefore:
        g-r = offset_g
        r-i = -offset_i
    """

    if cov is None or not np.isfinite(cov).all():
        return None

    gr = colors.get("g-r", np.nan)
    ri = colors.get("r-i", np.nan)

    if not (np.isfinite(gr) and np.isfinite(ri)):
        return None

    # Same quantities used in compute_gri_slope_from_colors()
    rg = -gr
    ig = -(gr + ri)

    r_rg = np.power(
        10.0,
        -0.4 * (rg - (SUN_APP_MAGS["r"] - SUN_APP_MAGS["g"]))
    )
    r_ig = np.power(
        10.0,
        -0.4 * (ig - (SUN_APP_MAGS["i"] - SUN_APP_MAGS["g"]))
    )

    k_slope = 10000.0 / abs(FILTER_WAVELENGTH_NM["r"] - FILTER_WAVELENGTH_NM["i"])
    a = 0.4 * np.log(10.0)

    # Partial derivatives of gri_slope with respect to colors
    dS_dgr = k_slope * a * (r_rg - r_ig)
    dS_dri = -k_slope * a * r_ig

    # Build gradient with respect to the full beta vector.
    # The first parameters are Fourier coefficients; the band offsets start after them.
    n_base = 1 + 2 * order

    band_indices = {}
    for j, band in enumerate(fitted_offset_bands, start=n_base):
        band_indices[band] = j

    grad = np.zeros(cov.shape[0], dtype=float)

    # g-r = offset_g, so dS/d(offset_g) = dS/d(g-r)
    idx_g = band_indices.get("g")
    if idx_g is not None:
        grad[idx_g] += dS_dgr
    else:
        return None

    # r-i = -offset_i, so dS/d(offset_i) = -dS/d(r-i)
    idx_i = band_indices.get("i")
    if idx_i is not None:
        grad[idx_i] += -dS_dri
    else:
        return None

    var_slope = float(grad @ cov @ grad.T)

    if not np.isfinite(var_slope) or var_slope < 0:
        return None

    return float(np.sqrt(var_slope))


def classify_taxonomy_demeo_carry(gri_slope: float, iz: float) -> str:
    if not (np.isfinite(gri_slope) and np.isfinite(iz)):
        return "nan"

    tax = "nan"
    if (-0.2 <= iz <= 0.185) and (-5.0 <= gri_slope <= 6.0):
        tax = "C"
    if (-0.2 <= iz <= 0.0) and (-5.0 <= gri_slope <= 0.0):
        tax = "B"
    if (-0.005 <= iz <= 0.185) and (2.5 <= gri_slope <= 9.0):
        tax = "X"
    if (0.085 <= iz <= 0.335) and (6.0 <= gri_slope <= 25.0):
        tax = "D"
    if (-0.005 <= iz <= 0.085) and (9.0 <= gri_slope <= 25.0):
        tax = "L"
    if (-0.265 <= iz <= -0.005) and (6.0 <= gri_slope <= 25.0):
        tax = "S"
    if (-0.265 <= iz <= -0.165) and (5.0 <= gri_slope <= 9.5):
        tax = "Q"
    if (-0.265 <= iz <= -0.115) and (21.5 <= gri_slope <= 28.0):
        tax = "A"
    if (-0.665 <= iz <= -0.265) and (5.0 <= gri_slope <= 25.0):
        tax = "V"
    return tax


def build_taxonomy_from_color_fit(color_fit: ColorFitResult) -> Dict[str, float]:
    offsets = color_fit.band_offsets_relative_to_r
    g_off = offsets.get("g", np.nan)
    r_off = offsets.get("r", 0.0)
    i_off = offsets.get("i", np.nan)
    z_off = offsets.get("z", np.nan)

    g_mag = g_off
    r_mag = r_off
    i_mag = i_off + r_mag if np.isfinite(i_off) else np.nan
    z_mag = z_off + r_mag if np.isfinite(z_off) else np.nan

    gr = color_fit.colors.get("g-r", np.nan)
    ri = color_fit.colors.get("r-i", np.nan)
    iz = color_fit.colors.get("i-z", np.nan)
    gri_slope = compute_gri_slope_from_colors(gr, ri)
    taxonomy = classify_taxonomy_demeo_carry(gri_slope, iz)

    return {
        "taxonomy": taxonomy,
        "g_mag": g_mag,
        "r_mag": r_mag,
        "i_mag": i_mag,
        "z_mag": z_mag,
        "gr": gr,
        "ri": ri,
        "iz": iz,
        "gri_slope": gri_slope,
        "gri_slope_error": color_fit.gri_slope_error,
    }


def build_taxonomy_band_summary(df: pd.DataFrame, color_fit: ColorFitResult, reference_band: str = "r") -> pd.DataFrame:
    rows = []
    offsets = color_fit.band_offsets_relative_to_r
    for band, gb in df.groupby("band", sort=True):
        mags = pd.to_numeric(gb["mag"], errors="coerce")
        mags = mags[np.isfinite(mags)]
        if len(mags) == 0:
            median_mag = np.nan
            mad_mag = np.nan
        else:
            median_mag = float(np.nanmedian(mags))
            mad_mag = float(np.nanmedian(np.abs(mags - median_mag)))
        rows.append({
            "band": str(band),
            "n_obs": int(len(gb)),
            "median_mag": median_mag,
            "mad_mag": mad_mag,
            f"offset_{band}_minus_{reference_band}": offsets.get(str(band), np.nan),
        })
    return pd.DataFrame(rows).sort_values("band").reset_index(drop=True)


def get_weighted_series(df: pd.DataFrame, value_col: str) -> WeightedSeries:
    values = df[value_col].to_numpy(dtype=float)
    if "rmsmag" not in df.columns:
        return WeightedSeries(values=values, errors=None)

    dy = df["rmsmag"].to_numpy(dtype=float)
    if not np.isfinite(dy).any():
        return WeightedSeries(values=values, errors=None)

    bad = ~np.isfinite(dy) | (dy <= 0)
    if np.all(bad):
        return WeightedSeries(values=values, errors=None)

    med = np.nanmedian(dy[~bad])
    dy = dy.copy()
    dy[bad] = med
    return WeightedSeries(values=values, errors=dy)


def fit_multiband_periodogram(
    df: pd.DataFrame,
    min_period_days: float = 0.00065,
    max_period_days: float = 3.0,
    n_freq: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, LombScargleMultiband]:
    use = df[df["is_inlier"]].copy()
    if len(use) < 10:
        raise RuntimeError("Too few inlier points for a period search.")

    t = use["mjd"].to_numpy(dtype=float)
    bands = use["band"].astype(str).to_numpy()
    ws = get_weighted_series(use, "corr_mag_centered")

    model = LombScargleMultiband(
        t=t,
        y=ws.values,
        dy=ws.errors,
        bands=bands,
        nterms_base=2,
        nterms_band=1,
        reg_base=None,
        reg_band=1e-6,
    )

    freq = make_frequency_grid(
        min_period_days=min_period_days,
        max_period_days=max_period_days,
        n_freq=n_freq,
    )
    power = model.power(freq)
    return freq, power, model


def fit_fourier_series(
    phase: np.ndarray,
    y: np.ndarray,
    order: int = 2,
    dy: Optional[np.ndarray] = None,
) -> np.ndarray:
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
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            return beta

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


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


def _build_multiband_design_matrix(
    phase: np.ndarray,
    bands: np.ndarray,
    reference_band: str = "r",
    order: int = 2,
) -> Tuple[np.ndarray, List[str]]:
    phase = np.asarray(phase, dtype=float) % 1.0
    bands = np.asarray(bands, dtype=str)

    cols = [np.ones_like(phase)]
    for k in range(1, order + 1):
        cols.append(np.cos(2.0 * np.pi * k * phase))
        cols.append(np.sin(2.0 * np.pi * k * phase))

    fitted_offset_bands = [b for b in sorted(np.unique(bands).tolist()) if b != reference_band]
    for band in fitted_offset_bands:
        cols.append((bands == band).astype(float))

    X = np.column_stack(cols)
    return X, fitted_offset_bands


def fit_shared_multiband_fourier_with_offsets(
    df: pd.DataFrame,
    period_days: float,
    order: int = 2,
    reference_band: str = "r",
) -> ColorFitResult:
    use = df[df["is_inlier"]].copy()
    if len(use) < 5:
        raise RuntimeError("Too few inlier points for color fitting.")

    t = use["mjd"].to_numpy(dtype=float)
    bands = use["band"].astype(str).to_numpy()
    ws = get_weighted_series(use, "corr_mag")

    t0 = float(np.nanmin(t))
    phase = phase_fold(t, period_days, t0=t0)
    X, fitted_offset_bands = _build_multiband_design_matrix(phase, bands, reference_band=reference_band, order=order)

    if ws.errors is not None:
        w = 1.0 / ws.errors
        Xw = X * w[:, None]
        yw = ws.values * w
        beta, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
        
        # Compute covariance matrix
        n_params = X.shape[1]
        dof = len(yw) - n_params
        if dof > 0 and rank == n_params:
            # Compute MSE in original data space: Σ w_i (y_i - ŷ_i)^2 / dof
            yhat = X @ beta
            mse = np.sum(w * (ws.values - yhat)**2) / dof
            if np.isfinite(mse) and mse > 0:
                XtWX = X.T @ (w[:, None] * X)
                try:
                    cov = mse * np.linalg.inv(XtWX)
                except np.linalg.LinAlgError:
                    cov = None
            else:
                cov = None
        else:
            cov = None
    else:
        beta, residuals, rank, s = np.linalg.lstsq(X, ws.values, rcond=None)
        
        # Compute covariance matrix for unweighted fit
        n_params = X.shape[1]
        dof = len(ws.values) - n_params
        if dof > 0 and rank == n_params:
            yhat = X @ beta
            mse = np.sum((ws.values - yhat)**2) / dof
            if np.isfinite(mse) and mse > 0:
                try:
                    cov = mse * np.linalg.inv(X.T @ X)
                except np.linalg.LinAlgError:
                    cov = None
            else:
                cov = None
        else:
            cov = None

    y_model = X @ beta
    resid = ws.values - y_model

    if ws.errors is not None:
        chi2 = float(np.sum((resid / ws.errors) ** 2))
    else:
        chi2 = float(np.sum(resid ** 2))
    dof = max(len(ws.values) - len(beta), 0)
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))

    n_base = 1 + 2 * order
    base_beta = beta[:n_base].copy()

    offsets = {reference_band: 0.0}
    for j, band in enumerate(fitted_offset_bands, start=n_base):
        offsets[band] = float(beta[j])

    all_possible_bands = ["u", "g", "r", "i", "z", "y"]
    for band in all_possible_bands:
        offsets.setdefault(band, np.nan)

    def color(a: str, b: str) -> float:
        oa = offsets.get(a, np.nan)
        ob = offsets.get(b, np.nan)
        if not (np.isfinite(oa) and np.isfinite(ob)):
            return np.nan
        return float(oa - ob)

    colors = {
        "g-r": color("g", "r"),
        "g-i": color("g", "i"),
        "r-i": color("r", "i"),
        "i-z": color("i", "z"),
    }

    # Compute color errors from covariance matrix
    color_errors = compute_color_errors_from_covariance(
        offsets, fitted_offset_bands, cov, reference_band=reference_band, order=order,
    )


    gri_slope_error = compute_gri_slope_error_from_covariance(
        colors=colors,
        fitted_offset_bands=fitted_offset_bands,
        cov=cov,
        reference_band=reference_band,
        order=order,
    )
    
    return ColorFitResult(
        period_days=float(period_days),
        t0_mjd=t0,
        base_beta=base_beta,
        band_offsets_relative_to_r=offsets,
        colors=colors,
        chi2=chi2,
        dof=dof,
        rms_resid=rms_resid,
        cov=cov,
        color_errors=color_errors,
        gri_slope_error=gri_slope_error,
    )


def compute_color_errors_from_covariance(
    band_offsets: Dict[str, float],
    fitted_offset_bands: List[str],
    cov: Optional[np.ndarray],
    reference_band: str = "r",
    order: int = 2,
) -> Dict[str, Optional[float]]:
    """
    Compute color index errors using error propagation from the covariance matrix.

    For a color index like g-r = offset_g - offset_r, the variance is:
    var(g-r) = var(offset_g) + var(offset_r) - 2*cov(offset_g, offset_r)

    Since offset_r is the reference (fixed at 0), var(offset_r) = 0 and cov terms with r are 0.
    So var(g-r) = var(offset_g) for colors involving the reference band.
    """
    if cov is None or not np.isfinite(cov).all():
        return {name: None for name in ["g-r", "g-i", "r-i", "i-z"]}

    # Map band names to their indices in the beta vector
    # The design matrix has: [c0, a1, b1, a2, b2, ..., offset_g, offset_i, offset_z, ...]
    # We need to find which indices correspond to which band offsets
    n_base = 1 + 2 * order  # c0 + 2*(a1, b1) for the given order
    band_indices = {}
    for j, band in enumerate(fitted_offset_bands, start=n_base):
        band_indices[band] = j

    # Get the covariance submatrix for the band offsets
    offset_indices = [band_indices.get(b) for b in fitted_offset_bands if b in band_indices]
    if not offset_indices:
        return {name: None for name in ["g-r", "g-i", "r-i", "i-z"]}

    cov_offsets = cov[np.ix_(offset_indices, offset_indices)]

    # Create a mapping from band name to its index in the offset covariance matrix
    offset_idx_map = {band: i for i, band in enumerate(fitted_offset_bands) if band in band_indices}

    def color_error(a: str, b: str) -> Optional[float]:
        """Compute error for color a-b."""
        # If both bands are the reference band, error is 0
        if a == reference_band and b == reference_band:
            return 0.0

        # If one band is the reference, error is just the variance of the other band
        if a == reference_band:
            idx_b = offset_idx_map.get(b)
            if idx_b is None:
                return None
            var = cov_offsets[idx_b, idx_b]
            return float(np.sqrt(var)) if var > 0 else None
        if b == reference_band:
            idx_a = offset_idx_map.get(a)
            if idx_a is None:
                return None
            var = cov_offsets[idx_a, idx_a]
            return float(np.sqrt(var)) if var > 0 else None

        # If neither band is the reference, need full error propagation
        idx_a = offset_idx_map.get(a)
        idx_b = offset_idx_map.get(b)
        if idx_a is None or idx_b is None:
            return None

        var_a = cov_offsets[idx_a, idx_a]
        var_b = cov_offsets[idx_b, idx_b]
        cov_ab = cov_offsets[idx_a, idx_b]

        var_diff = var_a + var_b - 2 * cov_ab
        return float(np.sqrt(var_diff)) if var_diff > 0 else None

    color_errors = {}
    color_errors["g-r"] = color_error("g", "r")
    color_errors["g-i"] = color_error("g", "i")
    color_errors["r-i"] = color_error("r", "i")
    color_errors["i-z"] = color_error("i", "z")

    return color_errors


def evaluate_candidate_minima(
    df: pd.DataFrame,
    period_days: float,
    n_phase: int = 500,
    order: int = 2,
) -> int:
    use = df[df["is_inlier"]].copy()
    ws = get_weighted_series(use, "corr_mag_centered")

    t = use["mjd"].to_numpy(dtype=float)
    phase = phase_fold(t, period_days, t0=np.nanmin(t))
    beta = fit_fourier_series(phase, ws.values, order=order, dy=ws.errors)

    phase_grid = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    y_fit = eval_fourier_series(phase_grid, beta, order=order)

    return count_local_minima(y_fit)


def select_candidate_periods(
    df: pd.DataFrame,
    freq: np.ndarray,
    power: np.ndarray,
    top_n: int = 80,
) -> List[CandidatePeriod]:
    idx = np.argsort(power)[::-1][:top_n]
    cands = []

    for i in idx:
        period = 1.0 / freq[i]
        n_minima = evaluate_candidate_minima(
            df=df,
            period_days=period,
            n_phase=500,
            order=2,
        )
        cands.append(
            CandidatePeriod(
                period_days=period,
                power=float(power[i]),
                n_minima=n_minima,
            )
        )

    two_min = [c for c in cands if c.n_minima == 2]
    return sorted(two_min if two_min else cands, key=lambda c: c.power, reverse=True)


def draw_taxonomy_boxes(ax: plt.Axes) -> None:
    for name, (xmin, xmax, ymin, ymax, color, tx, ty) in TAXONOMY_BOXES.items():
        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=color,
            linewidth=2.2,
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            tx,
            ty,
            name,
            color=color,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=2,
        )


def plot_taxonomy_point(
    taxonomy_info: Dict[str, float],
    target_label: str,
    out_prefix: str,
    xlim: Tuple[float, float] = (-10.0, 30.0),
    ylim: Tuple[float, float] = (-0.8, 0.4),
) -> List[str]:
    gri_slope = taxonomy_info.get("gri_slope", np.nan)
    iz = taxonomy_info.get("iz", np.nan)
    taxonomy = taxonomy_info.get("taxonomy", "nan")

    if not (np.isfinite(gri_slope) and np.isfinite(iz)):
        raise RuntimeError("Cannot plot taxonomy point because gri_slope and/or i-z are not finite.")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_taxonomy_boxes(ax)
    ax.scatter(
        [gri_slope],
        [iz],
        s=140,
        marker="*",
        edgecolors="black",
        linewidths=0.9,
        zorder=5,
        label=target_label,
    )
    ax.annotate(
        target_label,
        xy=(gri_slope, iz),
        xytext=(8, 8),
        textcoords="offset points",
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
    )

    ax.set_title(f"Asteroid taxonomic classification ({target_label}: {taxonomy})")
    ax.set_xlabel("gri slope [%/100nm]")
    ax.set_ylabel("i-z [mag]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)

    subtitle = (
        f"Point: gri_slope = {gri_slope:.4f}, i-z = {iz:.4f}\n"
        f"taxonomy = {taxonomy}"
    )
    fig.text(0.02, 0.02, subtitle)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = f"{out_prefix}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def plot_periodogram(freq: np.ndarray, power: np.ndarray, out_png: str) -> None:
    periods_hr = 24.0 / freq
    plt.figure(figsize=FIGSIZE)
    plt.plot(periods_hr, power)
    plt.xscale("log")
    plt.xlabel("Period (hours)")
    plt.ylabel("Power")
    plt.title("Multiband Lomb-Scargle Periodogram")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_phased_lightcurve(
    df: pd.DataFrame,
    best_period_days: float,
    out_png: str,
    order: int = 2,
) -> None:
    use = df[df["is_inlier"]].copy()
    ws = get_weighted_series(use, "corr_mag_centered")
    t = use["mjd"].to_numpy(dtype=float)
    y = ws.values
    b = use["band"].astype(str).to_numpy()

    phase = phase_fold(t, best_period_days, t0=np.nanmin(t))
    beta = fit_fourier_series(phase, y, order=order, dy=ws.errors)

    phase_grid = np.linspace(0.0, 2.0, 1000)
    y_fit = eval_fourier_series(phase_grid % 1.0, beta, order=order)

    plt.figure(figsize=FIGSIZE)
    for band in sorted(pd.unique(b)):
        m = b == band
        plt.scatter(phase[m], y[m], s=18, label=band)
        plt.scatter(phase[m] + 1.0, y[m], s=18)

    plt.plot(phase_grid, y_fit, lw=2.5, label="Fourier fit")

    plt.xlabel("Phase")
    plt.ylabel("Corrected magnitude (centered)")
    plt.title(f"Phased Lightcurve, P = {best_period_days * 24.0:.4f} h")
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
    parser.add_argument("--n-freq", type=int, default=50000)
    parser.add_argument("--out-prefix", default="2026_DO14")
    parser.add_argument("--taxonomy-plot-prefix", default=None, help="Optional prefix for taxonomy-point plot outputs; defaults to FIGURES/<out-prefix>_taxonomy_point")
    args = parser.parse_args()

    taxonomy_dir = Path("TAXONOMY")
    figures_dir = Path("FIGURES")
    taxonomy_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    summary_txt = taxonomy_dir / f"{args.out_prefix}_summary.txt"
    colors_csv = taxonomy_dir / f"{args.out_prefix}_colors.csv"
    taxonomy_csv = taxonomy_dir / f"{args.out_prefix}_taxonomy.csv"
    taxonomy_band_csv = taxonomy_dir / f"{args.out_prefix}_taxonomy_band_summary.csv"
    periodogram_png = figures_dir / f"{args.out_prefix}_periodogram.png"
    phased_png = figures_dir / f"{args.out_prefix}_phased.png"
    taxonomy_plot_prefix = str(figures_dir / f"{args.out_prefix}_taxonomy_point")

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
    )

    eph_csv = f"{args.out_prefix}_horizons_range.csv"
    eph_df.to_csv(eph_csv, index=False)
    print(f"Wrote dense Horizons table: {eph_csv}")

    merged = interpolate_ephemerides(df, eph_df)
    merged = build_corrected_lightcurve(merged)

    merged_csv = f"{args.out_prefix}_with_horizons_interp.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"Wrote merged/interpolated table: {merged_csv}")

    n_in = int(merged["is_inlier"].sum())
    n_out = int((~merged["is_inlier"]).sum())
    print(f"Inliers: {n_in}, outliers: {n_out}")

    model_bands = sorted(merged["band"].dropna().astype(str).unique().tolist())

    print("Running multiband Lomb-Scargle...")
    freq, power, _model = fit_multiband_periodogram(
        merged,
        min_period_days=args.min_period_days,
        max_period_days=args.max_period_days,
        n_freq=args.n_freq,
    )

    candidates = select_candidate_periods(
        df=merged,
        freq=freq,
        power=power,
        top_n=80,
    )
    if not candidates:
        raise RuntimeError("No candidate periods were found.")
    best = candidates[0]
    best_hours = best.period_days * 24.0

    print("\nTop candidates:")
    for c in candidates[:10]:
        print(
            f"  P = {c.period_days:.8f} d "
            f"({c.period_days * 24.0:.5f} h), "
            f"power = {c.power:.6f}, minima = {c.n_minima}"
        )

    print(f"\nBest period: {best.period_days:.8f} d = {best_hours:.5f} h")

    color_fit = fit_shared_multiband_fourier_with_offsets(
        merged,
        period_days=best.period_days,
        order=2,
        reference_band="r",
    )

    print("\nDerived color offsets relative to r:")
    for band in model_bands:
        print(f"  {band}-r offset = {color_fit.band_offsets_relative_to_r.get(band, np.nan):.6f}")

    print("\nDerived colors:")
    for name, value in color_fit.colors.items():
        if np.isfinite(value):
            print(f"  {name} = {value:.6f}")
        else:
            print(f"  {name} = NaN")

    taxonomy_info = build_taxonomy_from_color_fit(color_fit)
    taxonomy_band_summary = build_taxonomy_band_summary(merged, color_fit, reference_band="r")

    print("\nDerived taxonomy metrics:")
    for key in ["g_mag", "r_mag", "i_mag", "z_mag", "gr", "ri", "iz", "gri_slope"]:
        value = taxonomy_info.get(key, np.nan)
        if np.isfinite(value):
            print(f"  {key} = {value:.6f}")
        else:
            print(f"  {key} = NaN")
    print(f"  taxonomy = {taxonomy_info['taxonomy']}")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"target={args.target}\n")
        f.write(f"location={args.location}\n")
        f.write(f"step_minutes={args.step_minutes}\n")
        f.write(f"best_period_days={best.period_days:.10f}\n")
        f.write(f"best_period_hours={best_hours:.10f}\n")
        f.write(f"power={best.power:.10f}\n")
        f.write(f"n_minima={best.n_minima}\n")
        f.write(f"color_fit_chi2={color_fit.chi2:.10f}\n")
        f.write(f"color_fit_dof={color_fit.dof}\n")
        f.write(f"color_fit_rms_resid={color_fit.rms_resid:.10f}\n")
        
        for band in ["u", "g", "r", "i", "z", "y"]:
            val = color_fit.band_offsets_relative_to_r.get(band, np.nan)
            f.write(f"offset_{band}_minus_r={format_float_safe(val, '.10f')}\n")

        for name in ["g-r", "g-i", "r-i", "i-z"]:
            val = color_fit.colors.get(name, np.nan)
            f.write(f"color_{name.replace('-', '_')}={format_float_safe(val, '.10f')}\n")

        if color_fit.color_errors is not None:
            for name in ["g-r", "g-i", "r-i", "i-z"]:
                val = color_fit.color_errors.get(name, np.nan)
                f.write(f"color_{name.replace('-', '_')}_error={format_float_safe(val, '.10f')}\n")

        f.write(f"taxonomy={taxonomy_info['taxonomy']}\n")

        for key in ["g_mag", "r_mag", "i_mag", "z_mag", "gr", "ri", "iz", "gri_slope"]:
            val = taxonomy_info.get(key, np.nan)
            f.write(f"{key}={format_float_safe(val, '.10f')}\n")
            
    print(f"Wrote summary: {summary_txt.resolve()}")

    color_row = {
        "target": args.target,
        "best_period_days": best.period_days,
        "best_period_hours": best_hours,
        "power": best.power,
        "n_minima": best.n_minima,
        "color_fit_chi2": color_fit.chi2,
        "color_fit_dof": color_fit.dof,
        "color_fit_rms_resid": color_fit.rms_resid,
    }
    for band, val in color_fit.band_offsets_relative_to_r.items():
        color_row[f"offset_{band}_minus_r"] = val
    for name, val in color_fit.colors.items():
        color_row[f"color_{name}"] = val
    color_row.update(taxonomy_info)
    pd.DataFrame([color_row]).to_csv(colors_csv, index=False)
    print(f"Wrote color summary table: {colors_csv.resolve()}")

    pd.DataFrame([{**{"target": args.target, "best_period_days": best.period_days, "best_period_hours": best_hours}, **taxonomy_info}]).to_csv(taxonomy_csv, index=False)
    print(f"Wrote taxonomy summary table: {taxonomy_csv.resolve()}")

    taxonomy_band_summary.to_csv(taxonomy_band_csv, index=False)
    print(f"Wrote taxonomy band summary table: {taxonomy_band_csv.resolve()}")

    taxonomy_plot_prefix = args.taxonomy_plot_prefix or taxonomy_plot_prefix
    plot_periodogram(freq, power, periodogram_png)
    plot_phased_lightcurve(merged, best.period_days, phased_png, order=2)
    taxonomy_plot_paths = plot_taxonomy_point(
        taxonomy_info=taxonomy_info,
        target_label=args.target,
        out_prefix=taxonomy_plot_prefix,
    )
    print(f"Wrote plots: {periodogram_png.resolve()}, {phased_png.resolve()}, {', '.join(taxonomy_plot_paths)}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
