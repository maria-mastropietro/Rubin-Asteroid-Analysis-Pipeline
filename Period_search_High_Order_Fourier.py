#!/usr/bin/env python3
"""
Estimate an asteroid rotational period from multiband photometry using the
high-order Fourier analysis described in Section 3.1 of Greenstreet et al. (2026).

This script is designed as a companion to the attached Section 3.2 / multiband
Lomb-Scargle script, but follows the Section 3.1 approach more closely:

1) Read multiband photometry from CSV.
2) Query a dense JPL Horizons ephemeris over the observing span.
3) Interpolate r, delta, and phase angle alpha back onto each observation.
4) Correct the observation times for light travel time using delta / c.
5) For each trial frequency, fit
       m - 5 log10(r delta) = H_band + c1 alpha + c2 alpha^2 + g(t)
   where g(t) is a Fourier series of order k = 2..6.
6) Iteratively reject 3-sigma outliers during fitting.
7) For each Fourier order k, identify the best frequency by the weighted,
   unbiased dispersion statistic from Eq. (3) in the paper.
8) Select the preferred Fourier order using pairwise F-tests, preferring the
   lowest-order model unless a more complex one is significantly better.
9) Compute a same-order frequency threshold from an F-test and use it to report
   alternate solutions and a period uncertainty range.
10) Save a summary, candidate table, merged photometry, and diagnostic plots.

Notes
-----
- The paper used Horizons for r, delta, and alpha, then fit a quadratic phase
  function; that is what this script implements.
- For absolute magnitudes of "reliable" objects, the paper later refit with
  HG12*. This script keeps the quadratic phase model throughout.
- The paper sampled frequencies up to roughly 500-1000 per day. In this
  cycles/hour version, the default maximum is 1000/24 = 41.6667 cycles/hour.
- The "period doubling if the fitted lightcurve has one maximum" step is
  implemented approximately by counting maxima on the fitted Fourier model over
  one cycle.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from scipy.stats import f as f_dist

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
HOURS_PER_DAY = 24.0
LIGHT_TIME_DAYS_PER_AU = AU_KM / C_KM_S / SECONDS_PER_DAY


@dataclass
class FrequencyFitResult:
    frequency_cph: float
    period_days: float
    period_days_reported: float
    sigma: float
    chi2: float
    dof: int
    n_obs: int
    n_included: int
    n_params: int
    order: int
    coeffs: np.ndarray
    kept_mask: np.ndarray
    excluded_mask: np.ndarray
    band_names: List[str]
    n_model_maxima: int
    doubled_period: bool
    cov: Optional[np.ndarray] = None


@dataclass
class OrderSelectionResult:
    chosen_order: int
    best_fit: FrequencyFitResult
    best_by_order: Dict[int, FrequencyFitResult]
    comparison_pvalues: Dict[Tuple[int, int], float]


@dataclass
class PossibleSolution:
    frequency_cph: float
    period_days: float
    period_days_reported: float
    sigma: float
    delta_period_days: float


def normalize_band_label(x: str) -> str:
    if pd.isna(x):
        return str(x)
    x = str(x).strip()
    mapping = {
        "Lu": "u", "Lg": "g", "Lr": "r", "Li": "i", "Lz": "z", "Ly": "y",
        "u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "y": "y",
    }
    return mapping.get(x, x)


def load_and_prepare_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"mag", "band"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns at least {sorted(required)}")

    if "mjd" not in df.columns:
        if "obs_time" in df.columns:
            obs_time = pd.to_datetime(df["obs_time"], utc=True, errors="coerce")
            unix_sec = obs_time.astype("int64") / 1e9
            jd = unix_sec / SECONDS_PER_DAY + 2440587.5
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
        "r_au": pd.to_numeric(eph_df.get("r", np.nan), errors="coerce"),
        "delta_au": pd.to_numeric(eph_df.get("delta", np.nan), errors="coerce"),
        "alpha_deg": pd.to_numeric(eph_df.get("alpha", np.nan), errors="coerce"),
        "pred_V": pd.to_numeric(eph_df.get("V", np.nan), errors="coerce"),
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

    if np.nanmin(xo) < np.nanmin(x) or np.nanmax(xo) > np.nanmax(x):
        raise RuntimeError("Observation times fall outside the queried Horizons range.")

    for col in ycols:
        y = eph_df[col].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            out[col] = np.nan
            continue
        out[col] = np.interp(xo, x[finite], y[finite])

    out = out.rename(columns={"RA_deg": "RA_horizons_deg", "DEC_deg": "DEC_horizons_deg"})
    return out


def add_section31_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    needed = ["r_au", "delta_au", "alpha_deg"]
    if any(out[c].isna().all() for c in needed):
        raise RuntimeError("Ephemeris interpolation failed for one or more required columns.")

    out["light_time_days"] = out["delta_au"] * LIGHT_TIME_DAYS_PER_AU
    out["t_corr_mjd"] = out["mjd"] - out["light_time_days"]
    out["mag_reduced"] = out["mag"] - 5.0 * np.log10(out["r_au"] * out["delta_au"])
    return out


def make_weights(rmsmag: np.ndarray) -> np.ndarray:
    dy = np.asarray(rmsmag, dtype=float)
    w = np.ones_like(dy, dtype=float)
    good = np.isfinite(dy) & (dy > 0)
    if np.any(good):
        w[good] = 1.0 / np.square(dy[good])
    return w


def build_design_matrix(
    t_corr: np.ndarray,
    alpha_deg: np.ndarray,
    bands: Sequence[str],
    frequency_cph: float,
    order: int,
    band_names: Sequence[str],
) -> np.ndarray:
    t_corr = np.asarray(t_corr, dtype=float)
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    alpha2 = alpha_deg ** 2

    cols = [alpha_deg, alpha2]

    omega_t = 2.0 * np.pi * frequency_cph * HOURS_PER_DAY * t_corr
    for j in range(1, order + 1):
        cols.append(np.cos(j * omega_t))
        cols.append(np.sin(j * omega_t))

    bands_arr = np.asarray(bands)
    for band in band_names:
        cols.append((bands_arr == band).astype(float))

    return np.column_stack(cols)


def weighted_least_squares(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Perform weighted least squares and return coefficients and covariance matrix.
    """
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
    
    # Compute covariance matrix
    n_params = X.shape[1]
    dof = len(y) - n_params
    if dof > 0 and rank == n_params:
        # Compute MSE in original data space: Σ w_i (y_i - ŷ_i)^2 / dof
        yhat = X @ beta
        mse = np.sum(w * (y - yhat)**2) / dof
        if np.isfinite(mse) and mse > 0:
            # cov(beta) = mse * (X^T W X)^(-1)
            XtWX = X.T @ (w[:, None] * X)
            try:
                cov = mse * np.linalg.inv(XtWX)
                return beta, cov
            except np.linalg.LinAlgError:
                pass
    return beta, None


def compute_sigma_statistic(y: np.ndarray, yhat: np.ndarray, w: np.ndarray, n_params: int, n_obs_total: Optional[int] = None) -> Tuple[float, float, int]:
    """
    Compute sigma using Eq. (3) from Greenstreet et al. (2026):

        sigma^2 = N_obs * sum_i[ p_i (O_i - C_i)^2 ] / ((N_incl - n_par) * sum_j p_j)

    where N_obs is the total number of observations for the object, N_incl is the
    number included in the current fit after clipping, p_i are the weights, and
    n_par is the number of fitted parameters.

    We also return the unnormalized weighted residual sum chi2 = sum w r^2 so the
    existing F-test logic can continue to use it.
    """
    resid = y - yhat
    n_incl = len(y)
    dof = n_incl - n_params
    if dof <= 0:
        return np.inf, np.inf, dof

    chi2 = float(np.sum(w * resid * resid))
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0:
        return np.inf, chi2, dof

    if n_obs_total is None:
        n_obs_total = n_incl

    sigma2 = (float(n_obs_total) * chi2) / (dof * wsum)
    return float(np.sqrt(sigma2)), chi2, dof


def iterative_fit_for_frequency(
    df: pd.DataFrame,
    frequency_cph: float,
    order: int,
    sigma_clip_nsig: float = 3.0,
    max_iter: int = 10,
) -> FrequencyFitResult:
    work = df.copy().reset_index(drop=True)
    work["band"] = work["band"].astype(str)
    band_names = sorted(work["band"].unique().tolist())

    y_all = work["mag_reduced"].to_numpy(dtype=float)
    t_all = work["t_corr_mjd"].to_numpy(dtype=float)
    alpha_all = work["alpha_deg"].to_numpy(dtype=float)
    bands_all = work["band"].to_numpy()
    weights_all = make_weights(work["rmsmag"].to_numpy(dtype=float))

    finite = np.isfinite(y_all) & np.isfinite(t_all) & np.isfinite(alpha_all) & np.isfinite(weights_all)
    kept = finite.copy()

    beta = None
    chi2 = np.inf
    sigma = np.inf
    dof = -1

    for _ in range(max_iter):
        idx = np.where(kept)[0]
        if len(idx) <= (2 + 2 * order + len(band_names)):
            break

        X = build_design_matrix(t_all[idx], alpha_all[idx], bands_all[idx], frequency_cph, order, band_names)
        y = y_all[idx]
        w = weights_all[idx]
        beta, cov = weighted_least_squares(X, y, w)
        yhat = X @ beta
        sigma, chi2, dof = compute_sigma_statistic(y, yhat, w, X.shape[1], n_obs_total=len(work))
        if not np.isfinite(sigma) or sigma <= 0:
            break

        resid = y - yhat
        threshold = sigma_clip_nsig * sigma # / np.sqrt(np.clip(w, 1e-300, None))
        keep_local = np.abs(resid) <= threshold

        new_kept = kept.copy()
        new_kept[idx] = keep_local
        if np.array_equal(new_kept, kept):
            kept = new_kept
            break
        kept = new_kept

    idx = np.where(kept)[0]
    if beta is None or len(idx) <= (2 + 2 * order + len(band_names)):
        n_params = 2 + 2 * order + len(band_names)
        return FrequencyFitResult(
            frequency_cph=float(frequency_cph),
            period_days=float(1.0 / (frequency_cph * HOURS_PER_DAY)),
            period_days_reported=float(1.0 / (frequency_cph * HOURS_PER_DAY)),
            sigma=np.inf,
            chi2=np.inf,
            dof=-1,
            n_obs=int(len(work)),
            n_included=int(len(idx)),
            n_params=n_params,
            order=order,
            coeffs=np.full(n_params, np.nan),
            kept_mask=kept,
            excluded_mask=~kept,
            band_names=band_names,
            n_model_maxima=0,
            doubled_period=False,
        )

    X = build_design_matrix(t_all[idx], alpha_all[idx], bands_all[idx], frequency_cph, order, band_names)
    y = y_all[idx]
    w = weights_all[idx]
    beta, cov = weighted_least_squares(X, y, w)
    yhat = X @ beta
    sigma, chi2, dof = compute_sigma_statistic(y, yhat, w, X.shape[1], n_obs_total=len(work))

    n_maxima = count_model_maxima(beta, frequency_cph, order, band_names, alpha_mean=float(np.nanmean(alpha_all[idx])))
    base_period = 1.0 / (frequency_cph * HOURS_PER_DAY)
    

    # Use this to force the Fourier on one peak period
    #doubled = False
    #reported_period = base_period  

    # To not force it, use:
    doubled = n_maxima == 1
    reported_period = 2.0 * base_period if doubled else base_period


    return FrequencyFitResult(
        frequency_cph=float(frequency_cph),
        period_days=float(base_period),
        period_days_reported=float(reported_period),
        sigma=float(sigma),
        chi2=float(chi2),
        dof=int(dof),
        n_obs=int(len(work)),
        n_included=int(len(idx)),
        n_params=int(X.shape[1]),
        order=int(order),
        coeffs=beta,
        kept_mask=kept,
        excluded_mask=~kept,
        band_names=band_names,
        n_model_maxima=int(n_maxima),
        doubled_period=bool(doubled),
        cov=cov,
    )


def count_model_maxima(
    coeffs: np.ndarray,
    frequency_cph: float,
    order: int,
    band_names: Sequence[str],
    alpha_mean: float,
    n_grid: int = 4000,
) -> int:
    t = np.linspace(0.0, 1.0 / (frequency_cph * HOURS_PER_DAY), n_grid, endpoint=False)
    phase_poly = coeffs[0] * alpha_mean + coeffs[1] * alpha_mean * alpha_mean
    y = np.full_like(t, phase_poly)

    p = 2
    for j in range(1, order + 1):
        y += coeffs[p] * np.cos(2.0 * np.pi * j * frequency_cph * HOURS_PER_DAY * t)
        p += 1
        y += coeffs[p] * np.sin(2.0 * np.pi * j * frequency_cph * HOURS_PER_DAY * t)
        p += 1

    if len(band_names) > 0:
        y += coeffs[p]

    n = len(y)
    c = 0
    for i in range(n):
        ym1 = y[(i - 1) % n]
        y0 = y[i]
        yp1 = y[(i + 1) % n]
        if y0 > ym1 and y0 > yp1:
            c += 1
    return c


def make_frequency_grid(
    time_span_days: float,
    fmax: float,
    fmin: Optional[float] = None,
) -> np.ndarray:
    """Build a frequency grid in cycles/hour.

    The observation times are stored in days, but the public/search frequency
    unit is cycles/hour. The default minimum is still two cycles across the
    observing span, converted to cycles/hour.
    """
    time_span_hours = time_span_days * HOURS_PER_DAY
    if fmin is None:
        fmin = max(2.0 / time_span_hours, 1e-6)
    n_freq = int(max(1000, np.ceil(30.0 * time_span_hours * (fmax - fmin))))
    return np.linspace(fmin, fmax, n_freq)


def ftest_more_complex_better(simple: FrequencyFitResult, complex_: FrequencyFitResult) -> float:
    if not np.isfinite(simple.chi2) or not np.isfinite(complex_.chi2):
        return 0.0
    if simple.dof <= complex_.dof:
        return 0.0
    df_num = simple.n_params - complex_.n_params
    df_den = complex_.dof
    if df_num <= 0 or df_den <= 0:
        return 0.0

    num = (simple.chi2 - complex_.chi2) / df_num
    den = complex_.chi2 / df_den
    if den <= 0 or num < 0:
        return 0.0

    fval = num / den
    return float(f_dist.cdf(fval, df_num, df_den))


def choose_order(best_by_order: Dict[int, FrequencyFitResult]) -> OrderSelectionResult:
    orders = sorted(best_by_order)
    comparisons: Dict[Tuple[int, int], float] = {}

    chosen = orders[-1]
    for k in orders:
        ok = True
        for k2 in orders:
            if k2 <= k:
                continue
            p = ftest_more_complex_better(best_by_order[k], best_by_order[k2])
            comparisons[(k, k2)] = p
            if p >= 0.90:
                ok = False
        if ok:
            chosen = k
            break

    return OrderSelectionResult(
        chosen_order=chosen,
        best_fit=best_by_order[chosen],
        best_by_order=best_by_order,
        comparison_pvalues=comparisons,
    )


def same_order_sigma_threshold(
    fit_best: FrequencyFitResult,
    same_order_results: List[FrequencyFitResult],
    pvalue_target: float = 0.95,
) -> float:
    threshold = fit_best.sigma
    for res in same_order_results:
        if res is fit_best or not np.isfinite(res.chi2):
            continue
        better = fit_best if fit_best.chi2 <= res.chi2 else res
        worse = res if fit_best.chi2 <= res.chi2 else fit_best
        p = ftest_more_complex_better(better, worse)
        if worse is res and p >= pvalue_target:
            threshold = max(threshold, res.sigma)
    return threshold


def search_best_frequencies(df: pd.DataFrame, frequencies: np.ndarray, orders: Sequence[int]) -> Dict[int, List[FrequencyFitResult]]:
    all_results: Dict[int, List[FrequencyFitResult]] = {k: [] for k in orders}
    for order in orders:
        for freq in frequencies:
            all_results[order].append(iterative_fit_for_frequency(df, float(freq), int(order)))
    return all_results


def period_uncertainty_from_possible_solutions(
    best_fit: FrequencyFitResult,
    possible: List[PossibleSolution],
) -> Tuple[float, float, float]:
    periods = np.array([x.period_days_reported for x in possible], dtype=float)
    if len(periods) == 0:
        p = best_fit.period_days_reported
        return p, p, 0.0
    pmin = float(np.min(periods))
    pmax = float(np.max(periods))
    half_range = 0.5 * (pmax - pmin)
    return pmin, pmax, half_range


def compute_lightcurve_amplitude(
    fit: FrequencyFitResult,
    alpha_mean: float,
    n_grid: int = 5000,
) -> float:
    t = np.linspace(0.0, fit.period_days, n_grid, endpoint=False)
    y = np.full_like(t, fit.coeffs[0] * alpha_mean + fit.coeffs[1] * alpha_mean * alpha_mean)
    p = 2
    for j in range(1, fit.order + 1):
        y += fit.coeffs[p] * np.cos(2.0 * np.pi * j * fit.frequency_cph * HOURS_PER_DAY * t)
        p += 1
        y += fit.coeffs[p] * np.sin(2.0 * np.pi * j * fit.frequency_cph * HOURS_PER_DAY * t)
        p += 1
    return float(np.nanmax(y) - np.nanmin(y))


def compute_amplitude_error_from_covariance_hof(
    fit: FrequencyFitResult,
    alpha_mean: float,
    n_grid: int = 5000,
) -> Optional[float]:
    """
    Compute amplitude error using error propagation from the covariance matrix for HOF.
    
    The amplitude depends only on the Fourier coefficients (indices 2 to 2+2*order-1).
    The phase polynomial (indices 0,1) and band offsets (indices 2+2*order onwards) 
    do not affect the rotational amplitude.
    """
    if fit.cov is None or not np.isfinite(fit.cov).all():
        return None
    
    # Extract only the Fourier coefficients (rotational part)
    n_fourier = 2 * fit.order
    fourier_start = 2
    fourier_end = fourier_start + n_fourier
    fourier_coeffs = fit.coeffs[fourier_start:fourier_end]
    fourier_cov = fit.cov[fourier_start:fourier_end, fourier_start:fourier_end]
    
    # Compute amplitude at current coefficients
    t = np.linspace(0.0, fit.period_days, n_grid, endpoint=False)
    y = np.zeros_like(t, dtype=float)
    p = 0
    for j in range(1, fit.order + 1):
        y += fourier_coeffs[p] * np.cos(2.0 * np.pi * j * fit.frequency_cph * HOURS_PER_DAY * t)
        p += 1
        y += fourier_coeffs[p] * np.sin(2.0 * np.pi * j * fit.frequency_cph * HOURS_PER_DAY * t)
        p += 1
    amplitude = np.nanmax(y) - np.nanmin(y)
    
    # Compute gradient of amplitude with respect to Fourier coefficients numerically using central differences
    epsilon = 1e-8
    gradient = np.zeros(n_fourier)
    
    for i in range(n_fourier):
        coeffs_plus = fourier_coeffs.copy()
        coeffs_plus[i] += epsilon
        y_plus = np.zeros_like(t, dtype=float)
        p = 0
        for j in range(1, fit.order + 1):
            y_plus += coeffs_plus[p] * np.cos(2.0 * np.pi * j * fit.frequency_cph * HOURS_PER_DAY * t)
            p += 1
            y_plus += coeffs_plus[p] * np.sin(2.0 * np.pi * j * fit.frequency_cph * HOURS_PER_DAY * t)
            p += 1
        amplitude_plus = np.nanmax(y_plus) - np.nanmin(y_plus)
        
        coeffs_minus = fourier_coeffs.copy()
        coeffs_minus[i] -= epsilon
        y_minus = np.zeros_like(t, dtype=float)
        p = 0
        for j in range(1, fit.order + 1):
            y_minus += coeffs_minus[p] * np.cos(2.0 * np.pi * j * fit.frequency_cph * HOURS_PER_DAY * t)
            p += 1
            y_minus += coeffs_minus[p] * np.sin(2.0 * np.pi * j * fit.frequency_cph * HOURS_PER_DAY * t)
            p += 1
        amplitude_minus = np.nanmax(y_minus) - np.nanmin(y_minus)
        
        gradient[i] = (amplitude_plus - amplitude_minus) / (2 * epsilon)
    
    # Compute variance using error propagation
    try:
        variance = gradient.T @ fourier_cov @ gradient
        if variance > 0:
            return float(np.sqrt(variance))
    except (np.linalg.LinAlgError, ValueError):
        pass
    
    return None


def extract_band_H(coeffs: np.ndarray, order: int, band_names: Sequence[str]) -> Dict[str, float]:
    start = 2 + 2 * order
    return {band: float(coeffs[start + i]) for i, band in enumerate(band_names)}


def plot_sigma_periodogram(
    results: List[FrequencyFitResult],
    out_png: str,
    title: str,
    sigma_threshold: Optional[float] = None,
    best_fit: Optional[FrequencyFitResult] = None,
) -> None:
    """Plot sigma vs frequency in the style of the paper figure."""
    freqs = np.array([r.frequency_cph for r in results], dtype=float)
    sigmas = np.array([r.sigma for r in results], dtype=float)

    finite = np.isfinite(freqs) & np.isfinite(sigmas)
    freqs = freqs[finite]
    sigmas = sigmas[finite]
    order_idx = np.argsort(freqs)
    freqs = freqs[order_idx]
    sigmas = sigmas[order_idx]

    plt.figure(figsize=FIGSIZE)
    plt.plot(freqs, sigmas, lw=0.8, label="Periodogram")

    if sigma_threshold is not None and np.isfinite(sigma_threshold):
        plt.axhline(sigma_threshold, ls="--", lw=0.8, label=r"$\sigma$ threshold")

    if best_fit is not None and np.isfinite(best_fit.frequency_cph) and np.isfinite(best_fit.sigma):
        plt.plot(best_fit.frequency_cph, best_fit.sigma, marker="o", ms=3.0, linestyle="None", label=f"Solution: {best_fit.period_days_reported * 24.0:.4f} h")
    plt.xlim(0, 100.0 / HOURS_PER_DAY) 
    plt.xlabel("Frequency (cycles/hour)")
    plt.ylabel("Sigma")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def extract_phase_poly_coeffs(coeffs: np.ndarray) -> Tuple[float, float]:
    c1 = float(coeffs[0])
    c2 = float(coeffs[1])
    return c1, c2


def evaluate_fourier_only(
    coeffs: np.ndarray,
    order: int,
    frequency_cph: float,
    t: np.ndarray,
) -> np.ndarray:
    y = np.zeros_like(t, dtype=float)
    p = 2
    for j in range(1, order + 1):
        arg = 2.0 * np.pi * j * frequency_cph * HOURS_PER_DAY * t
        y += coeffs[p] * np.cos(arg)
        p += 1
        y += coeffs[p] * np.sin(arg)
        p += 1
    return y

def plot_phased_lightcurve(
    df: pd.DataFrame,
    fit: FrequencyFitResult,
    out_png: str,
    target_name: str,
) -> None:
    kept = fit.kept_mask
    use = df.loc[kept].copy().reset_index(drop=True)

    t = use["t_corr_mjd"].to_numpy(dtype=float)
    y = use["mag_reduced"].to_numpy(dtype=float)
    dy = use["rmsmag"].to_numpy(dtype=float)
    bands = use["band"].astype(str).to_numpy()
    alpha = use["alpha_deg"].to_numpy(dtype=float)

    # Display phase can still use the reported period
    phase = ((t - np.nanmin(t)) / fit.period_days_reported) % 1.0

    # Coefficients
    c1, c2 = extract_phase_poly_coeffs(fit.coeffs)
    band_H = extract_band_H(fit.coeffs, fit.order, fit.band_names) if len(fit.band_names) > 0 else {}

    # Remove BOTH the per-band constant and the per-point phase-angle term
    y_rot = y.copy()
    for i, b in enumerate(bands):
        if b in band_H:
            y_rot[i] -= band_H[b]
        y_rot[i] -= c1 * alpha[i] + c2 * alpha[i] * alpha[i]

    # Evaluate only the rotational part of the model
    phase_grid = np.linspace(0.0, 2.0, 2000)

    t0 = np.nanmin(t)
    phase = ((t - t0) / fit.period_days_reported) % 1.0

    phase_grid = np.linspace(0.0, 2.0, 2000)
    t_grid = t0 + phase_grid * fit.period_days_reported
    yfit_rot = evaluate_fourier_only(
        coeffs=fit.coeffs,
        order=fit.order,
        frequency_cph=fit.frequency_cph,
        t=t_grid,
    )

    yfit_rot = evaluate_fourier_only(
        coeffs=fit.coeffs,
        order=fit.order,
        frequency_cph=fit.frequency_cph,
        t=t_grid,
    )


    plt.figure(figsize=FIGSIZE)

    band_style = {
        "g": {"color": "blue",   "marker": "s"},  # square
        "i": {"color": "orange", "marker": "D"},  # diamond
        "r": {"color": "green",  "marker": "^"},  # triangle up
        "z": {"color": "red",    "marker": "v"},  # triangle down
        "u": {"color": "purple", "marker": "o"},  # round
    }

    for band in sorted(pd.unique(bands)):
        m = bands == band
        style = band_style.get(band, {"color": "gray", "marker": "o"})

        phase_plot = np.concatenate([phase[m], phase[m] + 1.0])
        y_plot = np.concatenate([-y_rot[m], -y_rot[m]])
        dy_plot = np.concatenate([dy[m], dy[m]])
        valid_errors = np.isfinite(dy_plot) & (dy_plot > 0)

        if np.any(valid_errors):
            plt.errorbar(
                phase_plot[valid_errors],
                y_plot[valid_errors],
                yerr=dy_plot[valid_errors],
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

        if np.any(~valid_errors):
            plt.scatter(
                phase_plot[~valid_errors],
                y_plot[~valid_errors],
                s=30,
                label=band if not np.any(valid_errors) else None,
                color=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )

    plt.plot(
        phase_grid,
        -1 * yfit_rot,
        lw=2.5,
        color="black",
        label=f"Fourier order {fit.order}",
    )


    plt.xlabel("Phase")
    plt.ylabel("Reduced magnitude") # $m - 5\log_{10}(r\Delta) - H_{\rm band} - c_1\alpha - c_2\alpha^2$"
    plt.title(f"HOF: {target_name}, phased lightcurve, P = {fit.period_days_reported * 24.0:.4f} h")
    plt.legend()
    plt.xlim(0.0,1.0)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_fit_residuals(df: pd.DataFrame, fit: FrequencyFitResult, out_png: str) -> None:
    kept = fit.kept_mask
    use = df.loc[kept].copy().reset_index(drop=True)

    t = use["t_corr_mjd"].to_numpy(dtype=float)
    y = use["mag_reduced"].to_numpy(dtype=float)
    alpha = use["alpha_deg"].to_numpy(dtype=float)
    bands = use["band"].astype(str).to_numpy()

    X = build_design_matrix(t, alpha, bands, fit.frequency_cph, fit.order, fit.band_names)
    yhat = X @ fit.coeffs
    resid = y - yhat

    phase = ((t - np.nanmin(t)) / fit.period_days_reported) % 1.0

    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()

    for band in sorted(pd.unique(bands)):
        m = bands == band
        color = ax._get_lines.get_next_color()
        plt.scatter(phase[m], resid[m], s=18, label=band, color=color)
        plt.scatter(phase[m] + 1.0, resid[m], s=18, color=color)
        
    plt.axhline(0.0, lw=1.5)
    plt.xlabel("Phase")
    plt.ylabel("Residual (data - full model)")
    plt.title("Phased residuals")
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
    parser.add_argument("--min-frequency-cph", type=float, default=None, help="Minimum frequency in cycles/hour")
    parser.add_argument("--max-frequency-cph", type=float, default=1000.0 / HOURS_PER_DAY, help="Maximum frequency in cycles/hour") # equivalent to 1000/24 cycles/hour
    parser.add_argument("--orders", default="2,3,4,5,6", help="Comma-separated Fourier orders to test")
    parser.add_argument("--out-prefix", default="2026_DO14")
    args = parser.parse_args()

    high_order_dir = Path("HIGH_ORDER_FOURIER")
    figures_dir = Path("FIGURES")
    high_order_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    base_prefix = f"{args.out_prefix}_high_order_Fourier"

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
    eph_csv = high_order_dir / f"{base_prefix}_horizons_range.csv"
    eph_df.to_csv(eph_csv, index=False)
    print(f"Wrote dense Horizons table: {eph_csv.resolve()}")

    merged = interpolate_ephemerides(df, eph_df)
    merged = add_section31_preprocessing(merged)
    merged_csv = high_order_dir / f"{base_prefix}_with_horizons_interp.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"Wrote merged/interpolated table: {merged_csv.resolve()}")

    time_span_days = float(merged["t_corr_mjd"].max() - merged["t_corr_mjd"].min())
    if time_span_days <= 0:
        raise RuntimeError("Observation time span must be positive.")

    frequencies = make_frequency_grid(time_span_days=time_span_days, fmax=args.max_frequency_cph, fmin=args.min_frequency_cph)
    print(f"Frequency grid: {len(frequencies)} samples from {frequencies.min():.6f} to {frequencies.max():.6f} c/h")

    orders = [int(x) for x in args.orders.split(",") if str(x).strip()]
    print(f"Testing Fourier orders: {orders}")

    all_results = search_best_frequencies(merged, frequencies, orders)
    best_by_order = {k: min(v, key=lambda r: r.sigma) for k, v in all_results.items()}

    order_selection = choose_order(best_by_order)
    best_fit = order_selection.best_fit

    print("\nBest fit by Fourier order:")
    for k in sorted(best_by_order):
        r = best_by_order[k]
        print(
            f"  order={k}: P={r.period_days_reported:.8f} d "
            f"({r.period_days_reported * 24.0:.5f} h), sigma={r.sigma:.6f}, "
            f"included={r.n_included}, maxima={r.n_model_maxima}, doubled={r.doubled_period}"
        )

    print(f"\nChosen order: {order_selection.chosen_order}")
    for (k1, k2), p in sorted(order_selection.comparison_pvalues.items()):
        print(f"  F-test p(complex order {k2} better than order {k1}) = {p:.4f}")

    same_order = all_results[best_fit.order]
    sigma_threshold = same_order_sigma_threshold(best_fit, same_order, pvalue_target=0.95)
    possible_fits_raw = [r for r in same_order if np.isfinite(r.sigma) and r.sigma <= sigma_threshold]

    possible = [
        PossibleSolution(
            frequency_cph=r.frequency_cph,
            period_days=r.period_days,
            period_days_reported=r.period_days_reported,
            sigma=r.sigma,
            delta_period_days=abs(r.period_days_reported - best_fit.period_days_reported),
        )
        for r in possible_fits_raw
    ]
    possible = sorted(possible, key=lambda x: (x.delta_period_days, x.sigma))

    pmin, pmax, half_range = period_uncertainty_from_possible_solutions(best_fit, possible)
    reliable_limit = max(2.0 * best_fit.period_days_reported, 7.0 / 24.0)
    is_reliable = half_range <= reliable_limit

    alpha_mean = float(np.nanmean(merged.loc[best_fit.kept_mask, "alpha_deg"]))
    amplitude_mag = compute_lightcurve_amplitude(best_fit, alpha_mean=alpha_mean)
    amplitude_error = compute_amplitude_error_from_covariance_hof(best_fit, alpha_mean=alpha_mean)
    band_H = extract_band_H(best_fit.coeffs, best_fit.order, best_fit.band_names)


    # ------------------------------------------------------------
    # FORMAL PERIOD ERROR: Montgomery & O'Donoghue (1999)
    # ------------------------------------------------------------
    use_hof = merged.loc[best_fit.kept_mask].copy().reset_index(drop=True)

    t_hof = use_hof["t_corr_mjd"].to_numpy(dtype=float)
    y_hof = use_hof["mag_reduced"].to_numpy(dtype=float)
    alpha_hof = use_hof["alpha_deg"].to_numpy(dtype=float)
    bands_hof = use_hof["band"].astype(str).to_numpy()

    X_hof = build_design_matrix(
        t_corr=t_hof,
        alpha_deg=alpha_hof,
        bands=bands_hof,
        frequency_cph=best_fit.frequency_cph,
        order=best_fit.order,
        band_names=best_fit.band_names,
    )

    yhat_hof = X_hof @ best_fit.coeffs
    residuals_hof = y_hof - yhat_hof

    dof_hof = max(1, len(y_hof) - best_fit.n_params)
    residual_rms_hof = float(
        np.sqrt(np.sum(residuals_hof**2) / dof_hof)
    )

    time_span_days_hof = float(np.nanmax(t_hof) - np.nanmin(t_hof))
    semi_amplitude_mag = 0.5 * amplitude_mag

    period_error_montgomery_days = np.nan
    period_error_montgomery_hours = np.nan
    period_error_conservative_days = half_range
    period_error_conservative_hours = half_range * 24.0

    if (
        np.isfinite(semi_amplitude_mag)
        and semi_amplitude_mag > 0
        and np.isfinite(residual_rms_hof)
        and residual_rms_hof >= 0
    ):
        # Use the fitted base period here.
        # If the code doubled the reported period, multiply the error by 2.
        period_error_base_days = compute_period_error(
            period_days=best_fit.period_days,
            amplitude=semi_amplitude_mag,
            n_data_points=len(y_hof),
            time_span_days=time_span_days_hof,
            residual_rms=residual_rms_hof,
        )

        period_multiplier = 2.0 if best_fit.doubled_period else 1.0
        period_error_montgomery_days = period_multiplier * period_error_base_days
        period_error_montgomery_hours = 24.0 * period_error_montgomery_days

        period_error_conservative_days = max(
            half_range,
            period_error_montgomery_days,
        )
        period_error_conservative_hours = 24.0 * period_error_conservative_days




    print(f"\nBest reported period: {best_fit.period_days_reported:.8f} d = {best_fit.period_days_reported * 24.0:.5f} h")
    print(f"Sigma threshold for alternate same-order solutions: {sigma_threshold:.6f}")
    print(f"Possible same-order solutions: {len(possible)}")
    print(f"Period range from thresholded solutions: {pmin:.8f} to {pmax:.8f} d")
    print(f"Half-range uncertainty: {half_range:.8f} d = {half_range * 24.0:.5f} h")
    if np.isfinite(period_error_montgomery_days):
        print(
            f"Formal Montgomery period error: "
            f"{period_error_montgomery_days:.8f} d = "
            f"{period_error_montgomery_hours:.5f} h"
        )
        print(f"Residual RMS used for Montgomery error: {residual_rms_hof:.5f} mag")
        print(
            f"Conservative period error: "
            f"{period_error_conservative_days:.8f} d = "
            f"{period_error_conservative_hours:.5f} h"
        )
    print(f"Reliability cutoff: {reliable_limit:.8f} d = {reliable_limit * 24.0:.5f} h")
    print(f"Reliable by paper-style criterion: {is_reliable}")
    print(f"Lightcurve amplitude: {amplitude_mag:.4f} mag")
    if amplitude_error is not None and np.isfinite(amplitude_error):
        print(f"Lightcurve amplitude error: {amplitude_error:.4f} mag")
    print(f"Per-band H offsets: {band_H}")

    summary_txt = high_order_dir / f"{base_prefix}_best_period.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"target={args.target}\n")
        f.write(f"location={args.location}\n")
        f.write(f"step_minutes={args.step_minutes}\n")
        f.write(f"time_span_days={time_span_days:.10f}\n")
        f.write(f"n_frequency_samples={len(frequencies)}\n")
        f.write(f"chosen_order={best_fit.order}\n")
        f.write(f"best_frequency_cph={best_fit.frequency_cph:.10f}\n")
        f.write(f"best_period_days={best_fit.period_days_reported:.10f}\n")
        f.write(f"best_period_hours={best_fit.period_days_reported * 24.0:.10f}\n")
        f.write(f"base_period_days_before_doubling={best_fit.period_days:.10f}\n")
        f.write(f"sigma={best_fit.sigma:.10f}\n")
        f.write(f"sigma_threshold={sigma_threshold:.10f}\n")
        f.write(f"n_possible_solutions={len(possible)}\n")
        f.write(f"period_min_days={pmin:.10f}\n")
        f.write(f"period_max_days={pmax:.10f}\n")
        f.write(f"period_half_range_days={half_range:.10f}\n")
        f.write(f"period_error_montgomery_days={period_error_montgomery_days:.10f}\n")
        f.write(f"period_error_montgomery_hours={period_error_montgomery_hours:.10f}\n")
        f.write(f"period_error_conservative_days={period_error_conservative_days:.10f}\n")
        f.write(f"period_error_conservative_hours={period_error_conservative_hours:.10f}\n")
        f.write(f"residual_rms_montgomery_mag={residual_rms_hof:.10f}\n")
        f.write(f"semi_amplitude_mag={semi_amplitude_mag:.10f}\n")
        f.write(f"reliability_limit_days={reliable_limit:.10f}\n")
        f.write(f"is_reliable={is_reliable}\n")
        f.write(f"amplitude_mag={amplitude_mag:.10f}\n")
        if amplitude_error is not None and np.isfinite(amplitude_error):
            f.write(f"amplitude_error_mag={amplitude_error:.10f}\n")
        else:
            f.write(f"amplitude_error_mag=NaN\n")
        f.write(f"doubled_period={best_fit.doubled_period}\n")
        f.write(f"n_model_maxima={best_fit.n_model_maxima}\n")
        for band, val in band_H.items():
            f.write(f"H_{band}={val:.10f}\n")
    print(f"Wrote summary: {summary_txt.resolve()}")

    candidates_csv = high_order_dir / f"{base_prefix}_possible_solutions.csv"
    pd.DataFrame([vars(x) for x in possible]).to_csv(candidates_csv, index=False)
    print(f"Wrote candidate solutions: {candidates_csv.resolve()}")

    best_by_order_csv = high_order_dir / f"{base_prefix}_best_by_order.csv"
    pd.DataFrame([
        {
            "order": r.order,
            "frequency_cph": r.frequency_cph,
            "period_days": r.period_days_reported,
            "period_hours": r.period_days_reported * 24.0,
            "sigma": r.sigma,
            "chi2": r.chi2,
            "dof": r.dof,
            "n_included": r.n_included,
            "n_model_maxima": r.n_model_maxima,
            "doubled_period": r.doubled_period,
        }
        for r in [best_by_order[k] for k in sorted(best_by_order)]
    ]).to_csv(best_by_order_csv, index=False)
    print(f"Wrote per-order best fits: {best_by_order_csv.resolve()}")

    sigma_png = figures_dir / f"{base_prefix}_sigma_periodogram_order{best_fit.order}.png"
    phased_png = figures_dir / f"{base_prefix}_phased.png"
    plot_sigma_periodogram(
        same_order,
        sigma_png,
        f"HOF periodogram - order {best_fit.order}: {args.target}",
        sigma_threshold=sigma_threshold,
        best_fit=best_fit,
    )
    plot_phased_lightcurve(merged, best_fit, phased_png, target_name=args.target)
    print(f"Wrote plots: {sigma_png.resolve()}, {phased_png.resolve()}")
    resid_png = figures_dir / f"{base_prefix}_residuals.png"
    plot_fit_residuals(merged, best_fit, resid_png)
    print(f"Wrote plots: {sigma_png.resolve()}, {phased_png.resolve()}, {resid_png.resolve()}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
