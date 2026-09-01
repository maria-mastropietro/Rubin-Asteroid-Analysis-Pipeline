#!/usr/bin/env python3
"""
Run the Multiband Lomb-Scargle and color/taxonomy workflows for an asteroid.

This wrapper assumes the following files are in the same directory:
    - Period_search_Multiband_Lomb_Scargle.py
    - Taxonomy.py
    - this file: Asteroid_LSM.py

Key features:
    - loads sibling scripts relative to __file__
    - registers dynamically imported modules in sys.modules
        before exec_module(), which avoids Python 3.13 dataclass failures
    - runs the LSM workflow only
    - computes color offsets, color differences, and taxonomy using the
        LSM best period
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.time import Time

import time


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



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECTION32_PATH = os.path.join(BASE_DIR, "Period_search_Multiband_Lomb_Scargle.py")
COLORS_TAXONOMY_PATH = os.path.join(BASE_DIR, "Taxonomy.py")


# Exact taxonomy rectangles from TAXONOMY_CARRY notebook cell 23 / 25
# (xmin, xmax, ymin, ymax, edgecolor, label_x, label_y)
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


@dataclass
class MethodResult:
    method: str
    best_period_days: Optional[float]
    best_period_hours: Optional[float]
    period_error_days: Optional[float]
    period_error_hours: Optional[float]
    amplitude_mag: Optional[float]
    amplitude_error_mag: Optional[float]
    summary_path: Optional[str]
    extra: Dict[str, str]
    status: str


@dataclass
class ColorTaxonomyAssessment:
    summary_path: Optional[str]
    colors_csv_path: Optional[str]
    taxonomy_csv_path: Optional[str]
    taxonomy_band_csv_path: Optional[str]
    taxonomy_plot_png_path: Optional[str]
    taxonomy_plot_pdf_path: Optional[str]
    taxonomy_plot_svg_path: Optional[str]
    colors: Dict[str, float]
    color_errors: Dict[str, Optional[float]]
    offsets: Dict[str, float]
    taxonomy_info: Dict[str, object]
    mean_alpha_deg: Optional[float]
    mean_alpha_error_deg: Optional[float]
    status: str


@dataclass
class Table2LikeSummary:
    designation: str
    h_mag: Optional[float]
    mean_mag_r: Optional[float]
    mean_mag_r_std: Optional[float]
    number_observations: int
    observation_date_range: Optional[str]
    arc_days: Optional[float]
    period_lsm_hours: Optional[float]
    period_lsm_error_hours: Optional[float]
    period_fourier_hours: Optional[float]
    period_fourier_error_hours: Optional[float]
    amplitude_lsm_mag: Optional[float]
    amplitude_error_lsm_mag: Optional[float]
    amplitude_fourier_mag: Optional[float]
    amplitude_error_fourier_mag: Optional[float]
    axial_elongation: Optional[float]
    axial_elongation_error: Optional[float]
    mean_alpha_deg: Optional[float]
    mean_alpha_error_deg: Optional[float]
    color_gr_lsm: Optional[float]
    color_gr_fourier: Optional[float]
    color_gr_lsm_error: Optional[float]
    color_gr_fourier_error: Optional[float]
    color_gi_lsm: Optional[float]
    color_gi_fourier: Optional[float]
    color_gi_lsm_error: Optional[float]
    color_gi_fourier_error: Optional[float]
    color_ri_lsm: Optional[float]
    color_ri_fourier: Optional[float]
    color_ri_lsm_error: Optional[float]
    color_ri_fourier_error: Optional[float]
    color_iz_lsm: Optional[float]
    color_iz_fourier: Optional[float]
    color_iz_lsm_error: Optional[float]
    color_iz_fourier_error: Optional[float]
    gri_slope_lsm: Optional[float]
    gri_slope_lsm_error: Optional[float]
    gri_slope_fourier: Optional[float]
    gri_slope_fourier_error: Optional[float]
    taxonomy_lsm: Optional[str]
    taxonomy_fourier: Optional[str]


def file_must_exist(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} was not found:\n"
            f"  {path}\n"
            f"Make sure the file exists in the same directory as this wrapper,\n"
            f"or edit the *_PATH constants near the top of the script."
        )


def load_module(name: str, path: str):
    """
    Dynamically import a module from a file path.

    Important:
    Python 3.13 dataclasses can fail unless the module is registered in
    sys.modules before exec_module() runs.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_key_value_summary(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def get_jpl_sbdb_h(target: str) -> Optional[float]:
    target_query = target.strip()
    if not target_query:
        return None

    base_url = "https://ssd-api.jpl.nasa.gov/sbdb.api"
    params = {
        "des": target_query,
        "phys-par": "1",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "python-urllib"}
    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    # Common returned structures
    candidates = []
    if isinstance(data, dict):
        obj = data.get("object")
        if isinstance(obj, dict):
            candidates.append(obj.get("h"))
            candidates.append(obj.get("H"))
        phys = data.get("phys_par")
        if isinstance(phys, dict):
            candidates.append(phys.get("H"))
            candidates.append(phys.get("H_g"))
            candidates.append(phys.get("H_v"))
            candidates.append(phys.get("h"))
        elif isinstance(phys, list):
            for item in phys:
                if isinstance(item, dict):
                    candidates.append(item.get("value"))
                    candidates.append(item.get("H"))
                    candidates.append(item.get("h"))
                    candidates.append(item.get("name"))

    for candidate in candidates:
        h = try_float(candidate)
        if h is not None:
            return h

    return None


def try_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def format_float(x: Optional[float], fmt: str = ".8f") -> str:
    if x is None:
        return "nan"
    return format(x, fmt)


def format_observation_date_range(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None

    if "obs_time" in df.columns:
        obs_time = pd.to_datetime(df["obs_time"], utc=True, errors="coerce")
        obs_time = obs_time.dropna()
        if len(obs_time) > 0:
            start = obs_time.min().strftime("%d-%m-%Y")
            end = obs_time.max().strftime("%d-%m-%Y")
            return f"{start} to {end}"

    if "mjd" in df.columns:
        mjd = pd.to_numeric(df["mjd"], errors="coerce")
        mjd = mjd[np.isfinite(mjd)]
        if len(mjd) > 0:
            start = Time(float(np.nanmin(mjd)), format="mjd", scale="utc").to_datetime()
            end = Time(float(np.nanmax(mjd)), format="mjd", scale="utc").to_datetime()
            return f"{start.strftime('%d-%m-%Y')} to {end.strftime('%d-%m-%Y')}"

    return None


def classify_from_notebook_boxes(gri_slope: float, iz: float) -> str:
    """
    Reproduces the notebook-style rectangular classification.
    Later boxes overwrite earlier ones, matching the notebook logic.
    """
    tax = "nan"

    ordered = [
        ("C", (-5.0,  6.0, -0.200,  0.185)),
        ("B", (-5.0,  0.0, -0.200,  0.000)),
        ("X", ( 2.5,  9.0, -0.005,  0.185)),
        ("D", ( 6.0, 25.0,  0.085,  0.335)),
        ("L", ( 9.0, 25.0, -0.005,  0.085)),
        ("S", ( 6.0, 25.0, -0.265, -0.005)),
        ("Q", ( 5.0,  9.5, -0.265, -0.165)),
        ("A", (21.5, 28.0, -0.265, -0.115)),
        ("V", ( 5.0, 25.0, -0.665, -0.265)),
    ]

    for name, (xmin, xmax, ymin, ymax) in ordered:
        if xmin <= gri_slope <= xmax and ymin <= iz <= ymax:
            tax = name

    return tax


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
    gri_slope: float,
    iz: float,
    label: str,
    taxonomy_from_file: Optional[str],
    out_prefix: str,
    xlim: Tuple[float, float] = (-10.0, 30.0),
    ylim: Tuple[float, float] = (-0.8, 0.4),
    secondary_gri_slope: Optional[float] = None,
    secondary_iz: Optional[float] = None,
    secondary_label: Optional[str] = None,
    secondary_taxonomy_from_file: Optional[str] = None,
    gri_slope_error: Optional[float] = None,
    iz_error: Optional[float] = None,
    secondary_gri_slope_error: Optional[float] = None,
    secondary_iz_error: Optional[float] = None,
) -> str:
    taxonomy_from_boxes = classify_from_notebook_boxes(gri_slope, iz)

    secondary_has_point = (
        secondary_gri_slope is not None
        and secondary_iz is not None
        and np.isfinite(float(secondary_gri_slope))
        and np.isfinite(float(secondary_iz))
    )
    secondary_taxonomy_from_boxes = None
    if secondary_has_point:
        secondary_taxonomy_from_boxes = classify_from_notebook_boxes(
            float(secondary_gri_slope),
            float(secondary_iz),
        )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_taxonomy_boxes(ax)

    ax.scatter(
        [gri_slope],
        [iz],
        s=220,
        marker="*",
        facecolors="tab:blue",
        edgecolors="black",
        linewidths=0.9,
        zorder=5,
        label=f"{label} (LSM)",
    )

    # Add error rectangle for primary point
    if gri_slope_error is not None and iz_error is not None:
        if np.isfinite(gri_slope_error) and np.isfinite(iz_error):
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                xy=(gri_slope - gri_slope_error, iz - iz_error),
                width=2 * gri_slope_error,
                height=2 * iz_error,
                edgecolor="tab:blue",
                facecolor="none",
                linewidth=1.5,
                alpha=0.7,
                zorder=4,
            )
            ax.add_patch(rect)

    ax.annotate(
        label,
        xy=(gri_slope, iz),
        xytext=(8, 8),
        textcoords="offset points",
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
    )

    if secondary_has_point:
        secondary_label = secondary_label or "HOF"
        ax.scatter(
            [float(secondary_gri_slope)],
            [float(secondary_iz)],
            s=220,
            marker="*",
            facecolors="yellow",
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
            label=f"{secondary_label} (HOF)",
        )

        # Add error rectangle for secondary point
        if secondary_gri_slope_error is not None and secondary_iz_error is not None:
            if np.isfinite(secondary_gri_slope_error) and np.isfinite(secondary_iz_error):
                from matplotlib.patches import Rectangle
                rect_secondary = Rectangle(
                    xy=(float(secondary_gri_slope) - secondary_gri_slope_error, float(secondary_iz) - secondary_iz_error),
                    width=2 * secondary_gri_slope_error,
                    height=2 * secondary_iz_error,
                    edgecolor="yellow",
                    facecolor="none",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=4,
                )
                ax.add_patch(rect_secondary)

        ax.annotate(
            secondary_label,
            xy=(float(secondary_gri_slope), float(secondary_iz)),
            xytext=(8, -10),
            textcoords="offset points",
            fontweight="bold",
            ha="left",
            va="top",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
        )

    title_tax = taxonomy_from_file if taxonomy_from_file else taxonomy_from_boxes
    if secondary_has_point and secondary_taxonomy_from_file:
        ax.set_title(
            f"Asteroid taxonomic classification: {label}"
        )
    else:
        ax.set_title(f"Asteroid taxonomic classification: {label}")
    ax.set_xlabel("gri slope [%/100nm]")
    ax.set_ylabel("i-z [mag]")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)

    subtitle = (
        f"LSM: gri_slope = {gri_slope:.4f}, i-z = {iz:.4f} | "
        f"Taxonomy = {taxonomy_from_boxes}"
    )

    if secondary_has_point:
        subtitle += (
            f"\nHOF: gri_slope = {float(secondary_gri_slope):.4f}, "
            f"i-z = {float(secondary_iz):.4f} | "
            f"Taxonomy = {secondary_taxonomy_from_boxes}"
        )

    fig.text(0.02, 0.02, subtitle)

    ax.legend(loc="lower left", framealpha=0.9)

    fig.tight_layout(rect=(0, 0.08 if secondary_has_point else 0.04, 1, 1))

    png_path = f"{out_prefix}.png"
    pdf_path = f"{out_prefix}.pdf"
    svg_path = f"{out_prefix}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches=None, facecolor="white")
    fig.savefig(pdf_path, bbox_inches=None, facecolor="white")
    fig.savefig(svg_path, bbox_inches=None, facecolor="white")

    plt.close(fig)

    return png_path


def extract_first_float_from_text(text: str, patterns: List[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def normalize_band_label(x: object) -> str:
    if pd.isna(x):
        return str(x)
    x = str(x).strip()
    mapping = {
        "Lu": "u", "Lg": "g", "Lr": "r", "Li": "i", "Lz": "z", "Ly": "y",
        "u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "y": "y",
    }
    return mapping.get(x, x)


def load_band_counts(csv_path: str) -> Dict[str, int]:
    df = pd.read_csv(csv_path)

    if "band" not in df.columns:
        return {}

    df["band"] = df["band"].map(normalize_band_label)
    df = df.dropna(subset=["band"]).copy()

    counts = df["band"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def count_bands_meeting_threshold(
    counts: Dict[str, int],
    min_obs: int,
    allowed: Tuple[str, ...] = ("g", "r", "i"),
) -> int:
    return sum(1 for band in allowed if counts.get(band, 0) >= min_obs)


def relative_period_difference(
    p1_days: Optional[float],
    p2_days: Optional[float],
) -> Optional[float]:
    if p1_days is None or p2_days is None:
        return None
    ref = 0.5 * (abs(p1_days) + abs(p2_days))
    if ref <= 0:
        return None
    return abs(p1_days - p2_days) / ref


def find_section31_summary(out_prefix: str) -> Optional[str]:
    high_order_dir = Path("HIGH_ORDER_FOURIER")
    base = f"{out_prefix}_high_order_Fourier"
    candidates = [
        str(high_order_dir / f"{base}_best_period.txt"),
        str(high_order_dir / f"{base}_summary.txt"),
        str(high_order_dir / f"{base}_results.txt"),
        f"{out_prefix}_best_period.txt",
        f"{out_prefix}_summary.txt",
        f"{out_prefix}_section31_best_period.txt",
        f"{out_prefix}_section31_summary.txt",
        f"{out_prefix}_results.txt",
    ]
    return find_first_existing(candidates)


def parse_section31_outputs(out_prefix: str, stdout: str, stderr: str) -> MethodResult:
    summary_path = find_section31_summary(out_prefix)
    info = parse_key_value_summary(summary_path) if summary_path else {}

    high_order_dir = Path("HIGH_ORDER_FOURIER")
    figures_dir = Path("FIGURES")
    base = f"{out_prefix}_high_order_Fourier"

    horizons_csv = find_first_existing([
        str(high_order_dir / f"{base}_horizons_range.csv"),
        f"{out_prefix}_horizons_range.csv",
        f"{out_prefix}_section31_horizons_range.csv",
    ])
    merged_csv = find_first_existing([
        str(high_order_dir / f"{base}_with_horizons_interp.csv"),
        f"{out_prefix}_with_horizons_interp.csv",
        f"{out_prefix}_section31_with_horizons_interp.csv",
    ])
    possible_csv = find_first_existing([
        str(high_order_dir / f"{base}_possible_solutions.csv"),
        f"{out_prefix}_possible_solutions.csv",
        f"{out_prefix}_section31_possible_solutions.csv",
    ])
    best_by_order_csv = find_first_existing([
        str(high_order_dir / f"{base}_best_by_order.csv"),
        f"{out_prefix}_best_by_order.csv",
        f"{out_prefix}_section31_best_by_order.csv",
    ])
    sigma_png = find_first_existing([
        str(figures_dir / f"{base}_sigma_periodogram_order{info.get('chosen_order', '')}.png") if info.get('chosen_order') else "",
        *[str(figures_dir / f"{base}_sigma_periodogram_order{k}.png") for k in range(1, 11)],
        f"{out_prefix}_sigma_periodogram_order{info.get('chosen_order', '')}.png" if info.get('chosen_order') else "",
    ])
    phased_png = find_first_existing([
        str(figures_dir / f"{base}_phased.png"),
        f"{out_prefix}_phased.png",
        f"{out_prefix}_section31_phased.png",
    ])
    residuals_png = find_first_existing([
        str(figures_dir / f"{base}_residuals.png"),
        f"{out_prefix}_residuals.png",
        f"{out_prefix}_section31_residuals.png",
    ])

  # Use base_period_days_before_doubling to force the Fourier on one peak period
  # To not force it, use best_period_days
    best_days = (
        try_float(info.get("best_period_days")) # base_period_days_before_doubling
        or try_float(info.get("best_period_days"))
        or try_float(info.get("period_days"))
        or try_float(info.get("preferred_period_days"))
    )
    best_hours = (
        try_float(info.get("best_period_hours"))
        or try_float(info.get("period_hours"))
        or try_float(info.get("preferred_period_hours"))
    )
    amplitude_mag = try_float(info.get("amplitude_mag"))
    amplitude_error_mag = try_float(info.get("amplitude_error_mag"))
    period_error_days = (
        try_float(info.get("period_error_montgomery_days"))
        or try_float(info.get("period_error_conservative_days"))
    )
    period_error_hours = (
        try_float(info.get("period_error_montgomery_hours"))
        or try_float(info.get("period_error_conservative_hours"))
    )

    if best_days is None:
        best_days = extract_first_float_from_text(
            stdout + "\n" + stderr,
            [
                r"best[_ ]period[_ ]days\s*[:=]\s*([0-9eE.+-]+)",
                r"preferred[_ ]period[_ ]days\s*[:=]\s*([0-9eE.+-]+)",
                r"best fit period\s*[:=]\s*([0-9eE.+-]+)\s*d",
            ],
        )

    if best_hours is None and best_days is not None:
        best_hours = best_days * 24.0

    if period_error_hours is None and period_error_days is not None:
        period_error_hours = period_error_days * 24.0

    if best_hours is None:
        best_hours = extract_first_float_from_text(
            stdout + "\n" + stderr,
            [
                r"best[_ ]period[_ ]hours\s*[:=]\s*([0-9eE.+-]+)",
                r"preferred[_ ]period[_ ]hours\s*[:=]\s*([0-9eE.+-]+)",
                r"best fit period\s*[:=]\s*([0-9eE.+-]+)\s*h",
            ],
        )

    status = "ok" if (best_days is not None or best_hours is not None) else "parse_failed"

    if horizons_csv:
        info["horizons_csv_path"] = horizons_csv
    if merged_csv:
        info["merged_csv_path"] = merged_csv
    if possible_csv:
        info["possible_solutions_csv_path"] = possible_csv
    if best_by_order_csv:
        info["best_by_order_csv_path"] = best_by_order_csv
    if sigma_png:
        info["sigma_periodogram_png_path"] = sigma_png
    if phased_png:
        info["phased_png_path"] = phased_png
    if residuals_png:
        info["residuals_png_path"] = residuals_png

    return MethodResult(
        method="High-Order Fourier",
        best_period_days=best_days,
        best_period_hours=best_hours,
        period_error_days=period_error_days,
        period_error_hours=period_error_hours,
        amplitude_mag=amplitude_mag,
        amplitude_error_mag=amplitude_error_mag,
        summary_path=summary_path,
        extra=info,
        status=status,
    )


def parse_section32_outputs(out_prefix: str, stdout: str, stderr: str) -> MethodResult:
    mls_dir = Path("MULTIBAND_LOMB_SCARGLE")
    figures_dir = Path("FIGURES")
    base = f"{out_prefix}_Multiband_Lomb_Scargle"

    summary_path = find_first_existing(
        [
            str(mls_dir / f"{base}_best_period.txt"),
            str(mls_dir / f"{base}_summary.txt"),
            f"{out_prefix}_best_period.txt",
            f"{out_prefix}_summary.txt",
            f"{out_prefix}_section32_best_period.txt",
            f"{out_prefix}_section32_summary.txt",
        ]
    )
    info = parse_key_value_summary(summary_path) if summary_path else {}

    horizons_csv = find_first_existing([
        str(mls_dir / f"{base}_horizons_range.csv"),
        f"{out_prefix}_horizons_range.csv",
        f"{out_prefix}_section32_horizons_range.csv",
    ])
    merged_csv = find_first_existing([
        str(mls_dir / f"{base}_with_horizons_interp.csv"),
        f"{out_prefix}_with_horizons_interp.csv",
        f"{out_prefix}_section32_with_horizons_interp.csv",
    ])
    periodogram_png = find_first_existing([
        str(figures_dir / f"{base}_periodogram.png"),
        f"{out_prefix}_periodogram.png",
        f"{out_prefix}_section32_periodogram.png",
    ])
    phased_png = find_first_existing([
        str(figures_dir / f"{base}_phased.png"),
        f"{out_prefix}_phased.png",
        f"{out_prefix}_section32_phased.png",
    ])

    best_days = try_float(info.get("best_period_days"))
    best_hours = try_float(info.get("best_period_hours"))
    amplitude_mag = try_float(info.get("amplitude_mag"))
    amplitude_error_mag = try_float(info.get("amplitude_error_mag"))
    period_error_days = try_float(info.get("period_error_montgomery_days"))
    period_error_hours = try_float(info.get("period_error_montgomery_hours"))

    if best_days is None:
        best_days = extract_first_float_from_text(
            stdout + "\n" + stderr,
            [
                r"best[_ ]period[_ ]days\s*[:=]\s*([0-9eE.+-]+)",
                r"best period\s*:\s*([0-9eE.+-]+)\s*d",
            ],
        )
    if best_hours is None:
        best_hours = extract_first_float_from_text(
            stdout + "\n" + stderr,
            [
                r"best[_ ]period[_ ]hours\s*[:=]\s*([0-9eE.+-]+)",
                r"best period\s*:\s*[0-9eE.+-]+\s*d\s*=\s*([0-9eE.+-]+)\s*h",
            ],
        )
    if best_hours is None and best_days is not None:
        best_hours = best_days * 24.0
    if period_error_hours is None and period_error_days is not None:
        period_error_hours = period_error_days * 24.0

    status = "ok" if (best_days is not None or best_hours is not None) else "parse_failed"

    if horizons_csv:
        info["horizons_csv_path"] = horizons_csv
    if merged_csv:
        info["merged_csv_path"] = merged_csv
    if periodogram_png:
        info["periodogram_png_path"] = periodogram_png
    if phased_png:
        info["phased_png_path"] = phased_png

    return MethodResult(
        method="Multiband Lomb-Scargle",
        best_period_days=best_days,
        best_period_hours=best_hours,
        period_error_days=period_error_days,
        period_error_hours=period_error_hours,
        amplitude_mag=amplitude_mag,
        amplitude_error_mag=amplitude_error_mag,
        summary_path=summary_path,
        extra=info,
        status=status,
    )


def run_subprocess(script_path: str, argv: List[str]) -> Tuple[int, str, str]:
    cmd = [sys.executable, script_path] + argv
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_section32(args: argparse.Namespace) -> MethodResult:
    out_prefix = args.out_prefix
    argv = [
        "--csv", args.csv,
        "--target", args.target,
        "--location", args.location,
        "--step-minutes", str(args.step_minutes),
        "--pad-minutes", str(args.pad_minutes),
        "--min-period-days", str(args.min_period_days),
        "--max-period-days", str(args.max_period_days),
        "--oversample", str(args.oversample),
        "--out-prefix", out_prefix,
    ]
    if args.id_type:
        argv.extend(["--id-type", args.id_type])

    code, stdout, stderr = run_subprocess(SECTION32_PATH, argv)
    result = parse_section32_outputs(out_prefix, stdout, stderr)
    if code != 0 and result.status == "ok":
        result.status = f"script_error_{code}"
    elif code != 0:
        result.status = f"failed_{code}"

    result.extra["returncode"] = str(code)
    if stdout:
        result.extra["stdout"] = stdout.strip()
    if stderr:
        result.extra["stderr"] = stderr.strip()
    return result


def run_section31(args: argparse.Namespace) -> MethodResult:
    out_prefix = args.out_prefix
    argv = [
        "--csv", args.csv,
        "--target", args.target,
        "--location", args.location,
        "--step-minutes", str(args.step_minutes),
        "--pad-minutes", str(args.pad_minutes),
        "--out-prefix", out_prefix,
    ]
    if args.id_type:
        argv.extend(["--id-type", args.id_type])

    code, stdout, stderr = run_subprocess(SECTION31_PATH, argv)
    result = parse_section31_outputs(out_prefix, stdout, stderr)
    if code != 0 and result.status == "ok":
        result.status = f"script_error_{code}"
    elif code != 0:
        result.status = f"failed_{code}"

    result.extra["returncode"] = str(code)
    if stdout:
        result.extra["stdout"] = stdout.strip()
    if stderr:
        result.extra["stderr"] = stderr.strip()
    return result


def assess_lsm_success(
    args: argparse.Namespace,
    r32: MethodResult,
) -> bool:
    return r32.status == "ok" and r32.best_period_days is not None


def run_colors_taxonomy_for_period(
    args: argparse.Namespace,
    period_days: Optional[float],
    output_stem: str,
    module_name: str = "taxonomy_mod",
    make_taxonomy_plot: bool = True,
) -> ColorTaxonomyAssessment:
    if period_days is None:
        return ColorTaxonomyAssessment(
            summary_path=None,
            colors_csv_path=None,
            taxonomy_csv_path=None,
            taxonomy_band_csv_path=None,
            taxonomy_plot_png_path=None,
            taxonomy_plot_pdf_path=None,
            taxonomy_plot_svg_path=None,
            colors={},
            color_errors={},
            offsets={},
            taxonomy_info={},
            mean_alpha_deg=None,
            mean_alpha_error_deg=None,
            status="no_period",
        )

    mod = load_module(module_name, COLORS_TAXONOMY_PATH)

    df = mod.load_and_prepare_csv(args.csv)
    eph_df = mod.query_horizons_range(
        target_id=args.target,
        mjd_min=float(df["mjd"].min()),
        mjd_max=float(df["mjd"].max()),
        location=args.location,
        id_type=args.id_type,
        step_minutes=args.step_minutes,
        pad_minutes=args.pad_minutes,
    )
    merged = mod.interpolate_ephemerides(df, eph_df)
    merged = mod.build_corrected_lightcurve(merged)

    color_fit = mod.fit_shared_multiband_fourier_with_offsets(
        merged,
        period_days=period_days,
        order=2,
        reference_band="r",
    )
    taxonomy_info = mod.build_taxonomy_from_color_fit(color_fit)
    taxonomy_band_summary = mod.build_taxonomy_band_summary(
        merged,
        color_fit,
        reference_band="r",
    )
    alpha_inliers = pd.to_numeric(merged.loc[merged["is_inlier"], "alpha_deg"], errors="coerce").dropna()
    mean_alpha_deg = try_float(np.nanmean(alpha_inliers)) if len(alpha_inliers) else None
    mean_alpha_error_deg = (
        try_float(np.nanstd(alpha_inliers, ddof=1))
        if len(alpha_inliers) > 1
        else None
    )

    taxonomy_dir = Path("TAXONOMY")
    figures_dir = Path("FIGURES")
    taxonomy_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    summary_txt = str(taxonomy_dir / f"{output_stem}_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"target={args.target}\n")
        f.write(f"best_period_days={period_days:.10f}\n")
        f.write(f"best_period_hours={period_days * 24.0:.10f}\n")
        f.write(f"color_fit_chi2={color_fit.chi2:.10f}\n")
        f.write(f"color_fit_dof={color_fit.dof}\n")
        f.write(f"color_fit_rms_resid={color_fit.rms_resid:.10f}\n")
        f.write(f"mean_alpha_deg={format_float(mean_alpha_deg, '.10f')}\n")
        f.write(f"mean_alpha_error_deg={format_float(mean_alpha_error_deg, '.10f')}\n")
        
        for band in ["u", "g", "r", "i", "z", "y"]:
            val = try_float(color_fit.band_offsets_relative_to_r.get(band))
            f.write(f"offset_{band}_minus_r={format_float(val, '.10f')}\n")

        for name in ["g-r", "g-i", "r-i", "i-z"]:
            val = try_float(color_fit.colors.get(name))
            f.write(f"color_{name.replace('-', '_')}={format_float(val, '.10f')}\n")
        
        taxonomy = taxonomy_info.get("taxonomy", "nan")
        f.write(f"taxonomy={taxonomy}\n")
        for key in ["g_mag", "r_mag", "i_mag", "z_mag", "gr", "ri", "iz", "gri_slope"]:
            val = try_float(taxonomy_info.get(key))
            f.write(f"{key}={format_float(val, '.10f')}\n")

    color_row: Dict[str, object] = {
        "target": args.target,
        "best_period_days": period_days,
        "best_period_hours": period_days * 24.0,
        "color_fit_chi2": color_fit.chi2,
        "color_fit_dof": color_fit.dof,
        "color_fit_rms_resid": color_fit.rms_resid,
        "mean_alpha_deg": mean_alpha_deg,
        "mean_alpha_error_deg": mean_alpha_error_deg,
    }
    for band, val in color_fit.band_offsets_relative_to_r.items():
        color_row[f"offset_{band}_minus_r"] = val
    for name, val in color_fit.colors.items():
        color_row[f"color_{name}"] = val
    color_row.update(taxonomy_info)

    colors_csv = str(taxonomy_dir / f"{output_stem}_colors.csv")
    pd.DataFrame([color_row]).to_csv(colors_csv, index=False)

    taxonomy_csv = str(taxonomy_dir / f"{output_stem}_taxonomy.csv")
    pd.DataFrame([{
        "target": args.target,
        "best_period_days": period_days,
        "best_period_hours": period_days * 24.0,
        **taxonomy_info,
    }]).to_csv(taxonomy_csv, index=False)

    taxonomy_band_csv = str(taxonomy_dir / f"{output_stem}_taxonomy_band_summary.csv")
    taxonomy_band_summary.to_csv(taxonomy_band_csv, index=False)

    taxonomy_plot_prefix = str(figures_dir / f"{output_stem}_taxonomy_point")
    gri_slope = try_float(taxonomy_info.get("gri_slope"))
    iz = try_float(taxonomy_info.get("iz"))
    taxonomy_plot_png = None
    taxonomy_plot_pdf = None
    taxonomy_plot_svg = None
    if make_taxonomy_plot and gri_slope is not None and iz is not None:
        taxonomy_plot_png = plot_taxonomy_point(
            gri_slope=float(gri_slope),
            iz=float(iz),
            label=args.target,
            taxonomy_from_file=(str(taxonomy_info.get("taxonomy")) if taxonomy_info.get("taxonomy") is not None else None),
            out_prefix=taxonomy_plot_prefix,
            gri_slope_error=taxonomy_info.get("gri_slope_error"),
            iz_error=color_fit.color_errors.get("i-z") if color_fit.color_errors else None,
        )

    return ColorTaxonomyAssessment(
        summary_path=summary_txt,
        colors_csv_path=colors_csv,
        taxonomy_csv_path=taxonomy_csv,
        taxonomy_band_csv_path=taxonomy_band_csv,
        taxonomy_plot_png_path=taxonomy_plot_png,
        taxonomy_plot_pdf_path=taxonomy_plot_pdf,
        taxonomy_plot_svg_path=taxonomy_plot_svg,
        colors=dict(color_fit.colors),
        color_errors=dict(color_fit.color_errors) if color_fit.color_errors is not None else {},
        offsets=dict(color_fit.band_offsets_relative_to_r),
        taxonomy_info=dict(taxonomy_info),
        mean_alpha_deg=mean_alpha_deg,
        mean_alpha_error_deg=mean_alpha_error_deg,
        status="ok",
    )


def run_colors_taxonomy(
    args: argparse.Namespace,
    r32: MethodResult,
) -> ColorTaxonomyAssessment:
    status = "no_section32_period" if r32.best_period_days is None else None
    result = run_colors_taxonomy_for_period(
        args=args,
        period_days=r32.best_period_days,
        output_stem=args.out_prefix,
        module_name="taxonomy_mod_lsm",
    )
    if status is not None:
        result.status = status
    return result


def run_colors_taxonomy_hof(
    args: argparse.Namespace,
    r31: MethodResult,
) -> ColorTaxonomyAssessment:
    status = "no_section31_period" if r31.best_period_days is None else None
    result = run_colors_taxonomy_for_period(
        args=args,
        period_days=r31.best_period_days,
        output_stem=f"{args.out_prefix}_HOF",
        module_name="taxonomy_mod_hof",
        make_taxonomy_plot=False,
    )
    if status is not None:
        result.status = status

    result.taxonomy_plot_png_path = None
    result.taxonomy_plot_pdf_path = None
    result.taxonomy_plot_svg_path = None
    return result

def overlay_taxonomy_plots_with_hof(
    args: argparse.Namespace,
    color_tax: ColorTaxonomyAssessment,
    color_tax_hof: ColorTaxonomyAssessment,
) -> ColorTaxonomyAssessment:
    if color_tax.status != "ok":
        return color_tax

    primary_gri = try_float(color_tax.taxonomy_info.get("gri_slope"))
    primary_iz = try_float(color_tax.taxonomy_info.get("iz"))
    if primary_gri is None or primary_iz is None:
        return color_tax

    secondary_gri = try_float(color_tax_hof.taxonomy_info.get("gri_slope"))
    secondary_iz = try_float(color_tax_hof.taxonomy_info.get("iz"))
    secondary_taxonomy = color_tax_hof.taxonomy_info.get("taxonomy") if color_tax_hof.status == "ok" else None

    figures_dir = Path("FIGURES")
    figures_dir.mkdir(exist_ok=True)
    taxonomy_plot_prefix = str(figures_dir / f"{args.out_prefix}_taxonomy_point")

    taxonomy_plot_png = plot_taxonomy_point(
        gri_slope=float(primary_gri),
        iz=float(primary_iz),
        label=str(args.target),
        taxonomy_from_file=(str(color_tax.taxonomy_info.get("taxonomy")) if color_tax.taxonomy_info.get("taxonomy") is not None else None),
        out_prefix=taxonomy_plot_prefix,
        secondary_gri_slope=(float(secondary_gri) if secondary_gri is not None else None),
        secondary_iz=(float(secondary_iz) if secondary_iz is not None else None),
        secondary_label=(str(args.target) if secondary_gri is not None and secondary_iz is not None else None),
        secondary_taxonomy_from_file=(str(secondary_taxonomy) if secondary_taxonomy is not None else None),
        gri_slope_error=color_tax.taxonomy_info.get("gri_slope_error"),
        iz_error=color_tax.color_errors.get("i-z") if color_tax.color_errors else None,
        secondary_gri_slope_error=color_tax_hof.taxonomy_info.get("gri_slope_error") if color_tax_hof.status == "ok" else None,
        secondary_iz_error=color_tax_hof.color_errors.get("i-z") if color_tax_hof.color_errors else None,
    )

    color_tax.taxonomy_plot_png_path = taxonomy_plot_png
    color_tax.taxonomy_plot_pdf_path = None
    color_tax.taxonomy_plot_svg_path = None
    return color_tax


def write_comparison_csv(
    path: str,
    rows: List[MethodResult],
    color_tax: ColorTaxonomyAssessment,
) -> None:
    fieldnames = [
        "method",
        "status",
        "best_period_days",
        "best_period_hours",
        "period_error_days",
        "period_error_hours",
        "summary_path",
        "horizons_csv_path",
        "merged_csv_path",
        "possible_solutions_csv_path",
        "best_by_order_csv_path",
        "sigma_periodogram_png_path",
        "periodogram_png_path",
        "phased_png_path",
        "residuals_png_path",
        "lsm_ok",
        "color_taxonomy_status",
        "color_g-r",
        "color_g-i",
        "color_r-i",
        "color_i-z",
        "taxonomy",
        "gri_slope",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {
                "method": r.method,
                "status": r.status,
                "best_period_days": "" if r.best_period_days is None else f"{r.best_period_days:.12f}",
                "best_period_hours": "" if r.best_period_hours is None else f"{r.best_period_hours:.12f}",
                "period_error_days": "" if r.period_error_days is None else f"{r.period_error_days:.12f}",
                "period_error_hours": "" if r.period_error_hours is None else f"{r.period_error_hours:.12f}",
                "summary_path": r.summary_path or "",
                "horizons_csv_path": r.extra.get("horizons_csv_path", ""),
                "merged_csv_path": r.extra.get("merged_csv_path", ""),
                "possible_solutions_csv_path": r.extra.get("possible_solutions_csv_path", ""),
                "best_by_order_csv_path": r.extra.get("best_by_order_csv_path", ""),
                "sigma_periodogram_png_path": r.extra.get("sigma_periodogram_png_path", ""),
                "periodogram_png_path": r.extra.get("periodogram_png_path", ""),
                "phased_png_path": r.extra.get("phased_png_path", ""),
                "residuals_png_path": r.extra.get("residuals_png_path", ""),
                "lsm_ok": str(r.status == "ok" and r.best_period_days is not None),
                "color_taxonomy_status": color_tax.status,
                "color_g-r": (
                    "" if try_float(color_tax.colors.get("g-r")) is None
                    else f"{float(color_tax.colors['g-r']):.12f}"
                ),
                "color_g-i": (
                    "" if try_float(color_tax.colors.get("g-i")) is None
                    else f"{float(color_tax.colors['g-i']):.12f}"
                ),
                "color_r-i": (
                    "" if try_float(color_tax.colors.get("r-i")) is None
                    else f"{float(color_tax.colors['r-i']):.12f}"
                ),
                "color_i-z": (
                    "" if try_float(color_tax.colors.get("i-z")) is None
                    else f"{float(color_tax.colors['i-z']):.12f}"
                ),
                "taxonomy": color_tax.taxonomy_info.get("taxonomy", ""),
                "gri_slope": (
                    "" if try_float(color_tax.taxonomy_info.get("gri_slope")) is None
                    else f"{float(color_tax.taxonomy_info['gri_slope']):.12f}"
                ),
            }
            w.writerow(row)


def write_comparison_txt(
    path: str,
    rows: List[MethodResult],
    csv_path: str,
    args: argparse.Namespace,
    color_tax: ColorTaxonomyAssessment,
    table2_row: Optional[Table2LikeSummary] = None,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Period comparison\n")
        f.write("=================\n\n")
        f.write(f"csv={args.csv}\n")
        f.write(f"target={args.target}\n")
        f.write(f"location={args.location}\n")
        f.write(f"id_type={args.id_type}\n")
        f.write(f"step_minutes={args.step_minutes}\n")
        f.write(f"pad_minutes={args.pad_minutes}\n")
        f.write(f"min_period_days={args.min_period_days}\n")
        f.write(f"max_period_days={args.max_period_days}\n")
        f.write(f"oversample={args.oversample}\n")
        f.write(f"comparison_csv={csv_path}\n\n")

        for r in rows:
            f.write(f"[{r.method}]\n")
            f.write(f"status={r.status}\n")
            f.write(f"best_period_days={'' if r.best_period_days is None else f'{r.best_period_days:.12f}'}\n")
            f.write(f"best_period_hours={'' if r.best_period_hours is None else f'{r.best_period_hours:.12f}'}\n")
            f.write(f"period_error_days={'' if r.period_error_days is None else f'{r.period_error_days:.12f}'}\n")
            f.write(f"period_error_hours={'' if r.period_error_hours is None else f'{r.period_error_hours:.12f}'}\n")
            f.write(f"summary_path={r.summary_path or ''}\n")
            for path_key in [
                "horizons_csv_path",
                "merged_csv_path",
                "possible_solutions_csv_path",
                "best_by_order_csv_path",
                "sigma_periodogram_png_path",
                "periodogram_png_path",
                "phased_png_path",
                "residuals_png_path",
            ]:
                if path_key in r.extra:
                    f.write(f"{path_key}={r.extra[path_key]}\n")
            for k in sorted(r.extra):
                if k == "returncode" or k in {
                    "horizons_csv_path",
                    "merged_csv_path",
                    "possible_solutions_csv_path",
                    "best_by_order_csv_path",
                    "sigma_periodogram_png_path",
                    "periodogram_png_path",
                    "phased_png_path",
                    "residuals_png_path",
                }:
                    continue
                f.write(f"{k}={r.extra[k]}\n")
            f.write("\n")

        f.write(f"lsm_ok={str(rows[0].status == 'ok' and rows[0].best_period_days is not None)}\n")
        f.write("\n[Colors and taxonomy]\n")
        f.write(f"status={color_tax.status}\n")
        f.write(f"summary_path={color_tax.summary_path or ''}\n")
        f.write(f"colors_csv_path={color_tax.colors_csv_path or ''}\n")
        f.write(f"taxonomy_csv_path={color_tax.taxonomy_csv_path or ''}\n")
        f.write(f"taxonomy_band_csv_path={color_tax.taxonomy_band_csv_path or ''}\n")
        f.write(f"taxonomy_plot_png_path={color_tax.taxonomy_plot_png_path or ''}\n")
        f.write(f"taxonomy_plot_pdf_path={color_tax.taxonomy_plot_pdf_path or ''}\n")
        f.write(f"taxonomy_plot_svg_path={color_tax.taxonomy_plot_svg_path or ''}\n")
        f.write(f"mean_alpha_deg={'' if color_tax.mean_alpha_deg is None else f'{color_tax.mean_alpha_deg:.12f}'}\n")
        f.write(f"mean_alpha_error_deg={'' if color_tax.mean_alpha_error_deg is None else f'{color_tax.mean_alpha_error_deg:.12f}'}\n")
        for band in ["u", "g", "r", "i", "z", "y"]:
            val = try_float(color_tax.offsets.get(band))
            f.write(f"offset_{band}_minus_r={'' if val is None else f'{val:.12f}'}\n")
        for name in ["g-r", "g-i", "r-i", "i-z"]:
            val = try_float(color_tax.colors.get(name))
            f.write(f"color_{name.replace('-', '_')}={'' if val is None else f'{val:.12f}'}\n")
        f.write(f"taxonomy={color_tax.taxonomy_info.get('taxonomy', '')}\n")
        for key in ["g_mag", "r_mag", "i_mag", "z_mag", "gr", "ri", "iz", "gri_slope"]:
            val = color_tax.taxonomy_info.get(key)
            if isinstance(val, (int, float)):
                f.write(f"{key}={float(val):.12f}\n")
            else:
                f.write(f"{key}={val if val is not None else ''}\n")

        f.write("\n[Summary]\n")
        if table2_row is not None:
            f.write(f"Period (LSM / Fourier) [h]: {format_float(table2_row.period_lsm_hours, '.6f')} / {format_float(table2_row.period_fourier_hours, '.6f')}\n")
            f.write(f"Period error from Montgomery (LSM / Fourier) [h]: {format_float(table2_row.period_lsm_error_hours, '.6f')} / {format_float(table2_row.period_fourier_error_hours, '.6f')}\n")
            f.write(f"Amplitude (LSM / Fourier) [mag]: {format_float(table2_row.amplitude_lsm_mag, '.6f')} / {format_float(table2_row.amplitude_fourier_mag, '.6f')}\n")
            f.write(f"Amplitude error (LSM / Fourier) [mag]: {format_float(table2_row.amplitude_error_lsm_mag, '.6f')} / {format_float(table2_row.amplitude_error_fourier_mag, '.6f')}\n")
            f.write(f"Axial elongation (paper-style, from LSM amplitude): {format_float(table2_row.axial_elongation, '.6f')}\n")
            f.write(f"Axial elongation error: {format_float(table2_row.axial_elongation_error, '.6f')}\n")
            f.write(f"Mean alpha used for axial elongation: {format_float(table2_row.mean_alpha_deg, '.6f')} deg\n")
            f.write(f"Mean alpha error used for axial elongation: {format_float(table2_row.mean_alpha_error_deg, '.6f')} deg\n")
            
            f.write(f"g-r (LSM / Fourier): {format_float(table2_row.color_gr_lsm, '.6f')} / {format_float(table2_row.color_gr_fourier, '.6f')}\n")
            f.write(f"g-r error (LSM / Fourier): {format_float(table2_row.color_gr_lsm_error, '.6f')} / {format_float(table2_row.color_gr_fourier_error, '.6f')}\n")

            f.write(f"g-i (LSM / Fourier): {format_float(table2_row.color_gi_lsm, '.6f')} / {format_float(table2_row.color_gi_fourier, '.6f')}\n")
            f.write(f"g-i error (LSM / Fourier): {format_float(table2_row.color_gi_lsm_error, '.6f')} / {format_float(table2_row.color_gi_fourier_error, '.6f')}\n")

            f.write(f"r-i (LSM / Fourier): {format_float(table2_row.color_ri_lsm, '.6f')} / {format_float(table2_row.color_ri_fourier, '.6f')}\n")
            f.write(f"r-i error (LSM / Fourier): {format_float(table2_row.color_ri_lsm_error, '.6f')} / {format_float(table2_row.color_ri_fourier_error, '.6f')}\n")

            f.write(f"i-z (LSM / Fourier): {format_float(table2_row.color_iz_lsm, '.6f')} / {format_float(table2_row.color_iz_fourier, '.6f')}\n")
            f.write(f"i-z error (LSM / Fourier): {format_float(table2_row.color_iz_lsm_error, '.6f')} / {format_float(table2_row.color_iz_fourier_error, '.6f')}\n")

            f.write(f"gri slope (LSM / Fourier): {format_float(color_tax.taxonomy_info.get('gri_slope'), '.6f')} / nan\n")
            f.write(f"gri slope error (LSM / Fourier): {format_float(color_tax.taxonomy_info.get('gri_slope_error'), '.6f')} / nan\n")
            f.write(f"Taxonomy (LSM / Fourier): {table2_row.taxonomy_lsm or 'nan'} / {table2_row.taxonomy_fourier or 'nan'}\n")
        else:
            row0 = rows[0] if rows else None
            f.write(f"Period (LSM / Fourier) [h]: {format_float(row0.best_period_hours if row0 else None, '.6f')} / nan\n")
            f.write(f"Period error from Montgomery (LSM / Fourier) [h]: {format_float(row0.period_error_hours if row0 else None, '.6f')} / nan\n")
            f.write(f"Amplitude (LSM / Fourier) [mag]: {format_float(rows[0].amplitude_mag, '.6f')} / {'nan'}\n")
            f.write(f"Axial elongation (paper-style, from LSM amplitude): {'nan'}\n")
            f.write(f"g-r (LSM / Fourier): {format_float(color_tax.colors.get('g-r'), '.6f')} / {'nan'}\n")
            f.write(f"g-i (LSM / Fourier): {format_float(color_tax.colors.get('g-i'), '.6f')} / {'nan'}\n")
            f.write(f"r-i (LSM / Fourier): {format_float(color_tax.colors.get('r-i'), '.6f')} / {'nan'}\n")
            f.write(f"i-z (LSM / Fourier): {format_float(color_tax.colors.get('i-z'), '.6f')} / {'nan'}\n")
            f.write(f"Mean alpha used for axial elongation: {format_float(color_tax.mean_alpha_deg, '.6f')} deg\n")
            f.write(f"Mean alpha error used for axial elongation: {format_float(color_tax.mean_alpha_error_deg, '.6f')} deg\n")
            f.write(f"gri slope (LSM / Fourier): {format_float(table2_row.gri_slope_lsm, '.6f')} / {format_float(table2_row.gri_slope_fourier, '.6f')}\n")
            f.write(f"gri slope error (LSM / Fourier): {format_float(table2_row.gri_slope_lsm_error, '.6f')} / {format_float(table2_row.gri_slope_fourier_error, '.6f')}\n")
            f.write(f"Taxonomy (LSM / Fourier): {color_tax.taxonomy_info.get('taxonomy', 'nan')} / {'nan'}\n")


def print_table(rows: List[MethodResult]) -> None:
    headers = ["Method", "Status", "Period (days)", "Period (hours)", "Period err (hours)", "Summary file", "Phased PNG"]
    data = [
        [
            r.method,
            r.status,
            format_float(r.best_period_days, ".10f"),
            format_float(r.best_period_hours, ".8f"),
            format_float(r.period_error_hours, ".8f"),
            r.summary_path or "nan",
            r.extra.get("phased_png_path", "nan"),
        ]
        for r in rows
    ]

    widths = [len(h) for h in headers]
    for row in data:
        for i, x in enumerate(row):
            widths[i] = max(widths[i], len(str(x)))

    def fmt(row: List[str]) -> str:
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))

    print("\nComparison")
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    print(fmt(headers))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in data:
        print(fmt(row))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    if len(rows) >= 2:
        r1, r2 = rows[0], rows[1]
        if r1.best_period_hours is not None and r2.best_period_hours is not None:
            dh = r1.best_period_hours - r2.best_period_hours
            print(f"Delta (High-Order Fourier - Multiband Lomb-Scargle): {dh:.8f} h")


def print_lsm_status(lsm_ok: bool, r32: MethodResult, args: argparse.Namespace) -> None:
    print("\nLSM fit status")
    print("--------------")
    print(f"Method: {r32.method}")
    print(f"Status: {r32.status}")
    print(f"Best period (days): {format_float(r32.best_period_days, '.10f')}")
    print(f"Best period (hours): {format_float(r32.best_period_hours, '.8f')}")
    print(f"Period error from Montgomery (hours): {format_float(r32.period_error_hours, '.8f')}")
    print(f"LSM fit succeeded: {lsm_ok}")


def print_colors_taxonomy(color_tax: ColorTaxonomyAssessment) -> None:
    print("\nColors and taxonomy")
    print("-------------------")
    print(f"Status: {color_tax.status}")

    if color_tax.status != "ok":
        return

    print("Offsets relative to r:")
    for band in ["u", "g", "r", "i", "z", "y"]:
        print(f"  {band}-r offset = {format_float(try_float(color_tax.offsets.get(band)), '.6f')}")

    print("Derived colors:")
    for name in ["g-r", "g-i", "r-i", "i-z"]:
        print(f"  {name} = {format_float(try_float(color_tax.colors.get(name)), '.6f')}")

    print("Derived taxonomy metrics:")
    for key in ["g_mag", "r_mag", "i_mag", "z_mag", "gr", "ri", "iz", "gri_slope"]:
        print(f"  {key} = {format_float(try_float(color_tax.taxonomy_info.get(key)), '.6f')}")
    print(f"  taxonomy = {color_tax.taxonomy_info.get('taxonomy', 'NA')}")




def compute_zero_phase_amplitude(amplitude_mag: Optional[float], mean_alpha_deg: Optional[float], slope_m: float = 0.02) -> Optional[float]:
    amp = try_float(amplitude_mag)
    alpha = try_float(mean_alpha_deg)
    if amp is None:
        return None
    if alpha is None:
        return amp
    denom = 1.0 + slope_m * alpha
    if denom <= 0:
        return None
    return amp / denom


def compute_axial_elongation_from_amplitude(amplitude_mag: Optional[float], mean_alpha_deg: Optional[float], slope_m: float = 0.02) -> Optional[float]:
    a0 = compute_zero_phase_amplitude(amplitude_mag, mean_alpha_deg, slope_m=slope_m)
    if a0 is None:
        return None
    return float(10.0 ** (0.4 * a0))


def compute_axial_elongation_error_from_amplitude(
    amplitude_mag: Optional[float],
    amplitude_error_mag: Optional[float],
    mean_alpha_deg: Optional[float],
    alpha_error_deg: Optional[float],
    slope_m: float = 0.02,
) -> Optional[float]:
    """
    Propagate amplitude error and phase-angle error through
    A0 = A / (1 + m*alpha)  and  a/b = 10^(0.4*A0).
    """
    amp = try_float(amplitude_mag)
    amp_err = try_float(amplitude_error_mag)
    alpha = try_float(mean_alpha_deg)
    alpha_err = try_float(alpha_error_deg)

    if amp is None or amp_err is None:
        return None

    # In batch mode, do not crash the wrapper if phase-angle information is unavailable.
    # The table will show nan for axial_elongation_error.
    if alpha is None or alpha_err is None:
        return None

    denom = 1.0 + slope_m * alpha
    if denom <= 0:
        return None

    a0 = amp / denom

    dA0_dA = 1.0 / denom
    dA0_dalpha = -slope_m * a0 / denom

    var_a0 = (dA0_dA * amp_err) ** 2
    var_a0 += (dA0_dalpha * alpha_err) ** 2

    sigma_a0 = var_a0 ** 0.5

    elongation = 10.0 ** (0.4 * a0)
    sigma_elongation = 0.4 * math.log(10.0) * elongation * sigma_a0

    return float(sigma_elongation)



def build_fourier_colors(
    r31: MethodResult,
    color_tax_hof: ColorTaxonomyAssessment,
) -> Dict[str, Optional[float]]:
    if color_tax_hof.status == "ok":
        color_gr = try_float(color_tax_hof.colors.get("g-r"))
        color_gi = try_float(color_tax_hof.colors.get("g-i"))
        color_ri = try_float(color_tax_hof.colors.get("r-i"))
        color_iz = try_float(color_tax_hof.colors.get("i-z"))
    else:
        color_gr = try_float(r31.extra.get("color_g_r"))
        color_gi = try_float(r31.extra.get("color_g_i"))
        color_ri = try_float(r31.extra.get("color_r_i"))
        color_iz = try_float(r31.extra.get("color_i_z"))

    return {
        "g-r": color_gr,
        "g-i": color_gi,
        "r-i": color_ri,
        "i-z": color_iz,
    }


def summarize_table2_like(
    args: argparse.Namespace,
    r32: MethodResult,
    color_tax: ColorTaxonomyAssessment,
) -> Table2LikeSummary:
    df = pd.read_csv(args.csv)
    if "band" in df.columns:
        df["band"] = df["band"].map(normalize_band_label)
    if "mag" in df.columns:
        df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    if "mjd" in df.columns:
        df["mjd"] = pd.to_numeric(df["mjd"], errors="coerce")

    observation_date_range = format_observation_date_range(df)

    r_mask = (df["band"] == "r") if "band" in df.columns else pd.Series(False, index=df.index)
    mean_mag_r = None
    mean_mag_r_std = None
    if "mag" in df.columns and r_mask.any():
        r_mags = pd.to_numeric(df.loc[r_mask, "mag"], errors="coerce")
        finite_r_mags = r_mags[np.isfinite(r_mags)]
        if len(finite_r_mags) > 0:
            mean_mag_r = try_float(np.nanmean(finite_r_mags))
        if len(finite_r_mags) > 1:
            mean_mag_r_std = try_float(np.nanstd(finite_r_mags, ddof=1))
    arc_days = None
    if "mjd" in df.columns and np.isfinite(df["mjd"]).any():
        arc_days = try_float(np.nanmax(df["mjd"]) - np.nanmin(df["mjd"]))

    h_mag = get_jpl_sbdb_h(args.target)
    if h_mag is None:
        h_mag = None

    axial_elongation = compute_axial_elongation_from_amplitude(
        r32.amplitude_mag,
        color_tax.mean_alpha_deg if color_tax.status == "ok" else None,
    )

    return Table2LikeSummary(
        designation=args.target,
        h_mag=h_mag,
        mean_mag_r=mean_mag_r,
        mean_mag_r_std=mean_mag_r_std,
        number_observations=int(len(df)),
        observation_date_range=observation_date_range,
        arc_days=arc_days,
        period_lsm_hours=r32.best_period_hours,
        period_lsm_error_hours=r32.period_error_hours,
        period_fourier_hours=None,
        period_fourier_error_hours=None,
        amplitude_lsm_mag=r32.amplitude_mag,
        amplitude_error_lsm_mag=r32.amplitude_error_mag,
        amplitude_fourier_mag=None,
        amplitude_error_fourier_mag=None,
        axial_elongation=axial_elongation,
        axial_elongation_error=compute_axial_elongation_error_from_amplitude(
            r32.amplitude_mag,
            r32.amplitude_error_mag,
            color_tax.mean_alpha_deg,
            color_tax.mean_alpha_error_deg,
        ),
        mean_alpha_deg=color_tax.mean_alpha_deg if color_tax.status == "ok" else None,
        mean_alpha_error_deg=color_tax.mean_alpha_error_deg if color_tax.status == "ok" else None,
        color_gr_lsm=try_float(color_tax.colors.get("g-r")),
        color_gr_fourier=None,
        color_gr_lsm_error=try_float(color_tax.color_errors.get("g-r")),
        color_gr_fourier_error=None,
        color_gi_lsm=try_float(color_tax.colors.get("g-i")),
        color_gi_fourier=None,
        color_gi_lsm_error=try_float(color_tax.color_errors.get("g-i")),
        color_gi_fourier_error=None,
        color_ri_lsm=try_float(color_tax.colors.get("r-i")),
        color_ri_fourier=None,
        color_ri_lsm_error=try_float(color_tax.color_errors.get("r-i")),
        color_ri_fourier_error=None,
        color_iz_lsm=try_float(color_tax.colors.get("i-z")),
        color_iz_fourier=None,
        color_iz_lsm_error=try_float(color_tax.color_errors.get("i-z")),
        color_iz_fourier_error=None,
        gri_slope_lsm=try_float(color_tax.taxonomy_info.get("gri_slope")) if color_tax.status == "ok" else None,
        gri_slope_lsm_error=try_float(color_tax.taxonomy_info.get("gri_slope_error")) if color_tax.status == "ok" else None,
        gri_slope_fourier=None,
        gri_slope_fourier_error=None,
        taxonomy_lsm=color_tax.taxonomy_info.get("taxonomy") if color_tax.status == "ok" else None,
        taxonomy_fourier=None,
    )


def write_table2_csv(path: str, row: Table2LikeSummary) -> None:
    pd.DataFrame([vars(row)]).to_csv(path, index=False)


def write_table2_txt(path: str, row: Table2LikeSummary, color_tax: ColorTaxonomyAssessment) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Table-2-like summary\n")
        f.write("====================\n\n")
        for k, v in vars(row).items():
            f.write(f"{k}={'' if v is None else v}\n")
        f.write(f"mean_alpha_deg={'' if color_tax.mean_alpha_deg is None else color_tax.mean_alpha_deg}\n")
        f.write(f"mean_alpha_error_deg={'' if color_tax.mean_alpha_error_deg is None else color_tax.mean_alpha_error_deg}\n")
        f.write("axial_elongation_formula=a/b = 10^(0.4 * A0), with A0 = A/(1 + 0.02*alpha_mean) using LSM amplitude\n")
        f.write("axial_elongation_error_formula=standard first-order propagation through A0 and a/b using sigma_A and sigma_alpha, where sigma_alpha is std(alpha_deg) for inlier observations\n")
        f.write("h_mag_note=H from JPL SBDB is used as the table H proxy when available. If unavailable, the Fourier-fit H_r is used as a fallback.\n")


def print_table2_summary(row: Table2LikeSummary, color_tax: ColorTaxonomyAssessment) -> None:
    print("\nTable-2-like summary")
    print("--------------------")
    print(f"Designation: {row.designation}")
    print(f"H mag (JPL SBDB H): {format_float(row.h_mag, '.6f')}")
    print(f"Mean r-band mag: {format_float(row.mean_mag_r, '.6f')} +/- {format_float(row.mean_mag_r_std, '.6f')}")
    print(f"Number of observations: {row.number_observations}")
    print(f"Observation date range: {row.observation_date_range or 'NA'}")
    print(f"Arc (days): {format_float(row.arc_days, '.6f')}")
    print(f"Period (LSM / Fourier) [h]: {format_float(row.period_lsm_hours, '.6f')} / {format_float(row.period_fourier_hours, '.6f')}")
    print(f"Period error from Montgomery (LSM / Fourier) [h]: {format_float(row.period_lsm_error_hours, '.6f')} / {format_float(row.period_fourier_error_hours, '.6f')}")
    print(f"Amplitude (LSM / Fourier) [mag]: {format_float(row.amplitude_lsm_mag, '.6f')} / {format_float(row.amplitude_fourier_mag, '.6f')}")
    print(f"Amplitude error (LSM / Fourier) [mag]: {format_float(row.amplitude_error_lsm_mag, '.6f')} / {format_float(row.amplitude_error_fourier_mag, '.6f')}")
    print(f"Axial elongation (paper-style, from LSM amplitude): {format_float(row.axial_elongation, '.6f')}")
    print(f"Axial elongation error: {format_float(row.axial_elongation_error, '.6f')}")
    print(f"g-r (LSM / Fourier): {format_float(row.color_gr_lsm, '.6f')} / {format_float(row.color_gr_fourier, '.6f')}")
    print(f"g-r error (LSM / Fourier): {format_float(row.color_gr_lsm_error, '.6f')} / {format_float(row.color_gr_fourier_error, '.6f')}")
    print(f"g-i (LSM / Fourier): {format_float(row.color_gi_lsm, '.6f')} / {format_float(row.color_gi_fourier, '.6f')}")
    print(f"g-i error (LSM / Fourier): {format_float(row.color_gi_lsm_error, '.6f')} / {format_float(row.color_gi_fourier_error, '.6f')}")
    print(f"r-i (LSM / Fourier): {format_float(row.color_ri_lsm, '.6f')} / {format_float(row.color_ri_fourier, '.6f')}")
    print(f"r-i error (LSM / Fourier): {format_float(row.color_ri_lsm_error, '.6f')} / {format_float(row.color_ri_fourier_error, '.6f')}")
    print(f"i-z (LSM / Fourier): {format_float(row.color_iz_lsm, '.6f')} / {format_float(row.color_iz_fourier, '.6f')}")
    print(f"i-z error (LSM / Fourier): {format_float(row.color_iz_lsm_error, '.6f')} / {format_float(row.color_iz_fourier_error, '.6f')}")
    print(f"Mean alpha used for axial elongation: {format_float(row.mean_alpha_deg, '.6f')} deg")
    print(f"Mean alpha error used for axial elongation: {format_float(row.mean_alpha_error_deg, '.6f')} deg")
    print(f"gri slope (LSM / Fourier): {format_float(row.gri_slope_lsm, '.6f')} / {format_float(row.gri_slope_fourier, '.6f')}")
    print(f"gri slope error (LSM / Fourier): {format_float(row.gri_slope_lsm_error, '.6f')} / {format_float(row.gri_slope_fourier_error, '.6f')}")
    print(f"Taxonomy (LSM / Fourier): {row.taxonomy_lsm or 'nan'} / {row.taxonomy_fourier or 'nan'}")
    print(f"LSM fit succeeded: {row.period_lsm_hours is not None}")


def latex_escape(text_value: object) -> str:
    s = "" if text_value is None else str(text_value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def pair_fmt(a: Optional[float], b: Optional[float], fmt: str = ".2f") -> str:
    return f"{format_float(a, fmt)} / {format_float(b, fmt)}"


def pair_fmt_str(a: Optional[str], b: Optional[str]) -> str:
    return f"{latex_escape(a or 'nan')} / {latex_escape(b or 'nan')}"


def single_fmt(a: Optional[float], fmt: str = ".2f") -> str:
    return format_float(a, fmt)


def pair_fmt_with_error(a: Optional[float], a_err: Optional[float], b: Optional[float], b_err: Optional[float], fmt: str = ".2f") -> str:
    a_str = f"${format_float(a, fmt)} \\pm {format_float(a_err, fmt)}$" if a is not None and a_err is not None else format_float(a, fmt)
    b_str = f"${format_float(b, fmt)} \\pm {format_float(b_err, fmt)}$" if b is not None and b_err is not None else format_float(b, fmt)
    return f"{a_str} / {b_str}"

def single_fmt_with_error(a: Optional[float], a_err: Optional[float], fmt: str = ".2f") -> str:
    if a is not None and a_err is not None:
        return f"${format_float(a, fmt)} \\pm {format_float(a_err, fmt)}$"
    return format_float(a, fmt)


def write_table2_latex(path: str, row: Table2LikeSummary) -> None:
    caption = (
        r"Reliable rotation periods from our LSM and high-order Fourier analyses. "
        r"The table is prepared for A\&A in landscape format."
    )
    label = "tab:reliable_rotation_periods"
    colspec = "lcccccccccccccc"
    header = (
        r"Designation & H Mag & Mean Mag (r Band) & Number of Observations & Observation Date Range & "
        r"Period (LSM/Fourier) & Amplitude (LSM/Fourier) & Mean alpha & Axial Elongation & "
        r"$g-r$ (LSM/Fourier) & $g-i$ (LSM/Fourier) & $r-i$ (LSM/Fourier) & $i-z$ (LSM/Fourier) & "
        r"gri slope (LSM/Fourier) & Taxonomy (LSM/Fourier) \\"
    )
    data_row = (
        f"{latex_escape(row.designation)} & "
        f"{single_fmt(row.h_mag, '.2f')} & "
        f"{single_fmt_with_error(row.mean_mag_r, row.mean_mag_r_std, '.2f')} & "
        f"{row.number_observations} & "
        f"{latex_escape(row.observation_date_range or 'NA')} & "
        f"{pair_fmt_with_error(row.period_lsm_hours, row.period_lsm_error_hours, row.period_fourier_hours, row.period_fourier_error_hours, '.4f')} & "
        f"{pair_fmt_with_error(row.amplitude_lsm_mag, row.amplitude_error_lsm_mag, row.amplitude_fourier_mag, row.amplitude_error_fourier_mag, '.2f')} & "
        f"{single_fmt_with_error(row.mean_alpha_deg, row.mean_alpha_error_deg, '.1f')} & "
        f"{single_fmt_with_error(row.axial_elongation, row.axial_elongation_error, '.2f')} & "
        f"{pair_fmt_with_error(row.color_gr_lsm, row.color_gr_lsm_error, row.color_gr_fourier, row.color_gr_fourier_error, '.2f')} & "
        f"{pair_fmt_with_error(row.color_gi_lsm, row.color_gi_lsm_error, row.color_gi_fourier, row.color_gi_fourier_error, '.2f')} & "
        f"{pair_fmt_with_error(row.color_ri_lsm, row.color_ri_lsm_error, row.color_ri_fourier, row.color_ri_fourier_error, '.2f')} & "
        f"{pair_fmt_with_error(row.color_iz_lsm, row.color_iz_lsm_error, row.color_iz_fourier, row.color_iz_fourier_error, '.2f')} & "
        f"{pair_fmt_with_error(row.gri_slope_lsm, row.gri_slope_lsm_error, row.gri_slope_fourier, row.gri_slope_fourier_error, '.2f')} & "
        f"{pair_fmt_str(row.taxonomy_lsm, row.taxonomy_fourier)} \\\\"
    )
    lines = [
        r"\begin{landscape}",
        r"\begin{table}",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{width=\linewidth}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        header,
        r"\hline",
        data_row,
        r"\hline",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\begin{minipage}{\linewidth}",
        r"\vspace{2mm}",
        r"\footnotesize\textit{Notes.} H Mag is taken from JPL SBDB when available, otherwise the Section 3.1 fitted $H_r$ is used as a fallback. Axial elongation is computed from the LSM amplitude with a phase-angle correction. The axial-elongation uncertainty propagates both amplitude uncertainty and the standard deviation of inlier phase angles. gri slope is taken from the taxonomy workflow.",
        r"\end{minipage}",
        r"\end{table}",
        r"\end{landscape}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input photometry CSV")
    p.add_argument("--target", default="2026 DO14", help="Horizons target designation")
    p.add_argument("--location", default="500", help='Horizons observer code, e.g. "500" or "X05"')
    p.add_argument("--id-type", default=None, help="Optional Horizons id_type")
    p.add_argument("--step-minutes", type=int, default=1, help="Horizons ephemeris step size in minutes")
    p.add_argument("--pad-minutes", type=int, default=10, help="Padding around observation time range")

    p.add_argument("--min-period-days", type=float, default=0.00065) 
    p.add_argument("--max-period-days", type=float, default=3.0) 
    p.add_argument("--oversample", type=int, default=100,
               help="Uniform-period-grid oversampling factor for Section 3.2")

    p.add_argument("--min-obs-per-band", type=int, default=30,
                   help="Minimum number of observations required in a band")
    p.add_argument("--min-bands", type=int, default=2,
                   help="Minimum number of gri bands meeting the observation threshold")
    p.add_argument("--amplitude-threshold", type=float, default=0.1,
                   help="Minimum Sections 3.1 and 3.2 amplitudes required for reliability")
    p.add_argument("--period-agreement-tol", type=float, default=0.10,
                   help="Maximum fractional period disagreement for reliability")

    p.add_argument("--out-prefix", default="period_compare")
    p.add_argument(
        "--import-check",
        action="store_true",
        help="Only test dynamic imports of the sibling scripts, then exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    file_must_exist(SECTION32_PATH, "Section 3.2 script")
    file_must_exist(COLORS_TAXONOMY_PATH, "Color/taxonomy script")
    file_must_exist(args.csv, "Input CSV")

    if args.import_check:
        load_module("period_search_section32_mod", SECTION32_PATH)
        load_module("taxonomy_mod_check", COLORS_TAXONOMY_PATH)
        print("Import check succeeded.")
        print(f"  Loaded: {SECTION32_PATH}")
        print(f"  Loaded taxonomy workflow: {COLORS_TAXONOMY_PATH}")
        return

    print("Running Multiband Lomb-Scargle workflow...")
    r32 = run_section32(args)
    if r32.status.startswith("failed"):
        print("\n[Multiband Lomb-Scargle failed]")
        if "stderr" in r32.extra:
            print(r32.extra["stderr"])
        elif "stdout" in r32.extra:
            print(r32.extra["stdout"])

    print("Running color/taxonomy workflow (LSM period)...")
    try:
        color_tax = run_colors_taxonomy(args, r32)
    except Exception as exc:
        color_tax = ColorTaxonomyAssessment(
            summary_path=None,
            colors_csv_path=None,
            taxonomy_csv_path=None,
            taxonomy_band_csv_path=None,
            taxonomy_plot_png_path=None,
            taxonomy_plot_pdf_path=None,
            taxonomy_plot_svg_path=None,
            colors={},
            color_errors={},
            offsets={},
            taxonomy_info={},
            mean_alpha_deg=None,
            mean_alpha_error_deg=None,
            status=f"failed: {exc}",
        )

    rows = [r32]
    lsm_ok = assess_lsm_success(args, r32)

    print_table(rows)
    print_lsm_status(lsm_ok, r32, args)
    print_colors_taxonomy(color_tax)

    comparison_dir = Path("COMPARISON")
    comparison_dir.mkdir(exist_ok=True)

    comparison_csv = str(comparison_dir / f"{args.out_prefix}_comparison.csv")
    comparison_txt = str(comparison_dir / f"{args.out_prefix}_comparison.txt")

    table2_row = None
    if color_tax.status == "ok":
        table2_row = summarize_table2_like(args, r32, color_tax)

    write_comparison_csv(comparison_csv, rows, color_tax)
    write_comparison_txt(comparison_txt, rows, comparison_csv, args, color_tax, table2_row)

    print(f"\nWrote comparison CSV: {comparison_csv}")
    print(f"Wrote comparison TXT: {comparison_txt}")

    if color_tax.status == "ok":
        print(f"Wrote color/taxonomy summary: {color_tax.summary_path}")
        print(f"Wrote colors CSV: {color_tax.colors_csv_path}")
        print(f"Wrote taxonomy CSV: {color_tax.taxonomy_csv_path}")
        print(f"Wrote taxonomy band summary CSV: {color_tax.taxonomy_band_csv_path}")
        if r32.extra.get("horizons_csv_path"):
            print(f"Wrote LSM Horizons CSV: {r32.extra.get('horizons_csv_path')}")
        if r32.extra.get("merged_csv_path"):
            print(f"Wrote LSM merged CSV: {r32.extra.get('merged_csv_path')}")
        if r32.extra.get("periodogram_png_path"):
            print(f"Wrote LSM periodogram PNG: {r32.extra.get('periodogram_png_path')}")
        if r32.extra.get("phased_png_path"):
            print(f"Wrote LSM phased PNG: {r32.extra.get('phased_png_path')}")
        if color_tax.taxonomy_plot_png_path:
            print(f"Wrote taxonomy plot PNG: {color_tax.taxonomy_plot_png_path}")
        if color_tax.taxonomy_plot_pdf_path:
            print(f"Wrote taxonomy plot PDF: {color_tax.taxonomy_plot_pdf_path}")
        if color_tax.taxonomy_plot_svg_path:
            print(f"Wrote taxonomy plot SVG: {color_tax.taxonomy_plot_svg_path}")
        #table2_row = summarize_table2_like(args, r32, color_tax)

        result_dir = Path("TEX_TABLE")
        result_dir.mkdir(exist_ok=True)
        
        table2_csv = result_dir / f"{args.out_prefix}_table2_like_summary.csv"
        table2_txt = result_dir / f"{args.out_prefix}_table2_like_summary.txt"
        table2_tex = result_dir / f"{args.out_prefix}_table2_like_summary.tex"

        write_table2_csv(table2_csv, table2_row)
        write_table2_txt(table2_txt, table2_row, color_tax)
        write_table2_latex(table2_tex, table2_row)

        print_table2_summary(table2_row, color_tax)

        print(f"Wrote Table-2-like CSV: {table2_csv.resolve()}")
        print(f"Wrote Table-2-like TXT: {table2_txt.resolve()}")
        print(f"Wrote Table-2-like LaTeX: {table2_tex.resolve()}")
 

    if any(r.status.startswith("failed") for r in rows):
        print("\nOne or more methods failed. Check the method-specific outputs and summary files above.")

    print(f"\nTotal runtime: {time.perf_counter() - t0:.3f} s")


if __name__ == "__main__":
    main()


# python Asteroid_physical_properties.py --csv "DO14.csv" --target "2026 DO14" --out-prefix "2026_DO14"
# python Asteroid_physical_properties.py --csv "MU8.csv" --target "2025 MU8" --out-prefix "2025_MU8"


# python Asteroid_physical_properties.py --csv "2021_JZ24.csv" --target "2021 JZ24" --out-prefix "2021_JZ24" 



