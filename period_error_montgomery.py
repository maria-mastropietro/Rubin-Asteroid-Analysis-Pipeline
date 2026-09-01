"""
Period Error Estimation Module

Functions for computing analytical errors in fitted periods, amplitudes, and phases
using the formulas from:

Montgomery, M. H., & O'Donoghue, D. 1999, Delta Scuti Star Newsletter, 13, 28
"A derivation of the errors for least squares fitting to time series data"
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_period_error(
    period_days: float,
    amplitude: float,
    n_data_points: int,
    time_span_days: float,
    residual_rms: float,
) -> float:
    """
    Compute the uncertainty in a fitted period using Montgomery & O'Donoghue (1999).
    
    Implements Equation 11 from the paper, with the transformation:
    σ(P) = P² * σ(f)
    
    Parameters
    ----------
    period_days : float
        Fitted rotational period in days
    amplitude : float
        Fitted amplitude of the sinusoidal signal (in magnitude units)
    n_data_points : int
        Total number of observations
    time_span_days : float
        Total time span of observations in days (T = tmax - tmin)
    residual_rms : float
        Root-mean-square of residuals (σ(m) in the paper).
        Computed as: residual_rms = sqrt( sum((data - model)²) / dof )
    
    Returns
    -------
    period_error : float
        Uncertainty in period, in the same units as period_days
    
    Raises
    ------
    ValueError
        If input parameters are invalid (negative, zero where required, etc.)
    
    Notes
    -----
    The formula is:
        σ(f) = sqrt(6) / (π * sqrt(N) * T) * (σ(m) / a)
        σ(P) = P² * σ(f)
    
    where:
        N = number of data points
        T = total time span in days
        a = amplitude
        σ(m) = RMS residual
        P = period
    
    This assumes:
    - White Gaussian noise (uncorrelated in time)
    - Single dominant sinusoidal component
    - Reasonably good signal-to-noise ratio (a >> σ(m))
    
    Under these conditions, the formula provides a lower limit on the true error.
    Correlated noise or multiple signals can increase true errors significantly.
    """
    # Input validation
    if n_data_points < 2:
        raise ValueError(f"n_data_points must be >= 2, got {n_data_points}")
    if time_span_days <= 0:
        raise ValueError(f"time_span_days must be positive, got {time_span_days}")
    if amplitude <= 0:
        raise ValueError(f"amplitude must be positive, got {amplitude}")
    if residual_rms < 0:
        raise ValueError(f"residual_rms cannot be negative, got {residual_rms}")
    if period_days <= 0:
        raise ValueError(f"period_days must be positive, got {period_days}")
    
    # Compute frequency error (Eq. 11 in Montgomery & O'Donoghue)
    sigma_freq = (
        np.sqrt(6.0)
        / (np.pi * np.sqrt(n_data_points) * time_span_days)
        * (residual_rms / amplitude)
    )
    
    # Convert to period error
    sigma_period = period_days**2 * sigma_freq
    
    return sigma_period

