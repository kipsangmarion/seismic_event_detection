"""
features/time_domain.py
Method 1 (Manual): Hand-crafted time-domain statistics.

Each feature was chosen for a specific physical reason:
  - RMS / max_amplitude : earthquakes carry far more ground-motion energy than noise
  - kurtosis            : earthquake onsets are impulsive (high kurtosis); noise is near-Gaussian
  - zero_crossing_rate  : proxy for dominant frequency content
  - skewness            : captures asymmetry in the wave envelope
"""

import numpy as np
import scipy.stats


def extract_time_domain(trace):
    """
    Extract time-domain features from a single ObsPy Trace.

    Parameters
    trace : obspy.Trace

    Returns
    dict[str, float]
    """
    x = trace.data.astype(float)

    return {
        "mean":               np.mean(x),
        "std":                np.std(x),
        "max_amplitude":      float(np.max(np.abs(x))),
        "rms":                float(np.sqrt(np.mean(x ** 2))),
        "zero_crossing_rate": float(np.mean(np.diff(np.sign(x)) != 0)),
        "kurtosis":           float(scipy.stats.kurtosis(x)),
        "skewness":           float(scipy.stats.skew(x)),
    }
