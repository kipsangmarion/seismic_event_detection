"""
features/spectral.py
Method 2 (Spectral): FFT-based frequency-band energy features.

Frequency bands are chosen to match seismologically meaningful ranges:
  0.1-1  Hz  : surface waves, long-period body waves
  1-5    Hz  : regional P and S waves (core detection band)
  5-10   Hz  : local/shallow earthquake high-frequency content
  10-20  Hz  : near-source high-frequency; mostly noise at teleseismic distances

dominant_frequency captures where most of the signal power sits.
"""

import numpy as np
from scipy.fft import fft, fftfreq

BANDS = [
    (0.1, 1.0),
    (1.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
]


def extract_spectral(trace):
    """
    Extract spectral energy features from a single ObsPy Trace.

    Parameters
    trace : obspy.Trace

    Returns
    dict[str, float]
    """
    x  = trace.data.astype(float)
    fs = trace.stats.sampling_rate
    N  = len(x)

    freqs = fftfreq(N, d=1.0 / fs)
    power = np.abs(fft(x)) ** 2

    features = {}
    for f_low, f_high in BANDS:
        mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) < f_high)
        key  = f"energy_{f_low}_{f_high}hz".replace(".", "p")
        features[key] = float(np.sum(power[mask]))

    # dominant frequency from the positive half of the spectrum
    pos_mask = freqs > 0
    features["dominant_frequency"] = float(
        freqs[pos_mask][np.argmax(power[pos_mask])]
    )

    return features
