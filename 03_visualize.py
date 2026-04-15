"""
03_visualize.py
Generates three visualization PNGs saved to visualizations/:

  01_spectrogram_grid.png  - 2x3 grid of spectrograms (earthquake vs noise)
  02_pca_scatter.png       - PC1 vs PC2 scatter plot colored by class
  03_feature_histograms.png - overlapping histograms for RMS, kurtosis, max_amplitude

Reads:
  features.csv             - output of 02_extract_features.py
  data/earthquake/*.mseed  - for raw waveforms used in spectrogram
  data/noise/*.mseed       - for raw waveforms used in spectrogram
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import spectrogram as scipy_spectrogram
from dotenv import load_dotenv
from obspy import read

load_dotenv()

# Config
EARTHQUAKE_DIR = os.getenv("EARTHQUAKE_DIR", "data/earthquake")
NOISE_DIR      = os.getenv("NOISE_DIR",      "data/noise")
FEATURES_FILE  = os.getenv("FEATURES_FILE",  "features.csv")
VIZ_DIR        = "visualizations"

SAMPLES_PER_CLASS = 3     # number of waveforms shown in the spectrogram grid
FREQ_MAX_HZ       = 20.0  # cap y-axis on spectrograms (seismic range of interest)

# consistent class colors used across all plots
COLOR_EQ    = "#d62728"   # red    — earthquake
COLOR_NOISE = "#1f77b4"   # blue   — noise


# Helpers
def load_n_traces(directory, prefix, n):
    """Load the first n readable traces from a directory."""
    traces = []
    for fname in sorted(os.listdir(directory)):
        if len(traces) == n:
            break
        if not (fname.startswith(prefix) and fname.endswith(".mseed")):
            continue
        try:
            st = read(os.path.join(directory, fname))
            st.merge(method=1, fill_value=0)
            if len(st) > 0 and len(st[0].data) > 0:
                traces.append(st[0])
        except Exception:
            continue
    return traces


# Plot 1: Spectrogram Grid
def select_representative_files(df, directory, prefix, label, n, highest=True):
    """
    Pick n files from directory ranked by max_amplitude.
    highest=True  → top n (best earthquake examples)
    highest=False → bottom n (cleanest noise examples)
    """
    subset = df[df["label"] == label].copy()
    subset = subset.sort_values("max_amplitude", ascending=not highest)
    selected = subset.head(n)["filename"].tolist()
    traces = []
    for fname in selected:
        path = os.path.join(directory, fname)
        if not os.path.exists(path):
            continue
        try:
            st = read(path)
            st.merge(method=1, fill_value=0)
            if len(st) > 0 and len(st[0].data) > 0:
                traces.append((fname, st[0]))
        except Exception:
            continue
    return traces


def compute_spectrogram(trace, nfft=256, noverlap=128):
    """Return (freqs, times, log_power) for a single trace."""
    fs    = trace.stats.sampling_rate
    x     = trace.data.astype(float)
    freqs, times, Sxx = scipy_spectrogram(x, fs=fs, nperseg=nfft, noverlap=noverlap)
    # Convert to dB, clip floor to avoid log(0)
    log_power = 10 * np.log10(np.maximum(Sxx, 1e-20))
    # Restrict to seismic frequency range of interest
    freq_mask = freqs <= FREQ_MAX_HZ
    return freqs[freq_mask], times, log_power[freq_mask, :]


def plot_spectrogram_grid(df):
    # Pick the most contrasting examples: highest-amplitude earthquakes, lowest-amplitude noise
    eq_traces    = select_representative_files(df, EARTHQUAKE_DIR, "event_", label=1,
                                               n=SAMPLES_PER_CLASS, highest=True)
    noise_traces = select_representative_files(df, NOISE_DIR,      "noise_", label=0,
                                               n=SAMPLES_PER_CLASS, highest=False)
    all_traces   = [eq_traces, noise_traces]

    # Step 1: compute all spectrograms first to find global power range
    spectrograms = []
    for traces in all_traces:
        row_specs = []
        for fname, trace in traces:
            freqs, times, log_power = compute_spectrogram(trace)
            row_specs.append((fname, freqs, times, log_power))
        spectrograms.append(row_specs)

    all_powers = np.concatenate([s[2].ravel() for row in spectrograms for s in row])
    vmin = np.percentile(all_powers, 5)   # ignore bottom 5% (noise floor)
    vmax = np.percentile(all_powers, 99)  # ignore top 1% (spikes)

    # Step 2: plot with shared color scale
    fig, axes = plt.subplots(
        2, SAMPLES_PER_CLASS,
        figsize=(5 * SAMPLES_PER_CLASS, 7),
        constrained_layout=True,
    )
    fig.suptitle("Spectrograms: Earthquake vs Noise (shared color scale, dB)",
                 fontsize=14, fontweight="bold")

    row_labels = ["Earthquake", "Noise"]
    row_colors = [COLOR_EQ, COLOR_NOISE]
    im = None

    for row, (specs, label, color) in enumerate(zip(spectrograms, row_labels, row_colors)):
        for col, (fname, freqs, times, log_power) in enumerate(specs):
            ax = axes[row][col]
            im = ax.pcolormesh(times, freqs, log_power,
                               cmap="inferno", vmin=vmin, vmax=vmax,
                               shading="auto")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            short = fname.replace(".mseed", "")
            ax.set_title(f"{label} — {short}", color=color, fontweight="bold", fontsize=9)

    # shared colorbar on the right
    if im is not None:
        fig.colorbar(im, ax=axes, label="Power (dB)", shrink=0.6, pad=0.02)

    out_path = os.path.join(VIZ_DIR, "01_spectrogram_grid.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


# Plot 2: PCA Scatter
def plot_pca_scatter(df):
    if "pc1" not in df.columns or "pc2" not in df.columns:
        print("[skip] PCA columns not found in features.csv — skipping scatter plot.")
        return

    eq    = df[df["label"] == 1]
    noise = df[df["label"] == 0]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(noise["pc1"], noise["pc2"],
               c=COLOR_NOISE, label="Noise", alpha=0.6, s=30, edgecolors="none")
    ax.scatter(eq["pc1"],    eq["pc2"],
               c=COLOR_EQ,    label="Earthquake", alpha=0.6, s=30, edgecolors="none")

    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title("PCA Scatter: PC1 vs PC2 by Class", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(VIZ_DIR, "02_pca_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# Plot 3: Feature Histograms
def plot_feature_histograms(df):
    features_to_plot = [
        ("rms",           "RMS Amplitude"),
        ("kurtosis",      "Kurtosis"),
        ("max_amplitude", "Max Amplitude"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle("Feature Distributions: Earthquake vs Noise", fontsize=14, fontweight="bold")

    eq    = df[df["label"] == 1]
    noise = df[df["label"] == 0]

    for ax, (col, title) in zip(axes, features_to_plot):
        if col not in df.columns:
            ax.set_title(f"{title}\n(not found)")
            continue

        eq_vals    = eq[col].dropna()
        noise_vals = noise[col].dropna()

        bins = np.linspace(
            min(eq_vals.min(), noise_vals.min()),
            max(eq_vals.max(), noise_vals.max()),
            40,
        )

        ax.hist(noise_vals, bins=bins, color=COLOR_NOISE, alpha=0.6,
                label="Noise",      density=True)
        ax.hist(eq_vals,    bins=bins, color=COLOR_EQ,    alpha=0.6,
                label="Earthquake", density=True)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(VIZ_DIR, "03_feature_histograms.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# Plot 4: Spectral Feature Scatter
def plot_spectral_scatter(df):
    x_col = "energy_1p0_5p0hz"   # core seismic detection band
    y_col = "energy_0p1_1p0hz"   # surface wave / long-period band

    if x_col not in df.columns or y_col not in df.columns:
        print("[skip] Spectral columns not found in features.csv — skipping spectral scatter.")
        return

    eq    = df[df["label"] == 1]
    noise = df[df["label"] == 0]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use log scale — spectral energy spans many orders of magnitude
    x_eq    = np.log10(eq[x_col].clip(lower=1))
    y_eq    = np.log10(eq[y_col].clip(lower=1))
    x_noise = np.log10(noise[x_col].clip(lower=1))
    y_noise = np.log10(noise[y_col].clip(lower=1))

    ax.scatter(x_noise, y_noise,
               c=COLOR_NOISE, label="Noise", alpha=0.6, s=30, edgecolors="none")
    ax.scatter(x_eq,    y_eq,
               c=COLOR_EQ,    label="Earthquake", alpha=0.6, s=30, edgecolors="none")

    ax.set_xlabel("log₁₀ Energy: 1–5 Hz band (core seismic)", fontsize=12)
    ax.set_ylabel("log₁₀ Energy: 0.1–1 Hz band (surface waves)", fontsize=12)
    ax.set_title("Spectral Feature Scatter: Earthquake vs Noise", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(VIZ_DIR, "04_spectral_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# Main
def main():
    os.makedirs(VIZ_DIR, exist_ok=True)

    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(
            f"{FEATURES_FILE} not found. Run 02_extract_features.py first."
        )

    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} rows from {FEATURES_FILE}\n")

    print("Generating spectrogram grid ...")
    plot_spectrogram_grid(df)

    print("Generating PCA scatter plot ...")
    plot_pca_scatter(df)

    print("Generating feature histograms ...")
    plot_feature_histograms(df)

    print("Generating spectral feature scatter ...")
    plot_spectral_scatter(df)

    print(f"\nAll visualizations saved to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
