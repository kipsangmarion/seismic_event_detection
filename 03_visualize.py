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
import matplotlib.gridspec as gridspec
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
def plot_spectrogram_grid():
    eq_traces    = load_n_traces(EARTHQUAKE_DIR, "event_", SAMPLES_PER_CLASS)
    noise_traces = load_n_traces(NOISE_DIR,      "noise_", SAMPLES_PER_CLASS)

    fig, axes = plt.subplots(
        2, SAMPLES_PER_CLASS,
        figsize=(5 * SAMPLES_PER_CLASS, 7),
        constrained_layout=True,
    )
    fig.suptitle("Spectrograms: Earthquake vs Noise", fontsize=14, fontweight="bold")

    row_labels = ["Earthquake", "Noise"]
    row_colors = [COLOR_EQ, COLOR_NOISE]
    all_traces = [eq_traces, noise_traces]

    for row, (traces, label, color) in enumerate(zip(all_traces, row_labels, row_colors)):
        for col, trace in enumerate(traces):
            ax  = axes[row][col]
            fs  = trace.stats.sampling_rate
            x   = trace.data.astype(float)

            ax.specgram(
                x,
                Fs=fs,
                NFFT=256,
                noverlap=128,
                cmap="inferno",
            )
            ax.set_ylim(0, FREQ_MAX_HZ)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"{label} {col + 1}", color=color, fontweight="bold")

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
    plot_spectrogram_grid()

    print("Generating PCA scatter plot ...")
    plot_pca_scatter(df)

    print("Generating feature histograms ...")
    plot_feature_histograms(df)

    print(f"\nAll visualizations saved to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
