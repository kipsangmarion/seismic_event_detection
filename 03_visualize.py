"""
03_visualize.py
Generates all visualization PNGs saved to visualizations/:

  Feature extraction plots (always generated):
    01_spectrogram_grid.png   - 2x3 grid of spectrograms (earthquake vs noise)
    02_pca_scatter.png        - PC1 vs PC2 scatter plot colored by class
    03_feature_histograms.png - overlapping histograms for RMS, max_amp, dom_freq
    04_spectral_scatter.png   - spectral band energy scatter
    05_raw_waveforms.png      - raw waveform panel
    06_iqr_scatter.png        - IQR outlier filter scatter
    07_pca_variance.png       - PCA explained variance

  Classifier result plots (requires classifier_results.pkl from 04_classify.py):
    08_roc_curves.png         - all 7 methods on one plot
    09_confusion_gb.png       - Gradient Boosting confusion matrix
    10_lasso_sparsity.png     - accuracy vs C with active feature count
    11_lr_coefficients.png    - LR feature coefficients
    12_accuracy_auc_bars.png  - accuracy + AUC, HC vs PCA per method
    13_lasso_survivors.png    - surviving LASSO feature coefficients
    14_gb_importances.png     - Gradient Boosting feature importances
    15_cv_bars.png            - 10-fold CV accuracy with error bars

Reads:
  features.csv              - output of 02_extract_features.py
  classifier_results.pkl    - output of 04_classify.py (optional)
  data/earthquake/*.mseed   - for raw waveforms used in spectrogram
  data/noise/*.mseed        - for raw waveforms used in spectrogram
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.signal import spectrogram as scipy_spectrogram
from dotenv import load_dotenv
from obspy import read
from sklearn.metrics import confusion_matrix

load_dotenv()

# Config
EARTHQUAKE_DIR = os.getenv("EARTHQUAKE_DIR", "data/earthquake")
NOISE_DIR      = os.getenv("NOISE_DIR",      "data/noise")
FEATURES_FILE  = os.getenv("FEATURES_FILE",  "features.csv")
VIZ_DIR        = "visualizations"
MODELS_DIR     = "models"

SAMPLES_PER_CLASS = 3     # number of waveforms shown in the spectrogram grid
FREQ_MAX_HZ       = 20.0  # cap y-axis on spectrograms (seismic range of interest)

TARGET_ACC = 0.85
PALETTE    = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
              "#e377c2", "#8c564b", "#bcbd22"]

# consistent class colors used across all plots
COLOR_EQ    = "#d62728"   # red - earthquake
COLOR_NOISE = "#1f77b4"   # blue - noise


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
    vmin = np.percentile(all_powers, 10)  # tighter clip for better contrast
    vmax = np.percentile(all_powers, 98)

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
                               cmap="turbo", vmin=vmin, vmax=vmax,
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
        ("rms",                "RMS Amplitude"),
        ("max_amplitude",      "Max Amplitude"),
        ("dominant_frequency", "Dominant Frequency (Hz)"),
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
    x_col = "energy_1p0_5p0hz"   # seismic detection band
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


# Plot 5: Raw Waveform Panel
def plot_raw_waveforms(df):
    eq_files    = df[df["label"] == 1].nlargest(2, "max_amplitude")["filename"].tolist()
    noise_files = df[df["label"] == 0].nsmallest(2, "max_amplitude")["filename"].tolist()

    entries = (
        [(f, EARTHQUAKE_DIR, "Earthquake", COLOR_EQ)    for f in eq_files] +
        [(f, NOISE_DIR,      "Noise",      COLOR_NOISE) for f in noise_files]
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Raw Waveforms: Earthquake vs Noise", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, (fname, directory, label, color) in zip(axes, entries):
        try:
            st   = read(os.path.join(directory, fname))
            st.merge(method=1, fill_value=0)
            tr   = st[0]
            fs   = tr.stats.sampling_rate
            t    = np.arange(len(tr.data)) / fs
            ax.plot(t, tr.data, color=color, lw=0.6)
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Amplitude (counts)", fontsize=10)
            ax.set_title(f"{label} — {fname.replace('.mseed','')}", color=color,
                         fontweight="bold", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)
        except Exception as e:
            ax.set_title(f"Could not load {fname}: {e}")

    out_path = os.path.join(VIZ_DIR, "05_raw_waveforms.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# Plot 6: IQR Outlier Scatter
def plot_iqr_scatter(df):
    q1          = df["max_amplitude"].quantile(0.25)
    q3          = df["max_amplitude"].quantile(0.75)
    iqr         = q3 - q1
    upper_fence = q3 + 3 * iqr

    eq    = df[df["label"] == 1].reset_index(drop=True)
    noise = df[df["label"] == 0].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(noise.index, noise["max_amplitude"],
               c=COLOR_NOISE, s=20, alpha=0.7, label="Noise", edgecolors="none")
    ax.scatter(eq.index + len(noise), eq["max_amplitude"],
               c=COLOR_EQ, s=20, alpha=0.7, label="Earthquake", edgecolors="none")

    ax.axhline(upper_fence, color="black", linestyle="--", lw=1.5,
               label=f"IQR fence (Q3+3×IQR = {upper_fence:.0f})")
    ax.axvline(len(noise) - 0.5, color="gray", linestyle=":", lw=1)
    ax.text(len(noise) / 2, ax.get_ylim()[1] * 0.97, "Noise",
            ha="center", color=COLOR_NOISE, fontsize=10, fontweight="bold")
    ax.text(len(noise) + len(eq) / 2, ax.get_ylim()[1] * 0.97, "Earthquake",
            ha="center", color=COLOR_EQ, fontsize=10, fontweight="bold")

    ax.set_xlabel("Sample index", fontsize=12)
    ax.set_ylabel("Max Amplitude (counts)", fontsize=12)
    ax.set_title("IQR Outlier Filter — Max Amplitude per Sample\n"
                 "(samples above dashed line were removed before training)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)

    out_path = os.path.join(VIZ_DIR, "06_iqr_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# Plot 7: PCA Explained Variance
def plot_pca_variance():
    pca_path = os.path.join(MODELS_DIR, "pca_model.pkl")
    if not os.path.exists(pca_path):
        print(f"[skip] {pca_path} not found — run 02_extract_features.py first.")
        return

    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    var_ratio = pca.explained_variance_ratio_
    cumvar    = np.cumsum(var_ratio)
    n         = len(cumvar)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(1, n + 1), var_ratio * 100, color="#4878d0", alpha=0.6,
           label="Individual component")
    ax.plot(range(1, n + 1), cumvar * 100, "r-o", ms=3, lw=1.5,
            label="Cumulative variance")

    for threshold, ls in [(50, ":"), (75, "--"), (90, "-.")]:
        idx = np.searchsorted(cumvar, threshold / 100)
        if idx < n:
            ax.axhline(threshold, color="gray", linestyle=ls, lw=1,
                       label=f"{threshold}% @ PC{idx+1}")

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance (%)", fontsize=12)
    ax.set_title("PCA Explained Variance — Raw Waveform Components",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, 105)
    fig.tight_layout()

    out_path = os.path.join(VIZ_DIR, "07_pca_variance.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# Shared save helper
def _save(fig, filename):
    path = os.path.join(VIZ_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# Classifier result plots — data provided by 04_classify.py
def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(9, 6))
    for res, color in zip(results, PALETTE):
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
                label=f"{res['name']} (AUC={res['auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Methods, Hand-crafted Features",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, "08_roc_curves.png")


def plot_confusion_gb(gb_result, y_te):
    cm = confusion_matrix(y_te, gb_result["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Noise", "Earthquake"], fontsize=12)
    ax.set_yticklabels(["Noise", "Earthquake"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Gradient Boosting — Confusion Matrix\n"
                 f"Acc={gb_result['acc']:.3f}  AUC={gb_result['auc']:.3f}",
                 fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    _save(fig, "09_confusion_gb.png")


def plot_lasso_sparsity(C_values, accuracies, n_nonzero):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.semilogx(C_values, accuracies, "b-o", lw=2, label="Val Accuracy")
    ax1.axhline(TARGET_ACC, color="red", linestyle="--", alpha=0.6,
                label=f"{TARGET_ACC:.0%} target")
    ax1.set_xlabel("C  (higher = less regularization)", fontsize=12)
    ax1.set_ylabel("Validation Accuracy", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(loc="lower left", fontsize=10)
    ax2 = ax1.twinx()
    ax2.semilogx(C_values, n_nonzero, "r--s", lw=2, label="Active features")
    ax2.set_ylabel("Active features", color="r", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.legend(loc="lower right", fontsize=10)
    fig.suptitle("LASSO: Accuracy vs Sparsity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "10_lasso_sparsity.png")


def plot_lr_coefficients(lr_model, feature_names):
    coefs   = lr_model.coef_[0]
    order   = np.argsort(np.abs(coefs))[::-1]
    names_s = [feature_names[i] for i in order]
    coefs_s = coefs[order]
    colors  = [COLOR_EQ if c > 0 else COLOR_NOISE for c in coefs_s]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names_s)), coefs_s, color=colors, edgecolor="white")
    ax.set_xticks(range(len(names_s)))
    ax.set_xticklabels(names_s, rotation=35, ha="right", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Coefficient", fontsize=11)
    ax.set_title("Logistic Regression Coefficients (L2)\n"
                 "Positive = predicts earthquake   |   Negative = predicts noise",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=[Patch(color=COLOR_EQ,    label="predicts earthquake"),
                        Patch(color=COLOR_NOISE, label="predicts noise")], fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "11_lr_coefficients.png")


def plot_accuracy_auc_bars(hc_results, pca_results):
    names = [r["name"] for r in hc_results]
    x, w  = np.arange(len(names)), 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle("Hand-crafted vs PCA Features — All Methods",
                 fontsize=13, fontweight="bold")

    for ax, metric, title, ylim in [
        (ax1, "acc", "Test Accuracy", (0, 1.05)),
        (ax2, "auc", "ROC AUC",       (0, 1.05)),
    ]:
        hc_vals  = [r[metric] for r in hc_results]
        pca_vals = [r[metric] for r in pca_results]
        ax.bar(x - w/2, hc_vals,  w, label="Hand-crafted", color="#4878d0", edgecolor="white")
        ax.bar(x + w/2, pca_vals, w, label="PCA",          color="#ee854a", edgecolor="white")
        ax.axhline(TARGET_ACC, color="red", linestyle="--", alpha=0.7,
                   label=f"{TARGET_ACC:.0%} target")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(title, fontsize=12)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    _save(fig, "12_accuracy_auc_bars.png")


def plot_lasso_survivors(lasso_model, feature_names):
    coefs        = lasso_model.coef_[0]
    active_idx   = np.where(coefs != 0)[0]
    active_names = [feature_names[i] for i in active_idx]
    active_coefs = coefs[active_idx]
    order        = np.argsort(np.abs(active_coefs))[::-1]
    names_s      = [active_names[i] for i in order]
    coefs_s      = active_coefs[order]
    colors       = [COLOR_EQ if c > 0 else COLOR_NOISE for c in coefs_s]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(names_s)), coefs_s, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=11)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient", fontsize=11)
    ax.set_title(f"LASSO Surviving Features ({len(names_s)} of {len(feature_names)})\n"
                 "Positive = predicts earthquake   |   Negative = predicts noise",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=[Patch(color=COLOR_EQ,    label="predicts earthquake"),
                        Patch(color=COLOR_NOISE, label="predicts noise")], fontsize=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "13_lasso_survivors.png")


def plot_gb_importances(gb_model, feature_names):
    importances = gb_model.feature_importances_
    order       = np.argsort(importances)[::-1]
    names_s     = [feature_names[i] for i in order]
    imp_s       = importances[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names_s)), imp_s, color="#e377c2", edgecolor="white")
    ax.set_xticks(range(len(names_s)))
    ax.set_xticklabels(names_s, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Feature Importance (mean decrease in impurity)", fontsize=11)
    ax.set_title("Gradient Boosting — Feature Importances",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "14_gb_importances.png")


def plot_cv_bars(cv_scores):
    names  = list(cv_scores.keys())
    means  = [cv_scores[n].mean() for n in names]
    stds   = [cv_scores[n].std()  for n in names]
    colors = PALETTE[:len(names)]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(names)), means, yerr=stds, color=colors,
                  edgecolor="white", capsize=5, error_kw={"lw": 2})
    ax.axhline(TARGET_ACC, color="red", linestyle="--", lw=1.5,
               label=f"{TARGET_ACC:.0%} target")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("10-Fold CV Accuracy — Hand-crafted Features\n"
                 "Error bars = ±1 standard deviation across folds",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, "15_cv_bars.png")


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

    print("Generating raw waveform panel ...")
    plot_raw_waveforms(df)

    print("Generating IQR outlier scatter ...")
    plot_iqr_scatter(df)

    print("Generating PCA explained variance ...")
    plot_pca_variance()

    # Classifier plots (08-15) — only if 04_classify.py has been run
    results_pkl = "classifier_results.pkl"
    if os.path.exists(results_pkl):
        print("\nLoading classifier results ...")
        with open(results_pkl, "rb") as f:
            res = pickle.load(f)

        hc_results  = res["hc_results"]
        pca_results = res["pca_results"]
        cv_scores   = res["cv_scores"]
        y_te        = res["y_te"]
        lr_model    = res["lr_model"]
        lasso_model = res["lasso_model"]
        gb_model    = res["gb_model"]
        C_vals      = res["C_vals"]
        lasso_accs  = res["lasso_accs"]
        lasso_nz    = res["lasso_nz"]
        hc_names    = res["hc_names"]

        gb_result = next(r for r in hc_results if r["name"] == "Gradient Boosting")

        print("Generating ROC curves ...")
        plot_roc_curves(hc_results)
        print("Generating Gradient Boosting confusion matrix ...")
        plot_confusion_gb(gb_result, y_te)
        print("Generating LASSO sparsity plot ...")
        plot_lasso_sparsity(C_vals, lasso_accs, lasso_nz)
        print("Generating LR coefficients ...")
        plot_lr_coefficients(lr_model, hc_names)
        print("Generating accuracy/AUC comparison bars ...")
        plot_accuracy_auc_bars(hc_results, pca_results)
        print("Generating LASSO survivor features ...")
        plot_lasso_survivors(lasso_model, hc_names)
        print("Generating Gradient Boosting importances ...")
        plot_gb_importances(gb_model, hc_names)
        print("Generating CV bars ...")
        plot_cv_bars(cv_scores)
    else:
        print(f"\n[skip] {results_pkl} not found — run 04_classify.py first for plots 08-15.")

    print(f"\nAll visualizations saved to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
