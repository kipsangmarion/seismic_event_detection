"""
02_extract_features.py
Loads all downloaded waveforms, runs three feature extraction methods,
and writes a single combined features.csv.

Columns in features.csv:
  filename, label,
  [7 time-domain features],
  [5 spectral features],
  [PCA_COMPONENTS pca features]

Labels: 1 = earthquake, 0 = noise

Also saves:
  models/scaler.pkl     - fitted StandardScaler (reuse in Milestone 2)
  models/pca_model.pkl  - fitted PCA model      (reuse in Milestone 2)
"""

import os
import warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from obspy import read

from features import extract_time_domain, extract_spectral, fit_pca

load_dotenv()

# Config
EARTHQUAKE_DIR  = os.getenv("EARTHQUAKE_DIR", "data/earthquake")
NOISE_DIR       = os.getenv("NOISE_DIR", "data/noise")
FEATURES_FILE   = os.getenv("FEATURES_FILE", "features.csv")
WAVEFORM_LENGTH = int(os.getenv("WAVEFORM_LENGTH", 6000))
PCA_COMPONENTS  = int(os.getenv("PCA_COMPONENTS", 50))
MODELS_DIR      = "models"


# Helpers
def load_trace(filepath):
    """
    Read a .mseed file and return a single merged Trace.
    Returns None if the file cannot be read or is empty.
    """
    try:
        st = read(filepath)
        st.merge(method=1, fill_value=0)   # merge gaps, fill with zeros
        if len(st) == 0 or len(st[0].data) == 0:
            return None
        return st[0]
    except Exception as e:
        print(f"  [warn] Could not read {os.path.basename(filepath)}: {e}")
        return None


def collect_files(directory, prefix, label):
    """Return list of (filepath, filename, label) tuples."""
    entries = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith(prefix) and fname.endswith(".mseed"):
            entries.append((os.path.join(directory, fname), fname, label))
    return entries


# Main
def main():
    # gather all file paths
    eq_files    = collect_files(EARTHQUAKE_DIR, "event_", label=1)
    noise_files = collect_files(NOISE_DIR,      "noise_", label=0)
    all_files   = eq_files + noise_files

    print(f"Found {len(eq_files)} earthquake files, {len(noise_files)} noise files.")
    print(f"Processing {len(all_files)} waveforms ...\n")

    rows      = []   # one dict per waveform
    waveforms = []   # raw arrays collected for PCA

    for i, (filepath, fname, label) in enumerate(all_files):
        trace = load_trace(filepath)
        if trace is None:
            print(f"  [skip] {fname}")
            waveforms.append(None)   # placeholder to keep index alignment
            continue

        td_feats  = extract_time_domain(trace)
        sp_feats  = extract_spectral(trace)

        row = {"filename": fname, "label": label}
        row.update(td_feats)
        row.update(sp_feats)
        rows.append(row)

        waveforms.append(trace.data.astype(float))

        if (i + 1) % 50 == 0 or (i + 1) == len(all_files):
            print(f"  [{i + 1:>3}/{len(all_files)}] time-domain + spectral done")

    # drop None placeholders and align with rows
    valid_waveforms = [w for w in waveforms if w is not None]

    print(f"\n[pca] Fitting PCA on {len(valid_waveforms)} waveforms "
          f"(length={WAVEFORM_LENGTH}, components={PCA_COMPONENTS}) ...")
    pca_cols = fit_pca(valid_waveforms, WAVEFORM_LENGTH, PCA_COMPONENTS, MODELS_DIR)

    # attach PCA columns to rows
    for i, row in enumerate(rows):
        for col_name, col_values in pca_cols.items():
            row[col_name] = col_values[i]

    df = pd.DataFrame(rows)
    df.to_csv(FEATURES_FILE, index=False)

    print(f"\nSaved {len(df)} rows x {len(df.columns)} columns → {FEATURES_FILE}")
    print(f"  Earthquake samples : {(df['label'] == 1).sum()}")
    print(f"  Noise samples      : {(df['label'] == 0).sum()}")
    print(f"  Features per row   : {len(df.columns) - 2}")  # minus filename + label


if __name__ == "__main__":
    main()
