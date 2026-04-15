# 18-752 Project: Seismic Event Detection
**Milestone 1: Feature Extraction**

Detects and classifies seismic events (earthquake vs. noise) from raw seismograph waveforms downloaded from the IRIS/EarthScope network in miniSEED format.

---

## Project Structure

```
seismic_event_detection/
├── data/
│   ├── earthquake/          # Raw .mseed waveforms around known earthquake events
│   └── noise/               # Raw .mseed waveforms at background noise periods
├── features/
│   ├── __init__.py
│   ├── time_domain.py       # Method 1 (Manual): RMS, kurtosis, zero-crossing rate, etc.
│   ├── spectral.py          # Method 2 (Spectral): FFT band energy + dominant frequency
│   └── pca_features.py      # Method 3 (Automatic): PCA on raw waveform vectors
├── models/
│   ├── pca_model.pkl        # Fitted PCA model (reuse in Milestone 2)
│   └── scaler.pkl           # Fitted StandardScaler (reuse in Milestone 2)
├── visualizations/
│   ├── 01_spectrogram_grid.png
│   ├── 02_pca_scatter.png
│   ├── 03_feature_histograms.png
│   └── 04_spectral_scatter.png
├── 01_download_data.py      # Phase 1: Download catalog + waveforms from EarthScope
├── 02_extract_features.py   # Phase 2: Extract all features -> features.csv
├── 03_visualize.py          # Phase 3: Generate visualization PNGs
├── features.csv             # Combined feature matrix (all methods, all waveforms)
├── earthquake_catalog.xml   # USGS earthquake catalog (QuakeML format)
├── .env                     # Local config (git-ignored)
├── .env.example             # Config template
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` - no API keys required. EarthScope data is publicly accessible.

```bash
cp .env.example .env
```

---

## Running the Pipeline

Run scripts in order:

```bash
# 1. Download ~200 earthquake + ~200 noise waveforms from EarthScope
python 01_download_data.py

# 2. Extract features + save features.csv + save PCA model 
python 02_extract_features.py

# 3. Generate visualizations in visualizations/
python 03_visualize.py
```

---

## Data

**Source:** IRIS/EarthScope FDSN web service, no account required.

**Station:** IU.ANMO (Albuquerque, NM) — Global Seismograph Network flagship station.

**Channel:** HHZ; high broadband vertical component, 100 samples/second.

| Class | Count | Description |
|---|---|---|
| Earthquake | 200 | 5-min windows around M4.0–7.0 events (2023–2024) |
| Noise | 200 | 5-min windows at random times, filtered against known events |

Noise windows are validated against the earthquake catalog (±10 min buffer) to avoid accidental contamination. An IQR-based outlier filter is applied during feature extraction as a secondary safeguard.

---

## Feature Extraction

Three methods are implemented, producing a single `features.csv` (one row per waveform):

| Method | Type | Features |
|---|---|---|
| Time-domain statistics | Manual | mean, std, RMS, max amplitude, zero-crossing rate, kurtosis, skewness |
| FFT spectral band energy | Spectral | Energy in 0.1–1, 1–5, 5–10, 10–20 Hz bands + dominant frequency |
| PCA on raw waveforms | Automatic | 50 principal components (full 300s window, 30,000 samples) |

---

## Configuration

All parameters are controlled via `.env`:

| Variable | Default | Description |
|---|---|---|
| `IRIS_CLIENT` | `EARTHSCOPE` | FDSN client name |
| `NETWORK` / `STATION` | `IU` / `ANMO` | Seismic network and station |
| `MIN_MAGNITUDE` / `MAX_MAGNITUDE` | `4.0` / `7.0` | Earthquake magnitude range |
| `EVENT_LIMIT` | `300` | Max catalog events to query |
| `WAVEFORM_PRE_SECONDS` | `60` | Seconds before event origin |
| `WAVEFORM_POST_SECONDS` | `240` | Seconds after event origin |
| `NOISE_BUFFER_SEC` | `600` | Exclusion window around known events for noise |
| `WAVEFORM_LENGTH` | `30000` | Samples used for PCA (30,000 = full 300s window) |
| `PCA_COMPONENTS` | `50` | Number of PCA components to retain |

---

## Milestone 2 (April 29)

The fitted `models/scaler.pkl` and `models/pca_model.pkl` are saved for reuse. Import `transform_pca` from `features.pca_features` to apply consistent feature extraction to new data without refitting.

Planned classifiers: Logistic Regression, Kalman filter, SVM, Naive Bayes, Random Forest.
Target: >85% accuracy on earthquake vs. noise binary classification.
