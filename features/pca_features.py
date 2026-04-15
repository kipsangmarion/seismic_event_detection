"""
features/pca_features.py
Method 3 (Automatic): PCA on padded/truncated raw waveform vectors.

No domain knowledge is used - PCA finds the directions of maximum variance
across all waveforms. The top components capture patterns a human might
not think to hand-engineer.

The fitted scaler and PCA model are saved to disk so Milestone 2
classifiers can transform new data consistently without refitting.
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pad_or_truncate(data, length):
    """Pad with zeros or truncate to exactly `length` samples."""
    if len(data) >= length:
        return data[:length]
    return np.pad(data, (0, length - len(data)))


def fit_pca(waveforms, waveform_length, n_components, models_dir="models"):
    """
    Fit StandardScaler + PCA on a list of raw waveform arrays and return
    the transformed feature matrix as a dict of column arrays.

    Parameters
    waveforms      : list of 1-D np.ndarray  (raw sample values)
    waveform_length: int   — fixed length to pad/truncate each waveform to
    n_components   : int   — number of PCA components to keep
    models_dir     : str   — directory where scaler.pkl and pca_model.pkl are saved

    Returns
    dict[str, np.ndarray]  — keys are "pc1", "pc2", ..., "pc{n_components}"
    """
    X = np.array([pad_or_truncate(w, waveform_length) for w in waveforms])

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca    = PCA(n_components=n_components)
    X_pca  = pca.fit_transform(X_scaled)

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print(f"[pca] Explained variance (top 5 components): "
          f"{pca.explained_variance_ratio_[:5].round(3).tolist()}")
    print(f"[pca] Total variance explained by {n_components} components: "
          f"{pca.explained_variance_ratio_.sum():.1%}")
    print(f"[pca] Models saved to {models_dir}/")

    return {f"pc{i + 1}": X_pca[:, i] for i in range(n_components)}


def transform_pca(waveforms, waveform_length, models_dir="models"):
    """
    Transform new waveforms using the already-fitted scaler and PCA.
    Used in Milestone 2 to apply consistent feature extraction to test data.

    Parameters
    waveforms      : list of 1-D np.ndarray
    waveform_length: int
    models_dir     : str

    Returns
    np.ndarray  shape (n_waveforms, n_components)
    """
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    pca_path    = os.path.join(models_dir, "pca_model.pkl")

    if not os.path.exists(scaler_path) or not os.path.exists(pca_path):
        raise FileNotFoundError(
            f"PCA models not found in {models_dir}/. Run 02_extract_features.py first."
        )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    X = np.array([pad_or_truncate(w, waveform_length) for w in waveforms])
    return pca.transform(scaler.transform(X))
