"""
classifiers/naive_bayes.py
Method 3 (Bayesian): PCA-decorrelated Gaussian Naive Bayes.

Naive Bayes assumes feature independence, which is badly violated when
hand-crafted features are correlated (e.g. rms and max_amplitude move together).
Fix: apply PCA first to produce orthogonal (uncorrelated) components, then
run GaussianNB on those. The independence assumption holds exactly in PCA space.

Also fits a plain GNB (no PCA) for comparison, and QDA (full covariance).
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def train(X_tr, y_tr, n_pca_components=None):
    """
    Fit a PCA -> GaussianNB pipeline.

    Parameters
      n_pca_components : int or None
          Number of PCA components before NB. None = keep all components
          (still decorrelates; same dimensionality, orthogonal basis).

    Returns
      sklearn.pipeline.Pipeline (fitted)
    """
    n_components = n_pca_components or X_tr.shape[1]
    model = Pipeline([
        ("pca", PCA(n_components=n_components)),
        ("nb",  GaussianNB()),
    ])
    model.fit(X_tr, y_tr)
    return model


def train_plain(X_tr, y_tr):
    """Fit plain GaussianNB (no decorrelation) for comparison."""
    model = GaussianNB()
    model.fit(X_tr, y_tr)
    return model


def train_qda(X_tr, y_tr, reg_param=0.1):
    """
    Fit QDA (full per-class covariance). reg_param stabilises the
    covariance estimate on small datasets.
    Returns None on failure.
    """
    try:
        model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
        model.fit(X_tr, y_tr)
        return model
    except Exception as e:
        print(f"  QDA failed ({e}), skipping.")
        return None
