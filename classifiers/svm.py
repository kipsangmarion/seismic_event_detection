"""
classifiers/svm.py
Method 2 (Robust): SVM with RBF kernel, tuned via cross-validated grid search.

Finds the maximum-margin hyperplane in a high-dimensional RBF feature space.
The RBF kernel handles the non-linear boundary between earthquake and noise
that linear classifiers cannot capture.
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train(X_tr, y_tr, random_state=42):
    """
    Fit RBF SVM via 5-fold grid search over C and gamma.

    Returns
      sklearn.model_selection.GridSearchCV (fitted, wraps the best SVC)
    """
    param_grid = {
        "C":     [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    model = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=random_state),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0,
    )
    model.fit(X_tr, y_tr)
    print(f"  Best params: {model.best_params_}  CV accuracy: {model.best_score_:.3f}")
    return model
