"""
classifiers/lasso.py
Method 4 (Sparsity): L1-regularized (LASSO) Logistic Regression.

L1 penalty drives irrelevant feature coefficients to exactly zero, performing
automatic feature selection. The sweep over regularization strength C shows
the accuracy vs sparsity tradeoff — which features survive at high regularization
are the most discriminative.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]


def train(X_tr, y_tr, feature_names, random_state=42):
    """
    Sweep LASSO regularization strength and return the best model.

    Parameters
      X_tr, y_tr     : scaled training data and labels
      feature_names  : list[str] — used to report active features
      random_state   : int

    Returns
      model       : fitted LogisticRegression at best C
      C_values    : list of C values swept
      accuracies  : list of validation accuracies per C
      n_nonzero   : list of active feature counts per C
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=random_state, stratify=y_tr
    )

    accuracies = []
    n_nonzero  = []
    models     = []

    for C in C_GRID:
        m = LogisticRegression(
            penalty="l1", C=C, solver="liblinear",
            max_iter=1000, random_state=random_state
        )
        m.fit(X_tr, y_tr)
        accuracies.append(accuracy_score(y_val, m.predict(X_val)))
        n_nonzero.append(int(np.sum(m.coef_[0] != 0)))
        models.append(m)

    best_idx   = int(np.argmax(accuracies))
    best_model = models[best_idx]
    active     = [feature_names[i] for i, c in enumerate(best_model.coef_[0]) if c != 0]

    print(f"  Best C={C_GRID[best_idx]}  val accuracy={accuracies[best_idx]:.3f}")
    print(f"  Active features ({len(active)}): {active}")

    return best_model, C_GRID, accuracies, n_nonzero
