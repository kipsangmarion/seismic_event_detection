"""
classifiers/gradient_boosting.py
Method 5 (Ensemble): Gradient Boosting.

Builds an additive ensemble of shallow decision trees, each correcting the
residual errors of the previous. Depth-2 trees keep individual learners weak
(high bias, low variance), while boosting over 300 rounds reduces bias without
overfitting. Subsample=0.8 adds stochastic regularization.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def train(X_tr, y_tr, random_state=42):
    """
    Fit a tuned Gradient Boosting classifier.

    Returns
      sklearn.ensemble.GradientBoostingClassifier (fitted)
    """
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.2,
        subsample=0.8,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    print(f"  CV on train split: mean={cv_scores.mean():.3f}  min={cv_scores.min():.3f}")

    return model
