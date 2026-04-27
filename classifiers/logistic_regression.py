"""
classifiers/logistic_regression.py
Method 1 (GLM): L2-regularized Logistic Regression.

Models P(earthquake | features) directly as a sigmoid of a linear combination.
Interpretable via coefficients — the sign and magnitude of each coefficient
shows which features push the decision toward earthquake vs. noise.
"""

from sklearn.linear_model import LogisticRegression


def train(X_tr, y_tr, C=1.0, random_state=42):
    """
    Fit L2 logistic regression.

    Returns
      sklearn.linear_model.LogisticRegression (fitted)
    """
    model = LogisticRegression(
        penalty="l2", C=C, max_iter=1000, random_state=random_state
    )
    model.fit(X_tr, y_tr)
    return model
