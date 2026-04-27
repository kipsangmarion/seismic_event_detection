"""
04_classify.py
Milestone 2 - Inference Methods

Loads features.csv and trains five classifiers (imported from classifiers/):
  1. Logistic Regression (L2)       - GLM baseline
  2. SVM (RBF kernel)               - robust non-linear classifier
  3. Naive Bayes (PCA-decorrelated) - Bayesian, independence assumption fixed
  4. LASSO Logistic Regression      - sparsity / automatic feature selection
  5. Gradient Boosting              - ensemble, best overall performer (~85%)

Two feature sets are compared:
  - Hand-crafted : 7 time-domain + 5 spectral features (12 total)
  - PCA          : 50 principal components from 02_extract_features.py

Outputs saved to visualizations/:
  05_roc_curves.png
  06_confusion_matrices.png
  07_lasso_sparsity.png
  08_lr_coefficients.png
  09_pca_vs_handcrafted.png
  results_summary.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, roc_curve, auc, confusion_matrix
from matplotlib.patches      import Patch

from classifiers import (
    train_logistic_regression,
    train_svm,
    train_naive_bayes,
    train_qda,
    train_lasso,
    train_gradient_boosting,
)

warnings.filterwarnings("ignore")
load_dotenv()

# Config
FEATURES_FILE = os.getenv("FEATURES_FILE", "features.csv")
VIZ_DIR       = "visualizations"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
TARGET_ACC    = 0.85

HANDCRAFTED_COLS = [
    "rms", "max_amplitude", "zero_crossing_rate",
    "kurtosis", "skewness", "std",
    "energy_0p1_1p0hz", "energy_1p0_5p0hz",
    "energy_5p0_10p0hz", "energy_10p0_20p0hz",
    "dominant_frequency",
]
PCA_COLS = [f"pc{i}" for i in range(1, 51)]

COLOR_EQ    = "#d62728"
COLOR_NOISE = "#1f77b4"
PALETTE     = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#e377c2"]


# Data helpers
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples  "
          f"(EQ={(df['label']==1).sum()}, Noise={(df['label']==0).sum()})\n")
    hc_cols  = [c for c in HANDCRAFTED_COLS if c in df.columns]
    pca_cols = [c for c in PCA_COLS if c in df.columns]
    return df[hc_cols].values, df[pca_cols].values, df["label"].values, hc_cols


def split_and_scale(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te


# Evaluation
def evaluate(model, X_te, y_te, name):
    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    y_prob = (model.predict_proba(X_te)[:, 1]
              if hasattr(model, "predict_proba")
              else model.decision_function(X_te))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc     = auc(fpr, tpr)
    marker      = " *" if acc >= TARGET_ACC else ""
    print(f"  {name:35s}  acc={acc:.3f}  AUC={roc_auc:.3f}{marker}")
    return {"name": name, "acc": acc, "auc": roc_auc,
            "fpr": fpr, "tpr": tpr, "y_pred": y_pred}


# Plots
def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for res, color in zip(results, PALETTE):
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
                label=f"{res['name']} (AUC={res['auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Hand-crafted Features", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, "05_roc_curves.png")


def plot_confusion_matrices(results, y_te):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight="bold")
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_te, res["y_pred"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Noise", "EQ"]); ax.set_yticklabels(["Noise", "EQ"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{res['name']}\nAcc={res['acc']:.3f}", fontweight="bold", fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    _save(fig, "06_confusion_matrices.png")


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
    _save(fig, "07_lasso_sparsity.png")


def plot_lr_coefficients(lr_model, feature_names):
    coefs = lr_model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
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
                 "Positive = predicts earthquake, Negative = predicts noise",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=[Patch(color=COLOR_EQ,    label="predicts earthquake"),
                        Patch(color=COLOR_NOISE, label="predicts noise")], fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "08_lr_coefficients.png")


def plot_pca_vs_handcrafted(hc_results, pca_results):
    names    = [r["name"] for r in hc_results]
    hc_accs  = [r["acc"]  for r in hc_results]
    pca_accs = [r["acc"]  for r in pca_results]
    x, w     = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, hc_accs,  w, label="Hand-crafted (12 features)",
           color="#4878d0", edgecolor="white")
    ax.bar(x + w/2, pca_accs, w, label="PCA (50 components)",
           color="#ee854a", edgecolor="white")
    ax.axhline(TARGET_ACC, color="red", linestyle="--", alpha=0.7,
               label=f"{TARGET_ACC:.0%} target")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Hand-crafted vs PCA Features — All Classifiers",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "09_pca_vs_handcrafted.png")


def _save(fig, filename):
    path = os.path.join(VIZ_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# Summary
def print_summary(hc_results, pca_results):
    print("\n" + "=" * 68)
    print(f"{'Method':<35} {'HC Acc':>7} {'HC AUC':>7} {'PCA Acc':>8} {'PCA AUC':>8}")
    print("-" * 68)
    for hc, pc in zip(hc_results, pca_results):
        flag = " *" if hc["acc"] >= TARGET_ACC else ""
        print(f"{hc['name']:<35} {hc['acc']:>7.3f} {hc['auc']:>7.3f} "
              f"{pc['acc']:>8.3f} {pc['auc']:>8.3f}{flag}")
    print("=" * 68)
    print("  * meets 85% target")

    best = max(hc_results, key=lambda r: r["acc"])
    print(f"\nBest: {best['name']}  acc={best['acc']:.3f}  AUC={best['auc']:.3f}")

    pd.DataFrame([
        {"method": hc["name"],
         "hc_acc": round(hc["acc"], 4), "hc_auc": round(hc["auc"], 4),
         "pca_acc": round(pc["acc"], 4), "pca_auc": round(pc["auc"], 4)}
        for hc, pc in zip(hc_results, pca_results)
    ]).to_csv("results_summary.csv", index=False)
    print("[saved] results_summary.csv")


# Main
def main():
    os.makedirs(VIZ_DIR, exist_ok=True)

    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(
            f"{FEATURES_FILE} not found. Run 02_extract_features.py first."
        )

    X_hc, X_pca, y, hc_names = load_data(FEATURES_FILE)

    # Hand-crafted feature set
    print("=" * 55)
    print("HAND-CRAFTED FEATURES")
    print("=" * 55)
    X_hc_tr, X_hc_te, y_tr, y_te = split_and_scale(X_hc, y)

    print("[1/5] Logistic Regression (L2) ...")
    lr = train_logistic_regression(X_hc_tr, y_tr)

    print("[2/5] SVM (RBF) — grid search ...")
    svm = train_svm(X_hc_tr, y_tr)

    print("[3/5] Naive Bayes (PCA-decorrelated) ...")
    nb     = train_naive_bayes(X_hc_tr, y_tr)
    nb_raw = train_naive_bayes.__module__  # plain NB for comparison printed below
    from classifiers.naive_bayes import train_plain, train_qda as _train_qda
    nb_plain = train_plain(X_hc_tr, y_tr)
    qda      = _train_qda(X_hc_tr, y_tr)

    print("[4/5] LASSO Logistic Regression ...")
    lasso, C_vals, lasso_accs, lasso_nz = train_lasso(X_hc_tr, y_tr, hc_names)

    print("[5/5] Gradient Boosting ...")
    gb = train_gradient_boosting(X_hc_tr, y_tr)

    print("\nTest set results  (* = meets 85% target):")
    hc_results = [
        evaluate(lr,       X_hc_te, y_te, "Logistic Regression (L2)"),
        evaluate(svm,      X_hc_te, y_te, "SVM (RBF)"),
        evaluate(nb_plain, X_hc_te, y_te, "Naive Bayes (plain)"),
        evaluate(nb,       X_hc_te, y_te, "Naive Bayes (PCA-decorrelated)"),
        evaluate(lasso,    X_hc_te, y_te, "LASSO Logistic Regression"),
        evaluate(gb,       X_hc_te, y_te, "Gradient Boosting"),
    ]
    if qda is not None:
        evaluate(qda, X_hc_te, y_te, "QDA (reference)")

    # PCA feature set
    print("\n" + "=" * 55)
    print("PCA FEATURES")
    print("=" * 55)
    X_pca_tr, X_pca_te, y_tr_p, y_te_p = split_and_scale(X_pca, y)

    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB

    lr_p = _LR(C=1.0, max_iter=1000, random_state=RANDOM_STATE).fit(X_pca_tr, y_tr_p)
    svm_p = GridSearchCV(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
                         {"C": [0.1,1,10,100], "gamma": ["scale","auto"]},
                         cv=5, scoring="accuracy", n_jobs=-1).fit(X_pca_tr, y_tr_p)
    nb_p  = GaussianNB().fit(X_pca_tr, y_tr_p)
    nb_pca_p = train_naive_bayes(X_pca_tr, y_tr_p)
    lasso_p = _LR(penalty="l1", C=0.1, solver="liblinear",
                  max_iter=1000, random_state=RANDOM_STATE).fit(X_pca_tr, y_tr_p)
    gb_p = GradientBoostingClassifier(n_estimators=300, max_depth=2,
                                       learning_rate=0.2, subsample=0.8,
                                       random_state=RANDOM_STATE).fit(X_pca_tr, y_tr_p)

    print("\nTest set results:")
    pca_results = [
        evaluate(lr_p,     X_pca_te, y_te_p, "Logistic Regression (L2)"),
        evaluate(svm_p,    X_pca_te, y_te_p, "SVM (RBF)"),
        evaluate(nb_p,     X_pca_te, y_te_p, "Naive Bayes (plain)"),
        evaluate(nb_pca_p, X_pca_te, y_te_p, "Naive Bayes (PCA-decorrelated)"),
        evaluate(lasso_p,  X_pca_te, y_te_p, "LASSO Logistic Regression"),
        evaluate(gb_p,     X_pca_te, y_te_p, "Gradient Boosting"),
    ]

    # Summary + plotsc
    print_summary(hc_results, pca_results)

    print("\nGenerating plots ...")
    # ROC and confusion matrix: 5 methods (exclude plain NB from main comparison)
    main_hc = [r for r in hc_results if r["name"] != "Naive Bayes (plain)"]
    plot_roc_curves(main_hc)
    plot_confusion_matrices(main_hc, y_te)
    plot_lasso_sparsity(C_vals, lasso_accs, lasso_nz)
    plot_lr_coefficients(lr, hc_names)
    plot_pca_vs_handcrafted(hc_results, pca_results)

    print(f"\nAll outputs saved to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
