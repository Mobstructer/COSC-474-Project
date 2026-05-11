"""
Music Genre Classification — Training Pipeline
Dataset: https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre

Usage:
    python train.py --data data/music_genre.csv
"""

import argparse
import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost not installed. Run: pip install xgboost")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]
TARGET_COL = "music_genre"
DROP_COLS = ["instance_id", "artist_name", "track_name", "obtained_date", "key", "mode"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────
# 1. LOAD & INSPECT
# ─────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  LOADING DATA")
    print(f"{'='*55}")
    df = pd.read_csv(path)
    print(f"  Shape      : {df.shape}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Genres     : {df[TARGET_COL].unique() if TARGET_COL in df.columns else 'N/A'}")
    print(f"  Null counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ─────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  CLEANING DATA")
    print(f"{'='*55}")

    # Drop irrelevant columns (ignore if absent)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped columns : {cols_to_drop}")

    # Drop rows where target is missing
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    print(f"  Rows dropped (no label) : {before - len(df)}")

    # Replace '?' with NaN (common in UCI-style datasets)
    df = df.replace("?", np.nan)

    # Convert feature columns to numeric
    feature_cols = [c for c in AUDIO_FEATURES if c in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Final shape : {df.shape}")
    return df


# ─────────────────────────────────────────
# 3. EDA PLOTS
# ─────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("  EXPLORATORY DATA ANALYSIS")
    print(f"{'='*55}")

    feature_cols = [c for c in AUDIO_FEATURES if c in df.columns]

    # --- Genre distribution ---
    fig, ax = plt.subplots(figsize=(12, 5))
    counts = df[TARGET_COL].value_counts()
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("viridis", len(counts)))
    ax.set_title("Genre Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(val), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/genre_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/genre_distribution.png")

    # --- Correlation heatmap ---
    if len(feature_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df[feature_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png", dpi=150)
        plt.close()
        print(f"  Saved: {OUTPUT_DIR}/correlation_matrix.png")

    # --- Feature distributions by genre ---
    key_features = [f for f in ["energy", "danceability", "acousticness", "tempo"] if f in feature_cols]
    if key_features:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, feat in enumerate(key_features):
            top_genres = df[TARGET_COL].value_counts().head(6).index
            subset = df[df[TARGET_COL].isin(top_genres)]
            for genre in top_genres:
                vals = subset[subset[TARGET_COL] == genre][feat].dropna()
                axes[i].hist(vals, bins=30, alpha=0.5, label=genre, density=True)
            axes[i].set_title(f"{feat.capitalize()} by Genre", fontweight="bold")
            axes[i].set_xlabel(feat)
            axes[i].legend(fontsize=7)
        plt.suptitle("Feature Distributions by Genre", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_distributions.png", dpi=150)
        plt.close()
        print(f"  Saved: {OUTPUT_DIR}/feature_distributions.png")


# ─────────────────────────────────────────
# 4. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("  PREPROCESSING")
    print(f"{'='*55}")

    feature_cols = [c for c in AUDIO_FEATURES if c in df.columns]
    print(f"  Features used : {feature_cols}")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  Classes       : {list(le.classes_)}")

    return X, y_encoded, le, feature_cols


# ─────────────────────────────────────────
# 5. BUILD MODELS
# ─────────────────────────────────────────
def build_pipelines(n_classes: int):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    models = {}

    # Logistic Regression (baseline)
    models["Logistic Regression"] = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                   solver="lbfgs", C=1.0)),
    ])

    # Random Forest
    models["Random Forest"] = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])

    # XGBoost
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", imputer),
            ("scaler", scaler),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )),
        ])

    return models


# ─────────────────────────────────────────
# 6. TRAIN & EVALUATE
# ─────────────────────────────────────────
def train_and_evaluate(models, X_train, X_test, y_train, y_test, le, feature_cols):
    print(f"\n{'='*55}")
    print("  TRAINING & EVALUATION")
    print(f"{'='*55}")

    results = {}
    best_model_name = None
    best_f1 = 0

    for name, pipe in models.items():
        print(f"\n  ── {name} ──")

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring="f1_macro", n_jobs=1)
        print(f"  CV F1 (macro) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full training set
        pipe.fit(X_train, y_train)

        # Test evaluation
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  Test F1 Macro : {f1:.4f}")

        results[name] = {
            "pipeline": pipe,
            "accuracy": acc,
            "f1": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\n  ★ Best model  : {best_model_name} (F1={best_f1:.4f})")
    return results, best_model_name


# ─────────────────────────────────────────
# 7. PLOTS — confusion matrix & feature importance
# ─────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, classes, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=15, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_feature_importance(pipeline, feature_cols, model_name):
    clf = pipeline.named_steps["clf"]
    importances = None

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).mean(axis=0)

    if importances is None:
        return

    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("viridis", len(fi))
    fi.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/feature_importance_{model_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_model_comparison(results):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    cv_means = [results[n]["cv_mean"] for n in names]

    x = np.arange(len(names))
    width = 0.28
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, accs, width, label="Test Accuracy", color="#4C72B0")
    ax.bar(x, f1s, width, label="Test F1 (macro)", color="#55A868")
    ax.bar(x + width, cv_means, width, label="CV F1 (macro)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=15, fontweight="bold")
    ax.legend()
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/model_comparison.png")


# ─────────────────────────────────────────
# 8. SAVE BEST MODEL
# ─────────────────────────────────────────
def save_artifacts(pipeline, le, feature_cols):
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/best_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    print("\n  Saved models/best_model.pkl")
    print("  Saved models/label_encoder.pkl")
    print("  Saved models/feature_cols.pkl")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main(data_path: str):
    df = load_data(data_path)
    df = clean_data(df)
    run_eda(df)

    X, y, le, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    models = build_pipelines(n_classes=len(le.classes_))
    results, best_name = train_and_evaluate(models, X_train, X_test, y_train, y_test, le, feature_cols)

    # Plots for all models
    print(f"\n{'='*55}")
    print("  GENERATING PLOTS")
    print(f"{'='*55}")
    plot_model_comparison(results)
    for name, r in results.items():
        plot_confusion_matrix(y_test, r["y_pred"], le.classes_, name)
        plot_feature_importance(r["pipeline"], feature_cols, name)

    # Full classification report for best model
    print(f"\n{'='*55}")
    print(f"  CLASSIFICATION REPORT — {best_name}")
    print(f"{'='*55}")
    print(classification_report(y_test, results[best_name]["y_pred"],
                                 target_names=le.classes_))

    save_artifacts(results[best_name]["pipeline"], le, feature_cols)
    print(f"\n{'='*55}")
    print("  DONE ✓")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train music genre classifier")
    parser.add_argument("--data", type=str, default="data/music_genre.csv",
                        help="Path to the dataset CSV")
    args = parser.parse_args()
    main(args.data)
