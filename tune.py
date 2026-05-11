"""
Music Genre Classification — Hyperparameter Tuning

Runs GridSearchCV on Random Forest and (optionally) XGBoost,
then saves the best model over the one produced by train.py.

Usage:
    python tune.py --data data/music_genre.csv --model rf
    python tune.py --data data/music_genre.csv --model xgb
"""

import argparse
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings("ignore")

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]
TARGET_COL   = "music_genre"
DROP_COLS    = ["instance_id", "artist_name", "track_name", "obtained_date", "key", "mode"]
RANDOM_STATE = 42


def load_and_prepare(data_path):
    df = pd.read_csv(data_path)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df = df.replace("?", np.nan)
    df = df.dropna(subset=[TARGET_COL])
    feature_cols = [c for c in AUDIO_FEATURES if c in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    X = df[feature_cols]
    return X, y, le, feature_cols


def tune_rf(X_train, y_train):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth":    [10, 20, None],
        "clf__min_samples_split": [2, 5],
        "clf__max_features": ["sqrt", "log2"],
    }
    print("\n  Running GridSearchCV for Random Forest …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro",
                      n_jobs=-1, verbose=1, refit=True)
    gs.fit(X_train, y_train)
    print(f"\n  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f}")
    return gs.best_estimator_


def tune_xgb(X_train, y_train, n_classes):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[ERROR] xgboost not installed. Run: pip install xgboost")
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    param_grid = {
        "clf__n_estimators":   [200, 400],
        "clf__max_depth":      [4, 6, 8],
        "clf__learning_rate":  [0.05, 0.1],
        "clf__subsample":      [0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.9],
    }
    print("\n  Running GridSearchCV for XGBoost …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro",
                      n_jobs=-1, verbose=1, refit=True)
    gs.fit(X_train, y_train)
    print(f"\n  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f}")
    return gs.best_estimator_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=str, default="data/music_genre.csv")
    parser.add_argument("--model", type=str, choices=["rf", "xgb"], default="rf",
                        help="Which model to tune: rf (Random Forest) or xgb (XGBoost)")
    args = parser.parse_args()

    X, y, le, feature_cols = load_and_prepare(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    if args.model == "rf":
        best_pipe = tune_rf(X_train, y_train)
    else:
        best_pipe = tune_xgb(X_train, y_train, n_classes=len(le.classes_))

    if best_pipe is None:
        return

    y_pred = best_pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n  Test F1 Macro : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    import os; os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipe,    "models/best_model.pkl")
    joblib.dump(le,           "models/label_encoder.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    print("\n  Saved tuned model to models/best_model.pkl")


if __name__ == "__main__":
    main()
