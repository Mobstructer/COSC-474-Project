"""
Music Genre Classification — Prediction Script

Usage:
    # Predict a single song from CLI flags
    python predict.py --danceability 0.72 --energy 0.91 --loudness -4.5 \
                      --speechiness 0.05 --acousticness 0.01 \
                      --instrumentalness 0.0 --liveness 0.1 \
                      --valence 0.8 --tempo 132 --duration_ms 210000

    # Predict a CSV of songs
    python predict.py --csv data/new_songs.csv
"""

import argparse
import joblib
import pandas as pd
import numpy as np

MODEL_PATH   = "models/best_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
FEATURES_PATH = "models/feature_cols.pkl"


def load_artifacts():
    pipeline     = joblib.load(MODEL_PATH)
    le           = joblib.load(ENCODER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return pipeline, le, feature_cols


def predict_single(pipeline, le, feature_cols, feature_values: dict):
    row = {f: feature_values.get(f, np.nan) for f in feature_cols}
    X = pd.DataFrame([row])
    pred_encoded = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]
    genre = le.inverse_transform([pred_encoded])[0]
    confidence = proba.max()
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(le.inverse_transform([i])[0], round(float(proba[i]), 4)) for i in top3_idx]
    return genre, confidence, top3


def predict_csv(pipeline, le, feature_cols, csv_path: str):
    df = pd.read_csv(csv_path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in CSV (will be NaN): {missing}")
    X = df[[c for c in feature_cols if c in df.columns]].reindex(columns=feature_cols)
    preds_encoded = pipeline.predict(X)
    genres = le.inverse_transform(preds_encoded)
    df["predicted_genre"] = genres
    out = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(out, index=False)
    print(f"Predictions saved to: {out}")
    print(df[["predicted_genre"]].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="Predict music genre")
    parser.add_argument("--csv", type=str, help="CSV file with multiple songs")

    # Single-song feature flags
    feature_names = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "duration_ms",
    ]
    for f in feature_names:
        parser.add_argument(f"--{f}", type=float, default=None)

    args = parser.parse_args()
    pipeline, le, feature_cols = load_artifacts()

    if args.csv:
        predict_csv(pipeline, le, feature_cols, args.csv)
    else:
        values = {f: getattr(args, f) for f in feature_names if getattr(args, f) is not None}
        if not values:
            print("[ERROR] Provide --csv or at least one feature flag.")
            return
        genre, confidence, top3 = predict_single(pipeline, le, feature_cols, values)
        print(f"\n  Predicted Genre : {genre}")
        print(f"  Confidence      : {confidence:.2%}")
        print(f"\n  Top 3 predictions:")
        for g, p in top3:
            bar = "█" * int(p * 30)
            print(f"    {g:<20} {p:.4f}  {bar}")
        print()


if __name__ == "__main__":
    main()
