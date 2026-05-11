# 🎵 Music Genre Classifier

Multiclass genre classification using Spotify-style audio features.  
Dataset: [Kaggle — Prediction of Music Genre](https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre)

---

## Project Structure

```
music-genre-classifier/
├── data/                     ← Put music_genre.csv here
├── models/                   ← Saved model artifacts (auto-created)
├── outputs/                  ← Plots and charts (auto-created)
├── train.py                  ← Full training pipeline
├── predict.py                ← Predict genre for new songs
├── tune.py                   ← Hyperparameter tuning (GridSearchCV)
├── requirements.txt
└── README.md
```

---

## Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Download `music_genre.csv` from [Kaggle](https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre)  
and place it in the `data/` folder.

### 3. Train
```bash
python train.py --data data/music_genre.csv
```

This will:
- Clean and preprocess the data
- Run EDA (saves plots to `outputs/`)
- Train Logistic Regression, Random Forest, and XGBoost
- Print cross-validation and test scores
- Save confusion matrices and feature importance plots
- Save the best model to `models/`

### 4. Predict a single song
```bash
python predict.py \
  --danceability 0.72 \
  --energy 0.91 \
  --loudness -4.5 \
  --speechiness 0.05 \
  --acousticness 0.01 \
  --instrumentalness 0.0 \
  --liveness 0.1 \
  --valence 0.8 \
  --tempo 132 \
  --duration_ms 210000
```

### 5. Predict a CSV of songs (if you have a csv of new songs)
```bash
python predict.py --csv data/new_songs.csv
```

### 6. Hyperparameter tuning (optional, slower)
```bash
python tune.py --data data/music_genre.csv --model rf   # Random Forest
python tune.py --data data/music_genre.csv --model xgb  # XGBoost
```

---

## Features Used

| Feature           | Description                                 |
|-------------------|---------------------------------------------|
| danceability      | How suitable for dancing (0–1)              |
| energy            | Perceptual measure of intensity (0–1)       |
| loudness          | Overall loudness in dB                      |
| speechiness       | Presence of spoken words (0–1)              |
| acousticness      | Confidence of acoustic quality (0–1)        |
| instrumentalness  | Predicts no vocals (0–1)                    |
| liveness          | Presence of live audience (0–1)             |
| valence           | Musical positiveness (0–1)                  |
| tempo             | Estimated BPM                               |
| duration_ms       | Track length in milliseconds                |

---

## Models & Expected Performance

| Model               | Expected Test Accuracy |
|---------------------|------------------------|
| Logistic Regression | 55–70%                 |
| Random Forest       | 70–85%                 |
| XGBoost             | 75–90%                 |

Performance varies with class balance and dataset version.

---

## Output Files

| File                                        | Description                     |
|---------------------------------------------|---------------------------------|
| `outputs/genre_distribution.png`           | Bar chart of class counts       |
| `outputs/correlation_matrix.png`           | Feature correlation heatmap     |
| `outputs/feature_distributions.png`        | Per-genre feature histograms    |
| `outputs/model_comparison.png`             | Accuracy/F1 bar chart           |
| `outputs/confusion_matrix_<Model>.png`     | Confusion matrix per model      |
| `outputs/feature_importance_<Model>.png`   | Feature importance per model    |
| `models/best_model.pkl`                    | Serialized best pipeline        |
| `models/label_encoder.pkl`                 | Genre label encoder             |
| `models/feature_cols.pkl`                  | Feature column list             |
