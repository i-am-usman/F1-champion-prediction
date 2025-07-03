# 🏎️ Formula 1 Champion Predictor (2018–2025)

## 📌 Objective

This project analyzes pit stop and performance data from Formula 1 seasons 2018–2024 to build predictive models for identifying the most likely World Champion in the 2025 season.

---

## 📁 Dataset

- **Source**: `f1_pitstops_2018_2024.csv`
- Records pit stop times, positions, laps, and performance metrics per driver across seasons
- **Key fields**:
  - `season`, `driver`, `position`, `laps`, `pit_time`
  - `totalpitstops`, `avgpitstoptime`

---

## 🔧 Processing Pipeline

### 1. Data Cleaning
- Standardized column names
- Converted relevant columns to numeric types
- Removed rows with missing critical data

### 2. Feature Aggregation
Aggregated statistics per driver per season:
- `avg_position`, `best_position`, `total_laps`
- `avg_pit_time`, `total_pitstops`, `avg_pitstop_time`
- Labeled champions as drivers with the lowest `avg_position` per season

### 3. Encoding
- Used `LabelEncoder` to encode driver names for modeling

---

## 📊 Visualizations

- 📈 Distribution of Pit Stop Times
- 📊 Average Laps per Driver per Season
- 🔥 Correlation Heatmap of Features
- 🌲 Feature Importance via Random Forest

All visualizations were built using Seaborn and Matplotlib.

---

## 🤖 Modeling & Evaluation

**Models Trained:**
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes

**Pipeline:**
- Scaled features using `StandardScaler`
- Used train-test split (70/30) with stratification on the target
- Evaluated models using:
  - Accuracy
  - F1 Score
  - Confusion Matrix
  - Classification Report

**Best Performing Model:**  
✅ Random Forest Classifier

---

## 🏁 Champion Prediction for 2025

Used 2024 driver stats as input to predict the 2025 champion using the trained Random Forest model. Drivers predicted with label `1` were considered potential champions.

---

## 🧪 Results Summary

**Model Accuracies:**
- Random Forest: ~[1.0]  
- Logistic Regression: ~[0.9773]  
- KNN: ~[0.9773]  
- Naive Bayes: ~[0.9773]

**F1 Scores:**
- Logistic Regression: 0.6667
- Random Forest: 1.0000
- KNN: 0.6667
- Naive Bayes: 0.6667

---

## 📂 Directory Structure

```
├── f1_champion_predictor.ipynb
├── f1_pitstops_2018_2024.csv
├── README.md
├── visualizations/
│   ├── pitstop_distribution.png
│   ├── laps_per_season.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── confusion_matrix_*.png
```

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## ✍️ Author

**Muhammad Usman**  
*F1 Enthusiast • Data Analyst*