
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("f1_pitstops_2018_2024.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)

# Convert and clean numeric columns
df["pit_time"] = pd.to_numeric(df["pit_time"], errors="coerce")
df["position"] = pd.to_numeric(df["position"], errors="coerce")
df = df.dropna(subset=["season", "driver", "position", "laps"])

# Aggregate per driver per season
agg = df.groupby(["season", "driver"]).agg({
    "position": ["mean", "min"],
    "laps": "sum",
    "pit_time": "mean",
    "totalpitstops": "sum",
    "avgpitstoptime": "mean"
}).reset_index()

agg.columns = ['season', 'driver',
               'avg_position', 'best_position',
               'total_laps', 'avg_pit_time',
               'total_pitstops', 'avg_pitstop_time']

# Label champion (best avg_position per season)
agg["champion"] = agg.groupby("season")["avg_position"].transform(lambda x: (x == x.min()).astype(int))

# Encode driver
le_driver = LabelEncoder()
agg["driver_encoded"] = le_driver.fit_transform(agg["driver"])

# Drop rows with missing values in features
features = ["avg_position", "best_position", "total_laps", "avg_pit_time", "total_pitstops", "avg_pitstop_time", "driver_encoded"]
agg = agg.dropna(subset=features)

# Visualization: Distribution of pit stop times
plt.figure(figsize=(8, 4))
sns.histplot(df["pit_time"].dropna(), bins=50, kde=True)
plt.title("Distribution of Pit Stop Times")
plt.xlabel("Pit Time (s)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Visualization: Average laps per season
plt.figure(figsize=(8, 4))
sns.barplot(data=agg, x="season", y="total_laps", estimator=np.mean)
plt.title("Average Laps Per Driver Per Season")
plt.ylabel("Average Total Laps")
plt.tight_layout()
plt.show()

# Visualization: Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(agg[features + ["champion"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Feature importance (Random Forest)
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(agg[features], agg["champion"])
importances = rf_temp.feature_importances_
feat_importances = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# Modeling
X = agg[features]
y = agg["champion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.3)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

results_clean = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results_clean[name] = {
        "accuracy": acc,
        "classification_report": report,
        "model": model
    }

# Accuracy summary
accuracy_summary = {name: round(results_clean[name]["accuracy"], 4) for name in results_clean}
best_model = max(accuracy_summary, key=accuracy_summary.get)
print("Model Accuracies:")
print(accuracy_summary)
print(f"Best Model: {best_model}")

# F1 scores and confusion matrices
f1_scores = {}
for name, result in results_clean.items():
    model = result["model"]
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    f1_scores[name] = f1

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

print("F1 Scores:")
for name, score in f1_scores.items():
    print(f"{name}: {score:.4f}")

# Predict 2025 Champion using 2024 data
df_2024 = agg[agg["season"] == 2024].copy()
X_2024 = df_2024[features]
X_2024_scaled = scaler.transform(X_2024)

rf_model = results_clean["Random Forest"]["model"]
y_pred_2025 = rf_model.predict(X_2024_scaled)
df_2024["predicted_champion_2025"] = y_pred_2025
predicted_champions = df_2024[df_2024["predicted_champion_2025"] == 1]

print("\nPredicted 2025 Champion(s):")
print(predicted_champions[["driver", "avg_position", "total_laps", "total_pitstops"]])
