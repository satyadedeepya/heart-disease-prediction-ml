import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load processed data
df = pd.read_csv("data/heart_large_cleaned.csv")

#Feature-Target split
X = df.drop(columns=['target'])
y = df['target']

print("Feature Shape: ", X.shape)
print("Target Shape: ", y.shape)

# Train-Test Split
X_train , X_test , y_train , y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y #Train and test sets have same class proportions as original data
    #Stratification ensures: Train ≈ Test ≈ Original distribution
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nTrain Distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest Distribution:")
print(y_test.value_counts(normalize=True))

# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

lr = LogisticRegression(max_iter=2000)

lr.fit(X_train_scaled, y_train)

# predictions
y_pred = lr.predict(X_test_scaled)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#threshold tuning

#get probabilities

y_probs = lr.predict_proba(X_test_scaled)[:,1]

#Apply custom threshold
threshold = 0.3
y_pred_custom = (y_probs >= threshold).astype(int)

print(f"\n--- Threshold = {threshold} ---")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom))

# testing different thresholds
thresholds = [0.5,0.4,0.3,0.2]

for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    cm = confusion_matrix(y_test,y_pred_t)

    tn, fp, fn, tp = cm.ravel()

    recall = tp/(tp+fn)

    print(f"\nThreshold: {t}")
    print(f"Recall: {recall:.2f}")
    print(f"FN: {fn}, FP: {fp}")


# RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest ---")

print("\nAccuracy:", accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

import joblib
import os

# Create model folder if not exists
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "model")

os.makedirs(model_dir, exist_ok=True)

# ======================
# SAVE FEATURE NAMES  (ADD THIS)
# ======================
feature_names = X.columns.tolist()
joblib.dump(feature_names, os.path.join(model_dir, "features.pkl"))

# ======================
# SAVE MODEL + SCALER
# ======================
joblib.dump(lr, os.path.join(model_dir, "lr_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

print("\nModel, scaler, and features saved successfully.")