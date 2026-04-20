import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

# for loading data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "data", "heart_large_cleaned.csv")

df = pd.read_csv(file_path)

X = df.drop(columns=['target'])
y = df['target']

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Feature importance
importances = rf.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10))

# Combine importance of cp features
cp_importance = feature_importance[
    feature_importance['Feature'].str.contains('cp_')
]['Importance'].sum()

print("\nTotal importance of cp features:", cp_importance)

from sklearn.inspection import permutation_importance

# Compute permutation importance
perm_importance = permutation_importance(
    rf, X, y,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

print("\nPermutation Importance (Top 10):")
print(perm_df.head(10))