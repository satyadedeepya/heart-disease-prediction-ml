import pandas as pd
from sklearn.model_selection import train_test_split

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

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

# predictions
y_pred = lr.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))