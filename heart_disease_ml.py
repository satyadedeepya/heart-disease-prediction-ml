import pandas as pd
df=pd.read_csv("heart_disease.csv")

X=df.drop("target",axis=1) #input features
y=df["target"] #output label

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model_lr= LogisticRegression(max_iter=1000)

model_lr.fit(X_train,y_train)

y_pred_lr=model_lr.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

print("Accuracy: ",accuracy_score(y_test,y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred_lr))

# ONE-SENTENCE GOLD RULE (MEMORIZE)

# In medical prediction, false negatives are more dangerous than false positives, so confusion matrix is more important than accuracy.

# If you say this in an interview, you score points.

# TP (True Positive) → Sick patient correctly predicted as sick

# TN (True Negative) → Healthy patient correctly predicted as healthy

# FP (False Positive) → Healthy patient wrongly predicted as sick

# FN (False Negative) → Sick patient wrongly predicted as healthy (dangerous)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train_scaled,y_train)

y_pred_knn=knn.predict(X_test_scaled)


from sklearn.metrics import accuracy_score, confusion_matrix

print("KNN Accuracy: ",accuracy_score(y_test,y_pred_knn))
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test,y_pred_knn,labels=[0,1]))

# On a small medical dataset, Logistic Regression showed conservative behavior with higher false negatives, while KNN, after feature scaling, correctly identified disease cases by leveraging local neighborhood patterns. This highlights the importance of model choice and preprocessing in medical prediction tasks.

print(df.shape)


from sklearn.metrics import precision_score, recall_score

print("LR Recalll: ",recall_score(y_test,y_pred_lr)) #sick people LR caught
print("LR Precision: ",precision_score(y_test,y_pred_lr)) #sickk ppl were actually sick

print("KNN Recall:", recall_score(y_test, y_pred_knn))
print("KNN Precision:", precision_score(y_test, y_pred_knn))


