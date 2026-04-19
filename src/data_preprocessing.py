import pandas as pd

df = pd.read_csv("C:/Users/Apple/heart_disease_prediction/data/heart_large.csv")

#dropping useless columns
if 'id' in df.columns:
    df = df.drop(columns=['id'])
if 'dataset' in df.columns:
    df = df.drop(columns=['dataset'])

#convert target
df['target'] = df['num'].apply(lambda x: 1 if x>0 else 0)
df = df.drop(columns=['num'])


#basic checking
print(df.head())

print("Shape:",df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("/nTarget Distribution: ")
print(df['target'].value_counts())

#Handling the missing values

df = df.drop(columns=['ca']) #dropping because of huge number of missing values

df = df.drop(columns=['thal']) #dropping because of huge number of missing values

df['slope'] = df['slope'].fillna(df['slope'].mode()[0]) #Replace all missing values in slope with the most frequent category

small_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
for col in small_cols:
    df[col] = df[col].fillna(df[col].median()) #Replace all missing values in slope with the median

# fill binary categorical columns
df['fbs'] = df['fbs'].fillna(df['fbs'].mode()[0])
df['exang'] = df['exang'].fillna(df['exang'].mode()[0])
df['restecg'] = df['restecg'].fillna(df['restecg'].mode()[0])

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

# ENCODING
df['sex'] = df['sex'].map({'Male':1,'Female':0}) # Label Encoding for binary categorical

df['fbs'] = df['fbs'].astype(int)
df['exang'] = df['exang'].astype(int) # fbs and exang already 0/1, ensure integer

df = pd.get_dummies(df, columns=['cp','restecg','slope'],drop_first=True) #To avoid multicollinearity

print("\nAfter Encoding: ")
print(df.head())

print(df.dtypes)

bool_cols = df.select_dtypes(include='bool').columns

for col in bool_cols:
    df[col] = df[col].astype(int)

print("\nAfter converting bool to int:")
print(df.dtypes)

# feature-target split

X = df.drop(columns=['target'])
y = df['target']

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# Save cleaned dataset
df.to_csv("data/heart_large_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully.")

