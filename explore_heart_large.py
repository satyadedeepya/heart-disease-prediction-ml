import pandas as pd

# Load the large dataset
df = pd.read_csv("heart_large.csv")

print("Shape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)

print("\nTarget distribution:")
print(df['num'].value_counts())

print("\nMissing values per column:")
print(df.isnull().sum())



df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

df.to_csv("heart_binary.csv", index=False)

df["num"].value_counts()
# STEP 3: Create binary target
# 0 = no disease, 1 = disease (num > 0)

df['target'] = (df['num'] > 0).astype(int)

print("\nBinary target distribution:")
print(df['target'].value_counts())

