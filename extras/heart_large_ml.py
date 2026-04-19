import pandas as pd
df = pd.read_csv("heart_large.csv")
print(df.shape) #920,16

df["target"] = (df["num"] > 0).astype(int)
print(df["target"].value_counts())