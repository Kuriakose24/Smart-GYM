import pandas as pd

# load dataset
df = pd.read_csv("datasets/squat_final_dataset.csv")

print("Total Samples:", len(df))

print("\nLabel Count:")
print(df["label"].value_counts())

print("\nLabel Percentage:")
print(df["label"].value_counts(normalize=True) * 100)