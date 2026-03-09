import pandas as pd

df = pd.read_csv("datasets/squat_balanced_dataset.csv")

print("Before Cleaning:", len(df))


# Remove impossible angles
df = df[
    (df["knee_angle"] > 40) & (df["knee_angle"] < 180) &
    (df["hip_angle"] > 40) & (df["hip_angle"] < 180) &
    (df["back_angle"] > 40) & (df["back_angle"] < 180)
]


print("After Cleaning:", len(df))

print("\nLabel Distribution:")
print(df["label"].value_counts())


df.to_csv("datasets/squat_clean_dataset.csv", index=False)

print("\nClean dataset saved")