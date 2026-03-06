import pandas as pd

# Load dataset
df = pd.read_csv("pushup_image_dataset.csv")

print("Original dataset size:", len(df))


# -----------------------------
# Remove impossible angles
# -----------------------------

df = df[
    (df["elbow_angle"] >= 20) & (df["elbow_angle"] <= 180) &
    (df["back_angle"] >= 20) & (df["back_angle"] <= 180) &
    (df["hip_angle"] >= 20) & (df["hip_angle"] <= 180) &
    (df["knee_angle"] >= 20) & (df["knee_angle"] <= 180)
]


# -----------------------------
# Remove unrealistic posture
# -----------------------------

df = df[
    (df["back_angle"] >= 100) &
    (df["hip_angle"] >= 100)
]


print("After cleaning:", len(df))


# -----------------------------
# Show label distribution
# -----------------------------

print("\nLabel counts:")
print(df["label"].value_counts())


# -----------------------------
# Save clean dataset
# -----------------------------

df.to_csv("pushup_image_dataset_clean.csv", index=False)

print("\nClean dataset saved as pushup_image_dataset_clean.csv")