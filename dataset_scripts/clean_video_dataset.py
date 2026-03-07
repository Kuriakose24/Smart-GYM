import pandas as pd

df = pd.read_csv("pushup_video_dataset.csv")

print("Original video dataset:", len(df))


# remove impossible angles
df = df[
    (df["elbow_angle"] >= 20) & (df["elbow_angle"] <= 180) &
    (df["back_angle"] >= 20) & (df["back_angle"] <= 180) &
    (df["hip_angle"] >= 20) & (df["hip_angle"] <= 180) &
    (df["knee_angle"] >= 20) & (df["knee_angle"] <= 180)
]


# remove unrealistic posture
df = df[
    (df["back_angle"] >= 100) &
    (df["hip_angle"] >= 100)
]


print("After cleaning:", len(df))

print("\nLabel counts:")
print(df["label"].value_counts())


df.to_csv("pushup_video_dataset_clean.csv", index=False)

print("\nClean video dataset saved")