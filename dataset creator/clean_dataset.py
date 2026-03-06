import pandas as pd

df = pd.read_csv("pushup_video_dataset.csv")

# Remove unrealistic angles
df = df[
    (df["elbow_angle"] > 30) &
    (df["back_angle"] > 90) &
    (df["hip_angle"] > 90) &
    (df["knee_angle"] > 90)
]

df.to_csv("pushup_dataset_clean.csv", index=False)

print("Dataset cleaned successfully")
print("Remaining samples:", len(df))