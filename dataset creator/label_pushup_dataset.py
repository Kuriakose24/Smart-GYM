import pandas as pd

df = pd.read_csv("pushup_dataset_clean.csv")

labels = []

for _, row in df.iterrows():

    elbow = row["elbow_angle"]
    back = row["back_angle"]
    hip = row["hip_angle"]

    if (
        (60 < elbow < 110 or 150 < elbow < 180)
        and back > 140
        and hip > 140
    ):
        labels.append("correct")
    else:
        labels.append("incorrect")

df["label"] = labels

df = df.drop(columns=["exercise"])

df.to_csv("pushup_labeled_dataset.csv", index=False)

print("Dataset labeled successfully")
print("Total samples:", len(df))
print(df["label"].value_counts())