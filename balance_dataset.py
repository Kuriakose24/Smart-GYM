import pandas as pd

df = pd.read_csv("pushup_image_dataset_clean.csv")

correct = df[df["label"] == "correct"]
incorrect = df[df["label"] == "incorrect"]

# match counts
correct_sample = correct.sample(len(incorrect), random_state=42)

balanced_df = pd.concat([correct_sample, incorrect])

print("Balanced dataset size:", len(balanced_df))
print("\nLabel counts:")
print(balanced_df["label"].value_counts())

balanced_df.to_csv("pushup_image_dataset_balanced.csv", index=False)

print("\nBalanced dataset saved")