import pandas as pd


df = pd.read_csv("datasets/squat_image_dataset.csv")

correct = df[df["label"] == "correct"]
incorrect = df[df["label"] == "incorrect"]

min_count = min(len(correct), len(incorrect))

correct_sample = correct.sample(min_count, random_state=42)
incorrect_sample = incorrect.sample(min_count, random_state=42)

balanced_df = pd.concat([correct_sample, incorrect_sample])

balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

balanced_df.to_csv("datasets/squat_image_dataset_balanced.csv", index=False)

print("Balanced dataset created")
print(balanced_df["label"].value_counts())