import pandas as pd

# Load dataset
df = pd.read_csv("datasets/squat_final_dataset.csv")

print("Original Distribution:")
print(df["label"].value_counts())


# Separate classes
correct = df[df["label"] == "correct"]
incorrect = df[df["label"] == "incorrect"]


# Downsample incorrect class
incorrect_balanced = incorrect.sample(len(correct), random_state=42)


# Combine
balanced_df = pd.concat([correct, incorrect_balanced])


# Shuffle dataset
balanced_df = balanced_df.sample(frac=1, random_state=42)


# Save dataset
balanced_df.to_csv("datasets/squat_balanced_dataset.csv", index=False)


print("\nBalanced Dataset:")
print(balanced_df["label"].value_counts())

print("\nTotal Samples:", len(balanced_df))