import pandas as pd

img = pd.read_csv("pushup_image_dataset_balanced.csv")
vid = pd.read_csv("pushup_video_dataset_clean.csv")

df = pd.concat([img, vid], ignore_index=True)

print("Image samples:", len(img))
print("Video samples:", len(vid))
print("Final dataset size:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts())

df.to_csv("pushup_final_dataset.csv", index=False)

print("\nFinal dataset saved as pushup_final_dataset.csv")