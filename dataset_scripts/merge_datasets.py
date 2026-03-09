import pandas as pd

img=pd.read_csv("datasets/squat_image_dataset.csv")
vid=pd.read_csv("datasets/squat_video_dataset.csv")

final=pd.concat([img,vid])

final.to_csv("datasets/squat_final_dataset.csv",index=False)

print("Final dataset:",len(final))