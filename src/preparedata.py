
import kaggle
import pandas as pd
import os
import ast
from tqdm import tqdm
import numpy as np
import shutil
from sklearn import model_selection
import zipfile


DOWNLOAD_PATH = "/content"
INPUT_PATH = "/content/input"

YOLO_DATA = "/content/yolov5/wheat_data"
IMAGES_DATA = YOLO_DATA + "/images"
LABELS_DATA = YOLO_DATA + "/labels"

def process_data(data,data_type="train"):
  for _,row in tqdm(data.iterrows(),total=len(data)):
    image_id = row[0]
    bboxes = row[1]
    yolo_data=[]
    for bbox in bboxes:
      x,y,w,h = bbox 
      x_center = (x + w)/2
      y_center = (y + h)/2

      x_center, y_center, w, h = x_center/1024, y_center/1024, w/1024, h/1024
      yolo_data.append([0,x_center,y_center,w,h])

    yolo_data = np.array(yolo_data)
    np.savetxt(
        os.path.join(YOLO_DATA,"labels/{}/{}.txt".format(data_type,image_id)),
        yolo_data,
        fmt = ["%d", "%f", "%f", "%f", "%f"]
    )
    shutil.copyfile(
        os.path.join(INPUT_PATH,"train/{}.jpg".format(image_id)),
        os.path.join(IMAGES_DATA,"{}/{}.jpg".format(data_type,image_id)),
    )


kaggle.api.authenticate()
kaggle.api.competition_download_files("global-wheat-detection", path=DOWNLOAD_PATH, quiet=False)

with zipfile.ZipFile(DOWNLOAD_PATH + "/global-wheat-detection.zip", 'r') as zip_ref:
    zip_ref.extractall(INPUT_PATH)


if __name__ == "__main__":
  df = pd.read_csv(os.path.join(INPUT_PATH,"train.csv"))
  df.bbox = df.bbox.apply(ast.literal_eval)
  df = df.groupby("image_id")["bbox"].apply(list).reset_index(name="bboxes")
  print(df.head(10))

  df_train, df_valid = model_selection.train_test_split(
      df,
      test_size=0.1,
      shuffle=True,
      random_state = 42
  )

  df_train = df_train.reset_index(drop=True)
  df_valid = df_valid.reset_index(drop=True)

  process_data(df_train, "train")
  process_data(df_valid, "validation")