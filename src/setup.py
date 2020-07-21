
import os

YOLO_DATA = "/content/yolov5/wheat_data"
IMAGES_DATA = YOLO_DATA + "/images"
LABELS_DATA = YOLO_DATA + "/labels"

IMAGE_TRAIN = IMAGES_DATA + "/train"
IMAGE_VAL = IMAGES_DATA + "/validation"
LABEL_TRAIN = LABELS_DATA + "/train"
LABEL_VAL = LABELS_DATA + "/validation"

if not os.path.exists(YOLO_DATA):
  os.mkdir(YOLO_DATA)

if not os.path.exists(IMAGES_DATA):
  os.mkdir(IMAGES_DATA)

if not os.path.exists(LABELS_DATA):
  os.mkdir(LABELS_DATA)

if not os.path.exists(IMAGE_TRAIN):
  os.mkdir(IMAGE_TRAIN)

if not os.path.exists(IMAGE_VAL):
  os.mkdir(IMAGE_VAL)

if not os.path.exists(LABEL_TRAIN):
  os.mkdir(LABEL_TRAIN)

if not os.path.exists(LABEL_VAL):
  os.mkdir(LABEL_VAL)