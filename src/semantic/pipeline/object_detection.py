import math

import cv2
import keras_cv
import numpy as np
from keras import Sequential
from keras.src.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.optimizers import SGD
from keras.src.saving.saving_api import load_model
from matplotlib import pyplot as plt


def show(image):
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyWindow("Object Detection")

chess_class_labels = ["accidents",
"accents",
"breve",
"breve+dot",
"semibreve",
"semibre+dot",
"minim",
"minim+dot",
"quarter",
"quarter+dot",
"quarter+1beam",
"eighth",
"quarter+1beam+dot",
"eighth + dot",
"quarter+2beam ",
"sixteenth",
"quarter+2beam+dot",
"sixteenth+dot",
"quarter+3beam ",
"Thirty-second ",
"quarter+3beam+dot",
"Thirty-second+dot",
"quarter+4beam",
"Sixty-fourth ",
"quarter+4beam+dot",
"Sixty-fourth+dot",
"double-whole",
"double-whole+dot",
"whole",
"whole+dot",
"half",
"half+dot",
"quarter-rest",
"quarter-rest+dot",
"eighth-rest",
"eighth-rest+dot",
"sixteenth-rest",
"sixteenth-rest+dot",
"Thirty-second-rest",
"Thirty-second-rest+dot",
"Sixty-fourth-rest",
"Sixty-fourth-rest+dot",
"connection-bar",
"vertical-separator",
"clef",
"dot",
"metric",
"tie"]



def normalize_image(image):
    img = image.copy()

    if len(image.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    border_v, border_h = (0, 0)
    if img.shape[0] < 512 <= img.shape[1]:
        img = img[:, :512]
        border_v = int((512 - img.shape[0]) / 2) + 1
    elif img.shape[1] < 512 <= img.shape[0]:
        img = img[:512, :]
        border_h = int((512 - img.shape[1]) / 2) + 1
    elif img.shape[0] < 512 and img.shape[1] < 512:
        border_v = int((512 - img.shape[0]) / 2) + 1
        border_h = int((512 - img.shape[1]) / 2) + 1

    img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    img = img[:512, :512]
    img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    img = cv2.resize(img, (512, 512))

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # show_image(img)
    input = (np.array([img]) / 255)
    input = input.astype(np.float32)
    return input, border_h, border_v


def plot_images_with_xywh_bounding_boxes(images,boxes,class_ids,class_labels,image_per_row=4,show_labels=True,confidences=None):
  class_colors = plt.cm.hsv(np.linspace(0, 1, len(class_labels)+1)).tolist()

  image_count=len(images)
  row_count=math.ceil(image_count/image_per_row)
  col_count=image_per_row

  _, axs = plt.subplots(nrows=row_count, ncols=col_count,figsize=(18, 4*row_count),squeeze=False)
  for r in range(row_count):
      for c in range(col_count):
        axs[r,c].axis('off')

  for i in range(image_count):
    r = i // image_per_row
    c = i % image_per_row

    axs[r,c].imshow(images[i])
    for box_idx in range(len(boxes[i])):
        box=boxes[i][box_idx]
        class_idx=class_ids[i][box_idx]
        color =class_colors[class_idx]
        xmin=box[0]
        ymin=box[1]
        w=box[2]
        h=box[3]
        axs[r,c].add_patch(plt.Rectangle((xmin,ymin), w, h, color=color, fill=False, linewidth=2))
        if show_labels:
          label ='{}'.format(class_labels[class_idx])
          if confidences is not None:
            label+=' {:.2f}'.format(confidences[i][box_idx])
          axs[r,c].text(xmin, ymin, label, size='large', color='white', bbox={'facecolor':color, 'alpha':1.0})

def plot_images_with_y_preds(images,y_preds,class_labels,image_per_row=4,show_labels=True):
  image_count=images.shape[0]
  plot_images_with_xywh_bounding_boxes(images,
                                      [y_preds['boxes'][i,:y_preds['num_detections'][i]] for i in range(image_count)],
                                      [y_preds['classes'][i,:y_preds['num_detections'][i]] for i in range(image_count)],
                                      class_labels,
                                      image_per_row=image_per_row,
                                      show_labels=show_labels,
                                      confidences=[y_preds['confidence'][i,:y_preds['num_detections'][i]] for i in range(image_count)])


class SymbolDetector:
    def __init__(self, model_path):
        self.model = get_model(model_path)

    def detect(self, image):
        input, _, _ = normalize_image(image)
        show(input[0])
        pred = self.model.predict(input)
        plot_images_with_y_preds(input, pred, chess_class_labels)
        print(pred)


