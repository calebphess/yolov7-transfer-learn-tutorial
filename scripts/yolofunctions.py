import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# function to draw an image with yolo detections
def draw_yolo_image(filename: str):
    detection_image = Image.open(filename)
    plt.imshow(np.array(detection_image))