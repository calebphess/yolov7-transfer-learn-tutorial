# Draws a random preprocessed image with bounding boxes and label

import os
import random

from PIL import Image
from PIL import ImageDraw

import matplotlib.pyplot as plt
import numpy as np

DETECTION_CLASS_MAPPING = {
    "plane" : 0,
    "ship" : 1,
    "storage-tank" : 2,
    "baseball-diamond": 3,
    "tennis-court" : 4,
    "basketball-court" : 5,
    "ground-track-field" : 6,
    "harbor" : 7,
    "bridge" : 8,
    "large-vehicle" : 9,
    "small-vehicle" : 10,
    "helicopter" : 11,
    "roundabout" : 12,
    "soccer-ball-field" : 13, 
    "swimming-pool" : 14,
    "container-crane" : 15,
    "airport" : 16,
    "helipad" : 17
}

class_id_to_name_mapping = dict(zip(DETECTION_CLASS_MAPPING.values(), DETECTION_CLASS_MAPPING.keys()))

def get_label_bboxes(label_file: str):
    label_list = []

    with open(label_file, "r") as file:
        label_list = file.read().split("\n")[:-1]
        label_list = [x.split(" ") for x in label_list]
        label_list = [[float(y) for y in x ] for x in label_list]

    return label_list

def plot_bounding_box(image, label_list):
    annotations = np.array(label_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()

def main():
    label_file = "../labels/" + random.choice(os.listdir("../labels"))
    print(label_file)
    image_file = label_file.replace("labels", "images").replace(".txt", ".png")
    image = Image.open(image_file)

    labels = get_label_bboxes(label_file)

    plot_bounding_box(image, labels)




if __name__ == "__main__":
    main()