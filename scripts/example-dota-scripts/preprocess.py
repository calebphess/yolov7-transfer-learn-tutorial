## Script to convert DOTA 2.0 labels into YOLO labels
## AUTHOR: Penn Hess

import random
import os
import shutil

from PIL import Image
from sklearn.model_selection import train_test_split


# prevent a PIL DOS error from being thrown
Image.MAX_IMAGE_PIXELS = 933120000

DOTA_LABEL_DIR = "../labels"
DOTA_IMAGE_DIR = "../images"
DOTA_FORMATTED_IMAGE_DIR = "../yolo_formatted_images"
DOTA_FORMATTED_LABEL_DIR = "../yolo_formatted_labels"

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

# creates a list of detection dicts from a given text file relative path
def read_dota_label_file(label_file_name:str) -> list:
    # list of detection dicts
    detections = []

    # open the file
    with open(label_file_name) as dota_label_file:
        # read each line
        for line in dota_label_file:

            detection = {}
            
            # get a list of values by spliting on white space
            values = line.split()

            # drop any data that isn't strictly bounding boxes
            if len(values) != 10:
                continue
            
            # set each detection dict field
            detection['x1'] = values[0]
            detection['y1'] = values[1]
            detection['x2'] = values[2]
            detection['y2'] = values[3]
            detection['x3'] = values[4]
            detection['y3'] = values[5]
            detection['x4'] = values[6]
            detection['y4'] = values[7]
            detection['class'] = values[8]
            # True or Flase if 1 or 0 respectively
            detection['difficult'] = (values[9] == '1')
            
            # add the detection to the list
            detections.append(detection)
            
    # return the list of detection dicts
    return detections

# validates all classes,
#   lists each class and counts of each class
def validate_classes(label_files: list) -> None:
    # create a list of classes
    classes = {}
    
    # loop through each file
    for label_file in label_files:
        
        # get the detections from the file
        detections = read_dota_label_file(label_file)
        
        # for each detection in the file
        for detection in detections:
            # if the class hasn't been seen before,
            #   add it and set it to one
            if detection["class"] not in DETECTION_CLASS_MAPPING:
                print(f"ERROR MISSING CLASS: {detection['class']}")
            
            if detection['class'] not in classes:
                classes[detection['class']] = 1
            # otherwise increment the count
            else:
                classes[detection['class']] += 1
        
        """
        if(detection['difficult']):
            print(f"Difficult Detection Found! {detection}")
        """
                
    # print all the classes and counts
    for class_name in classes:
        print(f"{class_name} : {classes[class_name]}")
        
    print()
    print(f"Total Classes: {len(classes)}")
    
# reads an image file and gets the dimensions of the image
#   returns the tuple (width, height)
def get_image_dimensions(image_file_name: str) -> tuple:
    image = Image.open(image_file_name)
    return image.size
    
    
# formats a list of detections to yolov5 standards
#   this means 5 values in the following order
#    class(as int) x1 y1 x2 y2
def format_detections(detections: list, image_height: int, image_width: int) -> list:
    # list to hold formatted detections
    formatted_detections = []
    
    # loop through each detection
    for detection in detections:
        formatted_detection = {}
        
        # set class value
        formatted_detection['class'] = DETECTION_CLASS_MAPPING[detection['class']]
        
        # find min and max x
        max_x = float(detection['x1'])
        min_x = float(detection['x1'])
        
        # for x2-x4 check if it's the max or min
        for i in range(2,5):
            x_val = f"x{i}"
            if float(detection[x_val]) > max_x:
                max_x = float(detection[x_val])
            elif float(detection[x_val]) < min_x:
                min_x = float(detection[x_val])
        
        # find min and max y
        max_y = float(detection['y1'])
        min_y = float(detection['y1'])
        
        # for x2-x4 check if it's the max or min
        for i in range(2,5):
            y_val = f"y{i}"
            if float(detection[y_val]) > max_y:
                max_y = float(detection[y_val])
            elif float(detection[y_val]) < min_y:
                min_y = float(detection[y_val])
                
        # convert to percentages find the center, height, and with of the bbox      
        percent_min_x = min_x / image_width
        percent_max_x = max_x / image_width
        percent_min_y = min_y / image_height
        percent_max_y = max_y / image_height
        
        formatted_detection['center_x'] = (percent_min_x + percent_max_x) / 2
        formatted_detection['center_y'] = (percent_min_y + percent_max_y) / 2
        formatted_detection['width'] = percent_max_x - percent_min_x
        formatted_detection['height'] = percent_max_y - percent_min_y
        
        # append detection to list of detections
        formatted_detections.append(formatted_detection)
        
    # return the list
    return formatted_detections

# saves yolo formatted detections to a file
def save_detections(file_name: str, detections: list) -> None:
    with open(file_name, 'a') as output_file:
        for detection in detections:
            print(f"{detection['class']} {detection['center_x']} {detection['center_y']} {detection['height']} {detection['width']}", file=output_file)
            
# Utility function to move images 
def copy_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False
            
def main():

    # Check whether the specified yolo label path exists or not
    new_labels_exist = os.path.exists(DOTA_FORMATTED_LABEL_DIR)

    # make remake split directories for train/test/validation data
    directories = [
        DOTA_FORMATTED_IMAGE_DIR,
        DOTA_FORMATTED_LABEL_DIR,
        DOTA_FORMATTED_IMAGE_DIR + '/train/',
        DOTA_FORMATTED_IMAGE_DIR + '/test/',
        DOTA_FORMATTED_IMAGE_DIR + '/validation/',
        DOTA_FORMATTED_LABEL_DIR + '/train/',
        DOTA_FORMATTED_LABEL_DIR + '/test/',
        DOTA_FORMATTED_LABEL_DIR + '/validation/'
    ]

    for directory in directories:
        try:
            os.mkdir(directory)
        except:
            print(f"{directory} exists... purging and remaking it")
            shutil.rmtree(directory)
            os.mkdir(directory)
    
    # get a list of file names from the label dir that end with .txt,
    #   and add the label dir to the beginning of the filename
    label_files = [os.path.join(DOTA_LABEL_DIR, file_name) for file_name in os.listdir(DOTA_LABEL_DIR) if file_name[-4:] == ".txt"]
    
    # validate_classes(label_files)
    
    print(f"Writing {len(label_files)} label files...")
    count = 0
    
    # loop through each file
    for label_file in label_files:
        count += 1
    
        # get the detections from the file
        detections = read_dota_label_file(label_file)
        
        # get the dimenesions from the corresponding image file
        image_file = label_file.replace(DOTA_LABEL_DIR, DOTA_IMAGE_DIR).replace('.txt', '.png')
        image_width, image_height = get_image_dimensions(image_file)
    
        # print(detections)
    
        # format the detections into yolo format
        formatted_detections = format_detections(detections, image_height, image_width)
        
        # write the detections to the formatted label file
        formatted_label_file = label_file.replace(DOTA_LABEL_DIR, DOTA_FORMATTED_LABEL_DIR)
        save_detections(formatted_label_file, formatted_detections)
        
        # print status update
        if count % 100 == 0:
            print(f"Progress: {count}/{len(label_files)}")
            
    print("Completed processing labels, now creating train/test/validation data...")
    
    # Read images and annotations
    images = [os.path.join(DOTA_IMAGE_DIR, x) for x in os.listdir(DOTA_IMAGE_DIR) if x[-3:] == "png"]
    labels = [os.path.join(DOTA_LABEL_DIR, x) for x in os.listdir(DOTA_LABEL_DIR) if x[-3:] == "txt"]

    images.sort()
    labels.sort()

    # Split the dataset into train-valid-test splits 
    train_images, validation_images, train_labels, validation_labels = train_test_split(images, labels, test_size = 0.2, random_state = 1)
    validation_images, test_images, validation_labels, test_labels = train_test_split(validation_images, validation_labels, test_size = 0.5, random_state = 1)

    # Copy the splits into their folders
    copy_files_to_folder(train_images, DOTA_FORMATTED_IMAGE_DIR + '/train/')
    copy_files_to_folder(validation_images, DOTA_FORMATTED_IMAGE_DIR + '/validation/')
    copy_files_to_folder(test_images, DOTA_FORMATTED_IMAGE_DIR + '/test/')
    copy_files_to_folder(train_labels, DOTA_FORMATTED_LABEL_DIR + '/train/')
    copy_files_to_folder(validation_labels, DOTA_FORMATTED_LABEL_DIR + '/validation/')
    copy_files_to_folder(test_labels, DOTA_FORMATTED_LABEL_DIR  + '/test/')
    
if __name__ == "__main__":
    main()