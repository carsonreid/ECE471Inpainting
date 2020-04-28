from os import walk
from os.path import isfile, join
from os import listdir
import numpy as np
import math
import cv2
import os
import glob
import json
import config

# https://stackoverflow.com/questions/4060221/how-to-reliably-open-a-file-in-the-same-directory-as-a-python-script
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from ece_inpaint import inpaint


def load_image_names_and_path():
    return [f for f in glob.glob(__location__ + '\\' + config.input_folder + '\\*.png') if os.path.isfile(os.path.join(__location__, f))]


def load_mask_data():
    with open(str(os.path.join(__location__, config.input_folder + '\\maskdata.json')), 'r') as file:
        return json.load(file)


# iterate through input folder and load image paths into array
image_paths = load_image_names_and_path()
# load the maskdata json file
mask_data = load_mask_data()

# iterate through each image path and inpaint the image using its corresponding
# mask data
for image_path in image_paths:
    image_file = os.path.basename(image_path)
    image_name, ext = os.path.splitext(image_file)
    final_image = inpaint(image_name, mask_data[image_name])
    cv2.imwrite('\\' + config.output_folder + '\\' + image_name + '-inpainted.png', final_image)
