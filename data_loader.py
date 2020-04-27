import numpy as np
import math
import cv2
import os
import glob
import json
# from skimage import data, img_as_float
# from skimage.metrics import structural_similarity as ssim
# https://stackoverflow.com/questions/4060221/how-to-reliably-open-a-file-in-the-same-directory-as-a-python-script
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import ece_inpaint

calculated_ssim = []
calculated_psnr = []
image_names = []
masked_image_names = []


def load_masked_image_names():
    return [f for f in glob.glob(__location__ + "/y/*.png") if os.path.isfile(os.path.join(__location__, f))]


def load_mask_data():
    with open(str(os.path.join(__location__, '/a.json')), 'r') as file:
        return json.load(file)


load_masked_image_names()
mask_data = load_mask_data()
for image_name in masked_image_names:
    final_image = inpaint(image_name, mask_data[image_name])
    cv2.imwrite('./inpainted/' + file_name + '-inpainted.jpg', inpainted_im)
