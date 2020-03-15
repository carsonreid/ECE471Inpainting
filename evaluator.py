import numpy as np
import math
import cv2
import os
import glob
import json
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
# https://stackoverflow.com/questions/4060221/how-to-reliably-open-a-file-in-the-same-directory-as-a-python-script
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

calculated_ssim = []
calculated_psnr = []


def ssim_images(img_1, img_2):
    ssim(img_1, img_2, data_range=max(img_1.max(), img_2.max()) - (min(img_1.min(), img_2.min())))


def mse(x, y):
    return np.linalg.norm(x - y)


def psnr(img_1, img_2):
    mse = np.mean((img_1 - img_2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def load_base_image_names():
    return [f for f in glob.glob(__location__ + "/x/*.png") if os.path.isfile(os.path.join(__location__, f))]


def load_inpainted_image_names():
    return [f for f in glob.glob(__location__ + "/inpainted/*.png") if os.path.isfile(os.path.join(__location__, f))]


def load_image(name):
    cv2.imread(os.path.join(os.path.join(__location__, "x"), name), cv2.IMREAD_COLOR)


def load_inpainted_image(name):
    cv2.imread(os.path.join(os.path.join(__location__, "inpainted"), name), cv2.IMREAD_COLOR)


def inpaint_image(file_name):
    mask = mask_data[file_name]
    im = cv2.imread(os.path.join(os.path.join(__location__, "y"), file_name), cv2.IMREAD_COLOR)
    inpainted_im = inpaint(im, mask)
    cv2.imwrite('./inpainted/' + file_name + '-inpainted.png', inpainted_im)


image_names = load_base_image_names()
image_names.sort()
inpainted_image_names = load_inpainted_image_names()
inpainted_image_names.sort()

for i in range(len(image_names)):
    image = load_image(image_names[i])
    inpainted = load_inpainted_image(inpainted_image_names[i])
    calculated_psnr.append((image_names[i], psnr(image, inpainted)))
    calculated_ssim.append((image_names[i], ssim_images(image, inpainted)))

with open('results.json', 'w') as f:
    json.dump({'psnr': calculated_psnr, 'ssim': calculated_ssim}, f)
