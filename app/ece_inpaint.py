import numpy as np
import math
import json
import cv2
import os
import subprocess
from decomposition import decompose
import config
import shutil

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def inpaint(image_file, mask):
    # TODO: Might have to remove all image_name and replace with image_file as
    # passed in the parameter
    image_name, ext = os.path.splitext(image_file)
    # make tmp folder for intermediary images
    temp_folder_path = os.path.join(__location__, config.temp_folder)
    if not os.path.exists(temp_folder_path):
        os.mkdir(temp_folder_path)

    # create and save a mask image for the C++ code to use
    mask_im = create_mask_image(image_file, mask)
    mini_inpaint_mask_path = os.path.join(__location__, config.temp_folder, "mini-inpaint-mask-" + image_file)
    cv2.imwrite(mini_inpaint_mask_path, mask_im)

    # 1. inpaint image using Tshumperle's method
    # see report for why this is commented out
    # new_image_path = os.path.join(__location__, temp_folder_path, "mini-inpainted-" + image_file)
    # mini_inpaint(os.path.join(__location__, "images", image_file),
    #                             os.path.join(__location__, temp_folder_path, 'mini-inpaint-mask-' + image_file),
    #                             new_image_path)
    # see report for thy this is commented out

    # load mini-inpainted image
    mini_inpainted_path = os.path.join(__location__, config.mini_inpainted_folder, "mini-inpainted-" + image_file)
    full_image = cv2.imread(mini_inpainted_path, cv2.IMREAD_COLOR)

    print(full_image.shape)
    # 2. decompose into structure and texture images
    structure, texture = structure_texture_decompose(full_image)

    # 3. compute tensors, eigenvalues, and eigenvectors of structure image
    # get the gradients x and y

    # get the covariance matrix

    # get the eigenvalues and eigenvectors

    # tensors, eigenvalues, eigenvectors = compute_tensors_eigens(structure)

    # 4. compute pixel priorities for all pixels in the mask
    pixel_priorities = compute_pixel_priorities(structure, texture, mask_im)  # TODO: find what will be needed as params

    # 5. inpaint the texture image
    mask_pixel_coordinates = get_mask_pixel_coordinates(mask_im)
    for mask_x, mask_y in mask_pixel_coordinates:
        # (a) find the pixel with top priority that hasn't been inpainted yet
        # ^ SKIP

        # find the best patch to fill from based on SSD
        ssd_list = []
        for x in range(len(full_image)):
            for y in range(len(full_image[0])):
                candidate_patch = get_9_patch(x, y, full_image)
                patch_to_fill = get_9_patch(mask_x, mask_y, full_image)
                ssd = ssd_patches(candidate_patch, patch_to_fill)
                ssd_list.append((ssd, (x, y)))
        sorted_ssd = min(ssd_list, key=lambda x: x[0])
        best_patch = sorted_ssd[0]

        patch_pixels = [(0, 0), (0, 1)]  # TODO: correctly generate a list of pixels in the current patch

        # (b) texture or structure?
        lambdaNegative = 0  # some value TODO: find out what the lambdas are
        lambdaPositive = 0  # some value
        beta = 0  # some value TODO: compute beta
        if lambdaPositive - lambdaNegative < beta:
            pass



        # (c)
        omega = 1 / 1  # TODO: should be omega/pixel, find out what omega is

    # 6. sum inpainted texture image and the structure image
    final_image = np.add(texture, structure)

    shutil.rmtree(temp_folder_path)

    return final_image


def mini_inpaint(file_name, mask_file_name, new_file_name):
    process = subprocess.Popen(["./cimg_inpaint/mini_inpaint",
                                "-i " + file_name, "-m " + mask_file_name, "-o " + new_file_name])  # TODO: check file paths
    while True:
        return_code = process.poll()
        if return_code is not None:
            break


def create_mask_image(image_name, mask_list):
    """
    Create a mask image with the shape of the original image and set the values
    to 1 where the removed part of the image is as provided by the mask_list.
    """
    image = cv2.imread(os.path.join(__location__, config.input_folder, image_name))
    mask_im = np.zeros(np.shape(image))

    for mask in mask_list:
        for y in range(mask[0], mask[0] + mask[2]):
            for x in range(mask[1], mask[1] + mask[2]):
                mask_im[y, x] = 255

    return mask_im


def structure_texture_decompose(image):
    return decompose(image)


def compute_tensors_eigens(image):
    return np.array([0]), np.array([1]), np.array([2])


def compute_pixel_priorities(structure, texture, mask_im):
    return np.array([])


def get_9_patch(x, y, image):
    im = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    patch = np.zeros((9, 9, 3))
    for i in range(y - 4, y + 4, 1):
        for j in range(x - 4, x + 4, 1):
            for k in range(3):
                patch[i][j][k] = im[i + 4][j + 4][k]
    return patch


# taken from https://stackoverflow.com/questions/2284611/sum-of-square-differences-ssd-in-numpy-scipy
def ssd_patches(p1, p2):
    dif = p1.ravel() - p2.ravel()
    return np.dot(dif, dif)


# takes a mask image and returns a list of coordinates included in the mask
def get_mask_pixel_coordinates(image):
    coords = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] != 0:
                coords.append((j, i))
    return coords