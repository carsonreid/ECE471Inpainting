import numpy as np
import math
import json
import cv2
import os
import subprocess
import decomposition
import config
import shutil

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def inpaint(image_name, mask):

    # make tmp folder for intermediary images
    temp_folder_path = os.path.join(__location__, config.temp_folder)
    if not os.path.exists(temp_folder_path):
        os.mkdir(temp_folder_path)

    # create and save a mask image for the C++ code to use
    mask_im = create_mask_image(mask)
    mini_inpaint_mask_path = os.path.join(__location__, config.temp_folder, "mini-inpaint-mask-"+image_name)
    cv2.imwrite(mini_inpaint_mask_path, mask_im)

    # 1. inpaint image using Tshumperle's method
    new_file_name = mini_inpaint(image_name, 'mini-inpaint-mask-' + image_name)

    # load mini-inpainted image
    full_image = cv2.imread(new_file_name, cv2.IMREAD_COLOR)

    # 2. decompose into structure and texture images
    structure, texture = structure_texture_decompose(full_image)

    # 3. compute tensors, eigenvalues, and eigenvectors of structure image
    tensors, eigenvalues, eigenvectors = compute_tensors_eigens(structure)

    # 4. compute pixel priorities for all pixels in the mask
    pixel_priorities = compute_pixel_priorities(structure, texture, mask_im)  # TODO: find what will be needed as params

    # 5. inpaint the texture image
    num_mask_pixels = np.sum(mask_im)
    for i in range(num_mask_pixels):
        # (a) find the pixel with top priority that hasn't been inpainted yet
        top_priority_index = np.argmax(pixel_priorities)
        row = top_priority_index / len(full_image[0])
        col = top_priority_index % len(full_image[0])

        # (b) texture or structure?
        lambdaNegative = 0  # some value TODO: find out what the lambdas are
        lambdaPositive = 0  # some value
        beta = 0  # some value
        if lambdaPositive - lambdaNegative < beta:
            pass  # TODO: apply texture synthesis equation

        # (c)
        omega = 1 / 1  # TODO: should be omega/pixel, find out what omega is

    # 6. sum inpainted texture image and the structure image
    final_image = np.add(texture, structure)

    shutil.rmtree(temp_folder_path)

    return final_image


def mini_inpaint(file_name, mask_file_name):
    new_file_name = "mini-inpainted-" + file_name
    process = subprocess.Popen(["./cimg_inpaint/mini_inpaint",
                                "-i " + file_name + " -m " + mask_file_name + " -o " + new_file_name])  # TODO: check file paths
    while True:
        return_code = process.poll()
        if return_code is not None:
            break

    return new_file_name


def create_mask_image(mask_data):
    return np.array([0])


def structure_texture_decompose(image):
    return decomposition.decompose(image)


def compute_tensors_eigens(image):
    return np.array([0]), np.array([1]), np.array([2])


def compute_pixel_priorities(structure, texture, mask_im):
    return np.array([])
