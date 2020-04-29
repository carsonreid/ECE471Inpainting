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
    new_file_name = os.path.join(__location__, temp_folder_path, "mini-inpainted-" + image_file)
    mini_inpaint(os.path.join(__location__, "images", image_file),
                                 os.path.join(__location__, temp_folder_path, 'mini-inpaint-mask-' + image_file),
                                 new_file_name)

    # load mini-inpainted image
    full_image = cv2.imread(new_file_name, cv2.IMREAD_COLOR)

    print(full_image.shape)
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

        patch_pixels = [(0, 0), (0, 1)]  # TODO: correctly generate a list of pixels in the current patch

        # (b) texture or structure?
        lambdaNegative = 0  # some value TODO: find out what the lambdas are
        lambdaPositive = 0  # some value
        beta = 0  # some value TODO: compute beta
        if lambdaPositive - lambdaNegative < beta:
            pixel_vals = []
            for patch_pixel in patch_pixels:
                # u = structure, v = texture

                pass  # TODO: apply texture synthesis equation

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
