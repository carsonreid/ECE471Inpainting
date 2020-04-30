import numpy as np
import math
import json
import cv2
import os
import subprocess
from decomposition import decompose
import config
import shutil
from sklearn.preprocessing import normalize


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
    # mini_inpainted_path = os.path.join(__location__,
    # config.mini_inpainted_folder, "mini-inpainted-" + image_file)
    mini_inpainted_path = os.path.join(__location__, config.mini_inpainted_folder, "mini-inpainted-" + image_file)
    print(mini_inpainted_path)
    full_image = cv2.imread(mini_inpainted_path, cv2.IMREAD_COLOR)

    print(full_image.shape)
    # 2. decompose into structure and texture images
    structure_image, texture_image = structure_texture_decompose(full_image)

    structure_path = os.path.join(__location__, config.temp_folder, "struct-" + image_file)
    # structure_image = cv2.imread(structure_path, cv2.IMREAD_COLOR)
    cv2.imwrite(structure_path, structure_image)

    # 3. compute tensors, eigenvalues, and eigenvectors of structure image
    # tensors, eigenvalues, eigenvectors = compute_tensors_eigens(structure)

    # add largest negative to all values
    # https://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues#7422584
    for i in range(3):
        minval = structure_image[..., i].min()
        maxval = structure_image[..., i].max()
        if minval != maxval:
            structure_image[..., i] -= minval
            structure_image[..., i] *= (255.0 / (maxval - minval))

    # convert structure image to grayscale
    # structure_image = structure_image.astype("uint8")
    # gray_structure_image = cv2.cvtColor(structure_image, cv2.COLOR_BGR2GRAY)

    # get the gradients x and y
    kernX = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])
    kernY = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])
    fx = cv2.filter2D(structure_image, -1, kernX)
    fy = cv2.filter2D(structure_image, -1, kernY)

    gradients = np.stack((fx, fy), axis=-1)
    # print(gradients.shape)

    G = np.array([[np.sum([[np.outer(channel, channel)] for channel in x], axis=0) for x in y] for y in gradients])
    # print(G.shape)

    # get the eigenvalues and eigenvectors
    G_x = np.squeeze(G, axis=2)
    eigvals = np.zeros((G.shape[0], G.shape[1], 2))
    eigvecs = np.zeros(G_x.shape)
    for i, y in enumerate(G):
        for j, x in enumerate(y):
            val, vec = np.linalg.eig(x[0])
            # print(val.shape, vec.shape)
            eigvals[i][j] = val
            eigvecs[i][j] = vec

    # print(eigvals.shape, eigvecs.shape)

    # 4. compute pixel priorities for all pixels in the mask
    # pixel_priorities = compute_pixel_priorities(structure, texture, mask_im)  # TODO: find what will be needed as params

    # 5. inpaint the texture image
    mask_pixel_coordinates = get_mask_pixel_coordinates(mask_im)
    for mask_x, mask_y in mask_pixel_coordinates.copy():  # copying bc the list will be modified while using it
        # (a) find the pixel with top priority that hasn't been inpainted yet
        # ^ SKIP

        if (mask_x, mask_y) in mask_pixel_coordinates:  # check the pixel hasn't been filled by another patch fill
            # find the best patch to fill from based on SSD
            ssd_list = []
            for x in range(len(texture_image)):
                for y in range(len(texture_image[0])):
                    candidate_patch = get_9_patch(x, y, texture_image)
                    patch_to_fill = get_9_patch(mask_x, mask_y, texture_image)
                    ssd = ssd_patches(candidate_patch, patch_to_fill)
                    ssd_list.append((ssd, (x, y)))
            sorted_ssd = min(ssd_list, key=lambda item: item[0])
            best_patch = sorted_ssd[0]

            # (b) texture or structure?
            lambdaNegative = 0  # some value TODO: find out what the lambdas are
            lambdaPositive = 0  # some value
            beta = 0  # some value TODO: compute beta
            if lambdaPositive - lambdaNegative < beta:
                pass

            pixels_to_copy = []  # set this to a list of tuples, ((r, g, b), (x, y)),
            # should be list of mix of structure+texture pixels

            for value, coords in pixels_to_copy:
                if all(channel == 255 for channel in mask_im[coords[1]][coords[0]]):
                    texture_image[coords[1]][coords[0]] = list(value)
                    mask_im[coords[1]][coords[0]] = np.array([0, 0, 0])
                    mask_pixel_coordinates.remove((coords[0], coords[1]))  # remove painted pixel from list of px to paint

    # 6. sum inpainted texture image and the structure image
    final_image = np.add(texture_image, structure_image)

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
