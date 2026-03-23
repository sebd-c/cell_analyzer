# auxiliary functions module

# Code destined to storing auxiliary
# functions to main module.
# this module was created by Angelo Luiz Angonezi, angeloangonezi2@gmail.com

######################################################################
# imports

# importing required libraries
from copy import copy
from os import listdir
from os import makedirs
from os.path import join
from sys import stdout
from numpy import ndarray
from cv2 import moments
from cv2 import boundingRect
from cv2 import minAreaRect
from cv2 import FONT_HERSHEY_COMPLEX
from cv2 import putText
from cv2 import LINE_8
from cv2 import imwrite
from cv2 import drawContours
from cv2 import copyMakeBorder
from cv2 import BORDER_CONSTANT
from math import sqrt
from cv2 import fitEllipse
from cv2 import arcLength
from cv2 import rotate
from cv2 import ROTATE_90_CLOCKWISE
from cv2 import BORDER_CONSTANT
from numpy import pad
from math import pi
from pandas import DataFrame
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage.feature import local_binary_pattern
from os.path import join
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from numpy import unique
from numpy import argwhere
from numpy import min as arr_min
from numpy import max as arr_max
from numpy import mean as arr_mean
from numpy import median as arr_median
from numpy import sum as arr_sum
from numpy import full
import cv2
import numpy as np
import pandas as pd
# from typing import List, Tuple

from shapely import Polygon
from shapely import maximum_inscribed_circle
######################################################################
# defining auxiliary functions





def get_unique_ids(mask: ndarray) -> list:
    """
    Given an image of segmentation masks, returns
    a list containing all object ids.
    """
    # getting current mask unique values
    object_ids = unique(mask)

    # converting ids to list
    object_ids = object_ids.tolist()

    if 0 in object_ids:
        # removing zero from list (background pixel)
        object_ids.remove(0)

    # sorting list
    object_ids = sorted(object_ids)

    # returning object ids list
    return object_ids


def filter_mask(mask: ndarray,
                object_id: int
                ) -> ndarray:
    """
    Given a mask with multiple ids,
    returns filtered mask containing
    only given id.
    """
    # defining placeholder for filtered mask
    filtered_mask = copy(mask)

    # removing pixels different from given id
    filtered_mask[mask != object_id] = 0

    # returning filtered mask
    return filtered_mask


def mask_image(image: ndarray,
               mask: ndarray,
               reverse: bool
               ) -> ndarray:
    """
    Given an image and a binary mask, applies
    mask to image, where pixels 1 in mask will
    be kept and pixels 0 discarded.
    If reverse is set to True, does the opposite.
    """
    # defining placeholder for masked image
    masked_image = copy(image)

    # checking reverse bool
    if reverse:

        # applying mask to image
        masked_image[mask != 0] = 0  # whatever is inside the mask is removed

    else:

        # applying mask to image
        masked_image[mask == 0] = 0  # whatever is inside the mask is kept

    # returning masked image
    return masked_image


def resize_labels(mask: ndarray,
                  distance: int = 1
                  ) -> ndarray:
    """
    GPU optimized version of 'expand_labels'
    function from scikit-image. Works both
    for dilation and erosion. If distance
    is positive, dilates, else, erodes.
    """
    # defining placeholder value for dilated mask
    dilated_mask = copy(mask)

    # getting distances sign bool
    distance_is_positive = distance > 0

    # getting absolute distance value (avoids issues with range in case of negative distances)
    distance_abs = abs(distance)

    # defining distance iterator
    distances = range(distance_abs)

    # getting current mask unique object ids
    object_ids = get_unique_ids(mask=mask)

    # iterating over distances
    for _ in distances:

        # iterating over object ids
        for object_id in object_ids:

            # getting current object mask
            current_mask = filter_mask(mask=dilated_mask,
                                       object_id=object_id)

            # getting current object undilated mask
            undilated_mask = (current_mask > 0)

            # checking if distance is positive
            if distance_is_positive:

                # dilating current mask
                current_mask = binary_dilation(input=current_mask)

                # getting masks difference
                masks_difference = (current_mask != undilated_mask)

                # getting current background pixels
                current_background = (dilated_mask == 0)

                # updating dilated mask (only where there are no masks yet) - prevents overlaps!
                dilated_mask[masks_difference & current_background] = object_id

            else:

                # eroding current mask
                current_mask = binary_erosion(input=current_mask)

                # getting masks difference
                masks_difference = (current_mask != undilated_mask)

                # updating dilated mask - enables touching borders erosion!
                dilated_mask[masks_difference] = 0

    # returning dilated mask
    return dilated_mask


def get_boundaries_mask(mask: ndarray) -> ndarray:
    """
    GPU optimized version of 'find_boundaries'
    function from scikit-image, when run
    with mode 'inner'.
    """
    # getting eroded mask
    eroded_mask = resize_labels(mask=mask,
                                distance=-1)

    # applying eroded mask to original mask (3D boolean)
    boundaries_mask = mask_image(image=mask,
                                 mask=eroded_mask,
                                 reverse=True)

    # returning boundaries mask
    return boundaries_mask


def get_coords(mask: ndarray
               ) -> ndarray:
    """
    Given a mask, returns coords
    matching given object id.
    """
    # getting current object pixel coords
    coords = argwhere(mask == 1)

    # transposing coords
    transposed_coords = coords.transpose()

    # returning transposed coords
    return transposed_coords


def get_round_mean(values: ndarray) -> int:
    """
    Given an array, returns
    round mean value.
    """
    # getting array mean
    mean_value = arr_mean(values)

    # extracting value from array
    mean_value = mean_value.item()

    # rounding mean value
    mean_value = round(mean_value)

    # returning round mean value
    return mean_value


def get_centroid(coords: ndarray) -> tuple:
    """
    Given an object pixel coords,
    returns its centroid.
    """
    # extracting coords arrays
    zs, ys, xs = coords

    # getting mean coord for each axis
    z_mean = get_round_mean(values=zs)
    y_mean = get_round_mean(values=ys)
    x_mean = get_round_mean(values=xs)

    # assembling final tuple
    object_centroid = (z_mean, y_mean, x_mean)

    # returning final tuple
    return object_centroid


def get_boundaries(coords: ndarray) -> tuple:
    """
    Given an object pixel coords,
    returns its boundaries, in structure:
    (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    # extracting coords arrays
    zs, ys, xs = coords

    # getting min/max coord for each axis
    z_min = arr_min(zs)
    z_max = arr_max(zs)
    y_min = arr_min(ys)
    y_max = arr_max(ys)
    x_min = arr_min(xs)
    x_max = arr_max(xs)

    # extracting values from array
    z_min = z_min.item()
    z_max = z_max.item()
    y_min = y_min.item()
    y_max = y_max.item()
    x_min = x_min.item()
    x_max = x_max.item()

    # converting data types
    z_min = round(z_min)
    z_max = round(z_max)
    y_min = round(y_min)
    y_max = round(y_max)
    x_min = round(x_min)
    x_max = round(x_max)

    # assembling boundaries tuple
    boundaries = (z_min, z_max,
                  y_min, y_max,
                  x_min, x_max)

    # returning boundaries tuple
    return boundaries


def get_centroid_boundaries(coords: tuple,
                            bounding_dims: tuple
                            ) -> tuple:
    """
    Given a centroid coords and bounding
    size dimensions, returns respective
    boundaries.
    """
    # extracting coords from tuples
    z, y, x = coords
    z_dim, y_dim, x_dim = bounding_dims

    # getting radius from dims
    z_radius = z_dim // 2
    y_radius = y_dim // 2
    x_radius = x_dim // 2

    # getting min/max coord for each axis
    z_min = z - z_radius
    z_max = z + z_radius
    y_min = y - y_radius
    y_max = y + y_radius
    x_min = x - x_radius
    x_max = x + x_radius

    # assembling boundaries tuple
    boundaries = (z_min, z_max,
                  y_min, y_max,
                  x_min, x_max)

    # returning boundaries tuple
    return boundaries


def get_radial_distances(centroid: tuple,
                         boundaries_coords: ndarray,
                         scales_dict: dict
                         ) -> ndarray:
    """
    Given an object centroid, and an
    array of given object boundaries,
    returns an array of its distances,
    scaled according to given dictionary.
    """
    # getting specific distance weights for each axis
    z_scale = scales_dict['z']
    y_scale = scales_dict['y']
    x_scale = scales_dict['x']

    # extracting values from centroid coord
    centroid_z, centroid_y, centroid_x = centroid

    # extracting coords arrays
    boundaries_zs, boundaries_ys, boundaries_xs = boundaries_coords

    # getting array shape
    arr_shape = len(boundaries_zs)

    # assembling centroid coords arrays
    centroid_zs = full(shape=arr_shape, fill_value=centroid_z)
    centroid_ys = full(shape=arr_shape, fill_value=centroid_y)
    centroid_xs = full(shape=arr_shape, fill_value=centroid_x)

    # getting current distances arrays
    distances_zs = boundaries_zs - centroid_zs
    distances_ys = boundaries_ys - centroid_ys
    distances_xs = boundaries_xs - centroid_xs

    # converting data types
    distances_zs = distances_zs.astype(float16)
    distances_ys = distances_ys.astype(float16)
    distances_xs = distances_xs.astype(float16)

    # scaling distances (applying weights according to scales dict)
    distances_zs *= z_scale
    distances_ys *= y_scale
    distances_xs *= x_scale

    # squaring distances
    distances_zs = distances_zs ** 2
    distances_ys = distances_ys ** 2
    distances_xs = distances_xs ** 2

    # getting squared distances sum
    distances_sum = distances_zs + distances_ys + distances_xs

    # getting square root of squared distances sum
    radial_distances = distances_sum ** 0.5

    # returning radial distances
    return radial_distances


def get_radius_ratio(radial_distances: ndarray) -> float:
    """
    Given an object radial distances,
    returns radius ratio.
    """
    # getting min/max radius
    min_radius = arr_min(radial_distances)
    max_radius = arr_max(radial_distances)

    # getting radius ratio
    radius_ratio = min_radius / max_radius

    # returning radius ratio
    return radius_ratio


def get_shortest_distance(coord: tuple,
                          coords: ndarray,
                          scales_dict: dict
                          ) -> tuple:
    """
    Given a set of coordinates to a dot,
    and an array of other coordinates,
    returns the shortest distance alongside
    its coordinates.
    """
    # getting radial distances (calculates all distances between dot coord and other coords)
    radial_distances = get_radial_distances(centroid=coord,
                                            boundaries_coords=coords,
                                            scales_dict=scales_dict)

    # getting min radial distance
    shortest_distance = arr_min(radial_distances)

    # finding min coordinate index
    min_index = argwhere(radial_distances == shortest_distance)

    # getting min distance lamina coords
    z_min_coord = coords[0][min_index][0]
    y_min_coord = coords[1][min_index][0]
    x_min_coord = coords[2][min_index][0]

    # extracting values from array
    z_min_coord = z_min_coord.item()
    y_min_coord = y_min_coord.item()
    x_min_coord = x_min_coord.item()

    # assembling shortest distance coord tuple
    shortest_coord = (z_min_coord, y_min_coord, x_min_coord)

    # assembling final tuple
    final_tuple = (shortest_distance, shortest_coord)

    # returning final tuple
    return final_tuple







def get_contour_textures(contour: ndarray,
                         phase_red: ndarray,
                         phase_green: ndarray,
                         phase_blue: ndarray
                         ) -> DataFrame:
    """
    given a contour,
    return its texture metrics
    """

    # organize images in dict for future looping
    phase_images = {
        'red': phase_red,
        'blue': phase_blue,
        'green': phase_green
    }

    # Dictionary to store all results
    texture_dict = {}

    # getting bounding box around the contour
    x, y, w, h = boundingRect(contour)

    for image in phase_red, phase_blue, phase_green:
        # getting image wanted patchs
        roi = image[y:y + h, x:x + w]

        # getting contour grey level co-ocurrence matrix
        glcm = greycomatrix(contour, [5], [0], 256, symmetric=True, normed=True)

        # Compute and store texture features with dynamic keys
        texture_dict[f'{phase_name}_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        texture_dict[f'{phase_name}_dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
        texture_dict[f'{phase_name}_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
        texture_dict[f'{phase_name}_energy'] = graycoprops(glcm, 'energy')[0, 0]
        texture_dict[f'{phase_name}_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
        texture_dict[f'{phase_name}_ASM'] = graycoprops(glcm, 'ASM')[0, 0]

    # assembling texture df, i.e, making a row
    texture_df = DataFrame(texture_dict, index=[0])  # noqa

    return texture_df


def get_files_in_folder(path_to_folder: str,
                        extension: str
                        ) -> list:
    """
    Given a path to a folder, returns a list containing
    all files in folder that match given extension.
    """
    # getting all files in folder
    all_files_in_folder = listdir(path_to_folder)

    # getting specific files
    files_in_dir = [file  # getting file
                    for file  # iterating over files
                    in all_files_in_folder  # in input folder
                    if file.endswith(extension)]  # only if file matches given extension

    # sorting list
    files_in_dir = sorted(files_in_dir)

    # returning list
    return files_in_dir


def make_contour_label(contour_index: int,
                       centroid_x: int or float,
                       centroid_y: int or float,
                       color: int or tuple,
                       thickness: int,
                       img_to_label: ndarray,
                       contour: ndarray,
                       ):
    """
    Writes a single contour label in an image
    """

    # making prep to put outlines and labels
    # fontScale
    font_scale = 1

    # fontStyle
    font_style = LINE_8

    # set font style
    font = FONT_HERSHEY_COMPLEX

    # the following is not related to the dict
    # but using the loop in the contours list
    putText(img_to_label,
            str(contour_index),
            (int(centroid_x), int(centroid_y)),
            font,
            font_scale,
            color,
            thickness,
            font_style)

    # drawing contours in img
    drawContours(img_to_label,
                 [contour],
                 -1,
                 color,
                 thickness)


def save_img(output_folder: str,
             file_name: str,
             img_to_save: ndarray
             ) -> None:
    """
    saves image in designated directory
    """
    # make output path
    output_path = join(output_folder, file_name)

    # save image
    imwrite(output_path, img_to_save)

    return

def make_dir_list(class_list: str,
                  output_folder: str
                  ) -> dict:

    """
    Given a list of strings,
    makes a directory with the name of each of them
    in a specified folder and returns
    a dict with folder name and path
    """
    # empty list to save names
    output_path_list = []

    # empty dict to save
    output_path_dict = {}

    # make a loop to create directories
    for class_name in class_list:
        # define class path
        class_path = join(output_folder, class_name)

        # make directory
        makedirs(class_path, exist_ok=True)

        # append list of directories
        output_path_list.append(class_path)

        # organize in dict
        dict_line = {class_name: class_path}

        output_path_dict.update(dict_line)
        #update dict

    return output_path_dict










