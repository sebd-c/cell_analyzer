# auxiliary functions module

# Code destined to storing auxiliary
# functions to main module.
# this module was created by Angelo Luiz Angonezi, angeloangonezi2@gmail.com

######################################################################
# imports

# importing required libraries
from copy import copy
from os import listdir
from os import mkdir
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
from math import sqrt
from cv2 import fitEllipse
from cv2 import arcLength
from math import pi
from pandas import DataFrame
from skimage.feature import graycomatrix, graycoprops
from os.path import join
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from numpy import mean
from numpy import median
from numpy import min
from numpy import unique
from numpy import argwhere
from numpy import min as arr_min
from numpy import max as arr_max
from numpy import mean as arr_mean
from numpy import median as arr_median
from numpy import full

######################################################################
# defining auxiliary functions


def spacer(char: str = '_',
           reps: int = 50
           ) -> None:
    """
    Given a char and a number of reps,
    prints a "spacer" string assembled
    by multiplying char by reps.
    :param char: String. Represents a character to be used
    as basis for spacer.
    :param reps: Integer. Represents number of character's
    repetitions used in spacer.
    :return: None.
    """
    # defining spacer string
    spacer_str = char * reps

    # printing spacer string
    print(spacer_str)


def flush_string(string: str) -> None:
    """
    Given a string, writes and flushes it in the console using
    sys library, and resets cursor to the start of the line.
    (writes N backspaces at the end of line, where N = len(string)).
    :param string: String. Represents a message to be written in the console.
    :return: None.
    """
    # getting string length
    string_len = len(string)

    # creating backspace line
    backspace_line = '\b' * string_len

    # writing string
    stdout.write(string)

    # flushing console
    stdout.flush()

    # resetting cursor to start of the line
    stdout.write(backspace_line)


def flush_or_print(string: str,
                   index: int,
                   total: int
                   ) -> None:
    """
    Given a string, prints string if index
    is equal to total, and flushes it on console
    otherwise.
    !(useful for progress tracking/progress bars)!
    :param string: String. Represents a string to be printed on console.
    :param index: Integer. Represents an iterable's index.
    :param total: Integer. Represents an iterable's total.
    :return: None.
    """
    # checking whether index is last

    # if current element is last
    if index == total:

        # printing string
        print(string)

    # if current element is not last
    else:

        # flushing string
        flush_string(string)


def print_execution_parameters(params_dict: dict) -> None:
    """
    Given a list of execution parameters,
    prints given parameters on console,
    such as:
    '''
    --Execution parameters--
    input_folder: /home/angelo/Desktop/ml_temp/imgs/
    output_folder: /home/angelo/Desktop/ml_temp/overlays/
    '''
    :param params_dict: Dictionary. Represents execution parameters names and values.
    :return: None.
    """
    # defining base params_string
    params_string = f'--Execution parameters--'

    # iterating over parameters in params_list
    for dict_element in params_dict.items():
        # getting params key/value
        param_key, param_value = dict_element

        # adding 'Enter' to params_string
        params_string += '\n'

        # getting current param string
        current_param_string = f'{param_key}: {param_value}'

        # appending current param string to params_string
        params_string += current_param_string

    # printing final params_string
    spacer()
    print(params_string)
    spacer()


def print_progress_message(base_string: str,
                           index: int,
                           total: int
                           ) -> None:
    """
    Given a base string (containing keywords #INDEX#
    and #TOTAL#) an index and a total (integers),
    prints execution message, substituting #INDEX#
    and #TOTAL# keywords by respective integers.
    !!!Useful for FOR loops execution messages!!!
    :param base_string: String. Represents a base string.
    :param index: Integer. Represents an execution index.
    :param total: Integer. Represents iteration maximum value.
    :return: None.
    """
    # getting percentage progress
    progress_ratio = index / total
    progress_percentage = progress_ratio * 100
    progress_percentage_round = round(progress_percentage)

    # assembling progress string
    progress_string = base_string.replace('#INDEX#', str(index))
    progress_string = progress_string.replace('#TOTAL#', str(total))
    progress_string += '...'
    progress_string += f' ({progress_percentage_round}%)'
    progress_string += '     '

    # showing progress message
    flush_or_print(string=progress_string,
                   index=index,
                   total=total)


def enter_to_continue():
    """
    Waits for user input ("Enter")
    and once press, continues to run code.
    """
    # defining enter_string
    enter_string = f'press "Enter" to continue'

    # waiting for user input
    input(enter_string)


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


def get_contour_centroid(contour: ndarray) -> tuple:
    """
    Given a contour, returns
    center coordinates in a tuple
    of following structure:
    (cx, cy)
    """
    # getting contour centroid
    m = moments(contour)

    # separating the centroid's coordinates
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])

    return cx, cy


def get_area_box(contour: ndarray, contour_area: float) -> float:
    """
    given a contour,
    returns its calculated area box
    """

    # get minimum area rectangle
    ((cx, cy), (width, height), theta) = minAreaRect(contour)

    # acquire its area
    contour_min_rect_area = width * height

    # calculates area box by definition given in IPP6
    contour_area_box = contour_area / contour_min_rect_area

    return contour_area_box


def get_distance(point_o: tuple, point_d: tuple) -> float:
    """
    given two points in space,
    x1, y1 and x2, y2,
    returns the distance between them
    """
    x1, y1 = point_o
    x2, y2 = point_d

    # calculates distance
    distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance


def get_contour_rratio(contour: ndarray, origin: tuple) -> float:
    """
    given a contour,
    gets its max and min
    distance from the centroid
    to the contour surface,
    and calculate the ratio between them
    i.e. the radius ratio
    """

    # holder variable to max_radius
    max_radius = float('-inf')

    # holder variable to min_radius
    min_radius = float('inf')

    # loop to run in coordinates
    for destiny in contour:

        # the actual type of "destiny" in contour is actually an array
        # this is why we need to first convert to list.

        # converting array to list
        destiny_list = destiny.tolist()

        # unpacking values
        destiny_x, destiny_y = destiny_list[0]

        # assembling destiny coords tuple
        destiny_coords = (destiny_x, destiny_y)

        # calculates distance
        distance = get_distance(point_o=origin,
                                point_d=destiny_coords)

        # compares curr dist to the radiuses
        # and updates than if necessary

        # max radius clause
        if distance >= max_radius:
            max_radius = distance

        # min radius clause
        if distance <= min_radius and distance != 0:
            min_radius = distance

    # calculates the ratio between the radiuses
    radius_ratio = max_radius / min_radius

    return radius_ratio


def get_contour_ellipse_feats(contour: ndarray) -> tuple:
    """
    given a contour, returns its "aspect"
    calculated bt the ratio between
    an ellipses major and minor axis,
    and its eccentricity, given by mathematical formula
    """
    # get the corresponding ellipse
    center, axis, theta = fitEllipse(contour)

    # unpacking width/height
    width, height = axis

    # getting major/minor axes
    minor_axis = width if width < height else height
    major_axis = width if width > height else height

    # calculates aspect
    # TODO: check here for errors of 0 division
    aspect = major_axis / minor_axis

    # calculates the eccentricity
    eccentricity = sqrt((major_axis ** 2) - (minor_axis ** 2)) / major_axis ** 2

    return aspect, eccentricity


def get_contour_roundness(contour: ndarray, contour_area: float) -> float:
    """
    given a contour, returns its "roundness"
    calculated as shown in IPP
    """

    # getting contour perimeter
    contour_perimeter = arcLength(contour, True)

    # calculating rondness
    roundness = (contour_perimeter ** 2) / (4 * contour_area * pi)

    return roundness

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


def save_img_outlayers(overlays_output_folder: str,
                       mask_name: str,
                       img_to_label: ndarray
                       ) -> None:
    """
    saves already labeled images in designated directory
    """
    overlays_output_path = join(overlays_output_folder, mask_name)
    imwrite(overlays_output_path, img_to_label)

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
        mkdir(class_path)

        # append list of directories
        output_path_list.append(class_path)

        # organize in dict
        dict_line = {class_name: class_path}

        output_path_dict.update(dict_line)
        #update dict

    return output_path_dict




