# auxiliary functions module

# Code destined to storing auxiliary
# functions to main module.
# this module was created by Angelo Luiz Angonezi, angeloangonezi2@gmail.com

######################################################################
# imports

# importing required libraries
from os import listdir
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
from numpy import mean
from numpy import median
from numpy import min

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
    given a countour,
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
                       centroid_x: float,
                       centroid_y: float,
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






