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
from pandas import DataFrame
from cv2 import imread
from cv2 import findContours
from cv2 import contourArea
from cv2 import RETR_EXTERNAL
from cv2 import boundingRect
from cv2 import CHAIN_APPROX_NONE
from cv2 import moments
from cv2 import minAreaRect
from math import sqrt
from cv2 import fitEllipse
from cv2 import arcLength
from math import pi
from pandas import concat
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


def get_specific_files_in_folder(path_to_folder: str,
                                 extension: str
                                 ) -> list:
    """
    Given a path to a folder, returns a list containing
    all files in folder that match given extension.
    :param path_to_folder: String. Represents a path to a folder.
    :param extension: String. Represents a specific file extension.
    :return: List[str]. Represents all files that match extension in given folder.
    """
    # getting all files in folder
    all_files_in_folder = listdir(path_to_folder)

    # getting specific files
    files_in_dir = [file                          # getting file
                    for file                      # iterating over files
                    in all_files_in_folder        # in input folder
                    if file.endswith(extension)]  # only if file matches given extension

    # sorting list
    files_in_dir = sorted(files_in_dir)

    # returning list
    return files_in_dir


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


def get_contour_min_rect_area(contour: ndarray) -> float:
    """
    Given a contour, returns
    the area of the minimum
    possible rectangle
    """
    # get minimum area rectangle
    (cx, cy, width, height, theta) = minAreaRect(contour)

    # acquire its area
    contour_min_rect_area = (width * height)/2

    return contour_min_rect_area


def get_area_box(contour: ndarray) -> float:
    """
    given a contour,
    returns its calculated area box
    """

    # yet again get the area of said contour
    contour_area = contourArea(contour)
    # and its minimum rectangle
    contour_min_rect_area = get_contour_min_rect_area(contour)

    # calculates area box by definition given in IPP6
    contour_area_box = contour_area/contour_min_rect_area

    return contour_area_box


def get_distance(point_o: tuple, point_d: tuple)-> float:
    """
    given two points in space,
    x1, y1 and x2, y2,
    returns the distance between them
    """
    x1, y1 = point_o
    x2, y2 = point_d

    # calculates distance
    distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance


def get_contour_rratio(contour: ndarray) -> float:
    """
    given a countour,
    gets its max and min
    distance from the centroid
    to the contour surface,
    and calculate the ratio between them
    i.e. the radius ratio
    """
    # getting contour centroid
    origin = get_contour_centroid(contour)

    # holder variable to max_radius
    max_radius = float('-inf')

    # holder variable to min_radius
    min_radius = float('inf')

    # loop to run in coordinates
    for destiny in contour:

        # calculates distance
        distance = get_distance(point_o=origin,
                                point_d=destiny)

        # compares curr dist to the radiuses
        # and updates than if necessary

        # max radius clause
        if distance >= max_radius:
            max_radius = distance

        # min radius clause
        if distance <= min_radius:
            min_radius = distance

    # calculates the ratio between the radiuses
    radius_ratio = max_radius / min_radius

    return radius_ratio


def get_contour_aspect(contour: ndarray) -> float:
    """
    given a contour, returns its "aspect"
    calculated bt the ratio between
    an ellipses major and minor axis
    """
    # get the corresponding ellipse
    center, axis, theta = fitEllipse(contour)

    #separate the axis
    if axis[0] > axis[1]:
        # major axis
        major_axis = axis[0]

        # minor axis
        minor_axis = axis[1]
    else:
        # major axis
        major_axis = axis[1]

        # minor axis
        minor_axis = axis[0]

    # calculates aspect
    aspect = major_axis/minor_axis

    return aspect


def get_contour_roundness(contour: ndarray) -> float:
    """
    given a contour, returns its "roundness"
    calculated as shown in IPP
    """

    #getting contour area
    contour_area = contourArea(contour)

    # getting contour perimeter
    contour_perimeter = arcLength(contour, True)

    # calculating rondness
    roundness = (contour_perimeter**2)/ (4 * contour_area * pi)

    return roundness


def get_eccentricity()
    # calculates the eccentricity
    eccentricity = sqrt((height ** 2) - (width ** 2)) / height ** 2
    if max_radius is None or h < max_radius:
        if w > min_radius and e < eccentricity:
            new_cnt_list.append(contours[i])
    pass


def make_image_contours_df(image_name: str,
                           image_path: str
                          ) -> DataFrame:
    """
    Given a path to a binary image,
    finds contours and returns data
    frame containing contours coords.
    """

    # reading image
    image = imread(image_path,
                   -1)

    # finding contours in image
    contours, _ = findContours(image, RETR_EXTERNAL, CHAIN_APPROX_NONE)

    # gets enumerate to have indexes
    contours_enumerate = enumerate(contours, 1)

    # make empty lists to put the contours' information in
    image_names = []
    contours_indices = []
    contours_coords = []
    contours_areas = []
    contours_area_boxes = []
    contours_rratios = []
    contours_aspects = []
    countours_roundnesses = []

    # loop inside an image
    # to work with that images' contours
    for i, contour in contours_enumerate:

        # getting current contours indices
        contours_indices.append(i)

        # getting current image col lists
        image_names.append(image_name)

        # getting current contours coords
        contours_coords.append(get_contour_centroid(contour))

        # getting current contours areas
        contours_areas.append(contourArea(contour))

        # getting current contours area boxes
        # area / area box
        contours_area_boxes.append(get_area_box(contour))

        # getting current contours' radius ratio
        contours_rratios.append(get_contour_rratio(contour))

        # getting current contours' aspect
        contours_aspects.append(get_contour_aspect(contour))

        # getting current contours' roundness
        countours_roundnesses.append(get_contour_roundness(contour))

    # by the end of that loop, you now have lists of the contours'
    # information in order, now moving to organizing them into a dictionary

    # assembling contours dict
    contours_dict = {'image_name': image_names,
                     'index': contours_indices,
                     'centroid_coords': contours_coords,
                     'area': contours_areas,
                     'area_box': contours_area_boxes,
                     'radius_ratio': contours_rratios,
                     'aspect': contours_aspects,
                     'roundness': countours_roundnesses
                     }

    # assembling contours df
    image_contours_df = DataFrame(contours_dict)

    # returning contours df
    return image_contours_df

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
    files_in_dir = [file                          # getting file
                    for file                      # iterating over files
                    in all_files_in_folder        # in input folder
                    if file.endswith(extension)]  # only if file matches given extension

    # sorting list
    files_in_dir = sorted(files_in_dir)

    # returning list
    return files_in_dir


def make_contours_dfs(image_name: str,
                           image_path: str
                          ) -> DataFrame:
    """
    # TODO escrever descrição
    """
    pass

def make_contours_dfs(input_folder: str,
                      images_extension: str,
                      output_folder: str,
                      ) -> None:
    """
    Given a path to a folder containing
    cytoplasms masks, generates a df
    containing the wanted information,
    and saving the results
    in the output folder.
    """
    # getting files in input folder
    files = get_files_in_folder(path_to_folder=input_folder,
                                extension=images_extension)
    files_num = len(files)

    # create empty list to hold the dfs
    dfs_list = []

    # iterating over files
    for file_index, file in enumerate(files, 1):

        # printing progress message
        base_string = 'generating segmentation df #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=files_num)

        # getting current image input/output paths
        input_path = join(input_folder,
                          file)

        # get image contour df
        image_df = make_image_contours_df(image_name=file,
                                          image_path=input_path)

        # append curr img df to dir df
        dfs_list.append(image_df)

    # concating "dfs" from dfs lists into
    # a pandas dataframe
    contour_df = concat(dfs_list, ignore_index=True)

    # create the path to save the output path
    output_path = join(output_folder,
                       'contour_df.csv') # with the chosen extension

    # saving df
    contour_df.to_csv(output_path, index=False)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('analysis complete!')



