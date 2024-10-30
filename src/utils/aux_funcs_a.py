# auxiliary functions module
# Code destined to storing auxiliary
# functions to main module.

######################################################################
# imports

from os import listdir
from sys import stdout
from typing import Union
from cv2 import cvtColor
from numpy import ndarray
from os.path import exists
from numpy import argwhere
from pandas import DataFrame
from cv2 import COLOR_GRAY2BGR
from cv2 import IMREAD_GRAYSCALE
from os import get_terminal_size
from numpy import uint8 as np_uint8
from cv2 import imread as cv_imread
from numpy import zeros as np_zeroes
from numpy import uint16 as np_uint16
from skimage.io import imread as sk_imread
from skimage.io import imsave as sk_imsave

######################################################################
# defining auxiliary functions


def spacer(char: str = '_',
           reps: int = 50
           ) -> None:
    """
    Given a char and a number of reps,
    prints a "spacer" string assembled
    by multiplying char by reps.
    """
    # defining spacer string
    spacer_str = char * reps

    # printing spacer string
    print(spacer_str)


def get_console_width() -> int:
    """
    Returns current console width.
    """
    # getting console dimensions
    width, _ = get_terminal_size()

    # returning console width
    return width


def flush_string(string: str) -> None:
    """
    Given a string, writes and flushes it in the console using
    sys library, and resets cursor to the start of the line.
    (writes N backspaces at the end of line, where N = len(string)).
    """
    # getting console width
    console_width = get_console_width()

    # getting string length
    string_len = len(string)

    # getting size difference
    width_diff = console_width - string_len

    # discounting from width diff to avoid console overflow
    width_diff -= 5

    # getting spacer
    empty_space = ' ' * width_diff

    # updating string
    string += empty_space

    # getting updated string length
    string_len = len(string)

    # creating backspace line
    backspace_line = '\b' * string_len

    # writing string
    stdout.write(string)

    # flushing console
    stdout.flush()

    # resetting cursor to start of the line
    stdout.write(backspace_line)


def clear_console(string: str) -> None:
    """
    Given a string, writes empty space
    to cover string size in console.
    """
    # getting string length
    string_len = len(string)

    # creating empty line
    empty_line = ' ' * string_len

    # creating backspace line
    backspace_line = '\b' * string_len

    # writing string
    stdout.write(empty_line)

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


def print_progress_message(base_string: str,
                           index: int,
                           total: int
                           ) -> None:
    """
    Given a base string (containing keywords #INDEX#
    and #TOTAL#) an index and a total (integers),
    prints execution message, substituting #INDEX#
    and #TOTAL# keywords by respective integers.
    !(useful for FOR loops execution messages)!
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


def enter_to_continue(skip: bool) -> None:
    """
    If skip param is False,
    waits for user input ("Enter")
    and once press, continues to run code.
    """
    # checking skip bool
    if skip:

        # returning None
        return None

    # defining enter_string
    enter_string = f'press "Enter" to continue'

    # waiting for user input
    input(enter_string)


def get_number_string(num: int,
                      digits: int = 2
                      ) -> str:
    """
    Given a number, returns formatted
    number with leading zeroes so that
    the number of digits param is preserved.
    """
    # formating number string
    number_string = f'{num:0{digits}d}'

    # returning formatted number string
    return number_string


def get_valid_path(paths_list: list) -> str or None:
    """
    Given a list of paths, returns
    first valid (existing) path.
    If no paths are valid, returns None.
    """
    # iterating over paths list
    for path in paths_list:

        # checking whether path exists
        if exists(path):

            # returning current path
            return path

    # returning None
    return None


def get_blank_image(width: int,
                    height: int,
                    slices: int = 1
                    ) -> ndarray:
    """
    Given an image width/height, returns
    numpy array of given dimension
    filled with zeroes.
    """
    # defining matrix shape based on number of dimensions
    if slices == 1:  # 2d
        shape = (height, width)
    else:  # 3d
        shape = (slices, height, width)

    # creating blank matrix
    blank_matrix = np_zeroes(shape=shape)

    # returning blank matrix
    return blank_matrix


def load_bgr_img(image_path: str) -> ndarray:
    """
    Given a path to an image,
    returns image as BGR ndarray.
    """
    # opening current image
    current_image = cv_imread(image_path,
                              -1)  # unchanged (original format)

    # checking current image shape (grayscale/BGR)
    image_shape = current_image.shape
    image_len = len(image_shape)

    # checking image type
    if image_len < 3:  # not bgr (grayscale)

        # converting current image to bgr
        current_image = cvtColor(current_image,
                                 COLOR_GRAY2BGR)

    # returning image
    return current_image


def load_grayscale_img(image_path: str) -> ndarray:
    """
    Given a path to an image,
    returns image as grayscale ndarray.
    """
    # opening current image
    current_image = cv_imread(image_path,
                              IMREAD_GRAYSCALE)

    # returning image
    return current_image


def load_stack(image_path: str or Union) -> ndarray:
    """
    Given a path to an image stack,
    returns image as stacked ndarray.
    """
    # opening image stack
    image_stack = sk_imread(image_path)

    # returning stack
    return image_stack


def convert_to_8bit(image: ndarray) -> ndarray:
    """
    Given an image stack, converts it
    to numpy uint8 format, returning
    updated array.
    """
    # converting image to uint8 format
    image = image.astype(np_uint8)

    # returning image
    return image


def convert_to_16bit(image: ndarray) -> ndarray:
    """
    Given an image stack, converts it
    to numpy uint16 format, returning
    updated array.
    """
    # converting image to uint16 format
    image = image.astype(np_uint16)

    # returning image
    return image


def save_stack(save_path: str,
               image_stack: ndarray
               ) -> None:
    """
    Given an image stack, saves it
    to given save path.
    """
    # saving image
    sk_imsave(save_path,
              image_stack,
              check_contrast=False)


def get_dimensions_num(image: ndarray) -> int:
    """
    Given an image, returns its
    number of dimensions.
    """
    # getting image shape
    image_shape = image.shape

    # getting dimensions num
    dimensions_num = len(image_shape)

    # returning dimensions num
    return dimensions_num


def get_high_intensity_pixels_df(image: ndarray,
                                 pixel_threshold: int = 0
                                 ) -> DataFrame:
    """
    Docstring.
    """
    # finding high intensity pixels
    high_intensity_pixels_indices = argwhere(image > pixel_threshold)  # order: ZYX

    # converting array to data frame
    high_intensity_pixels_df = DataFrame(high_intensity_pixels_indices)

    # renaming df columns
    high_intensity_pixels_df.columns = ['z', 'y', 'x']

    # returning pixels df
    return high_intensity_pixels_df

# end of current module
