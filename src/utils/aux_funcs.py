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
import cv2 as cv
from os.path import join

# from typing import List, Tuple

######################################################################
# defining auxiliary functions

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
                       centroid_x: int | float,
                       centroid_y: int | float,
                       color: int | tuple,
                       thickness: int,
                       img_to_label: ndarray,
                       contour: ndarray,
                       ):
    """
    Writes a single contour label in an image
    """

    # make img rgb for colored outlines
    colored_img = cv2.cvtColor(img_to_label, cv2.COLOR_GRAY2BGR)

    # making prep to put outlines and labels
    # fontScale
    font_scale = 1

    # fontStyle
    font_style = cv.LINE_8

    # set font style
    font = cv.FONT_HERSHEY_COMPLEX

    # the following is not related to the dict
    # but using the loop in the contours list
    cv.putText(colored_img,
               str(contour_index),
               (int(centroid_x), int(centroid_y)),
               font,
               font_scale,
               color,
               thickness,
               font_style)

    # drawing contours in img
    cv.drawContours(img_to_label,
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
    cv.imwrite(output_path, img_to_save)

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










