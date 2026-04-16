# This module contain functions related to
# geometric properties of cellular objects
# this module was created by Débora Sousa, sebd-c @ github
##################################################################
# imports
import numpy as np
import cv2 as cv
import math as m
#################################################################
# functions
def get_contour_centroid(contour: np.ndarray) -> tuple:
    """
    Given a contour, returns
    center coordinates in a tuple
    of following structure:
    (cx, cy)
    """
    # getting contour centroid
    contour_moments = cv.moments(contour)

    # separating the centroid's coordinates
    cx = int(contour_moments['m10'] / contour_moments['m00'])
    cy = int(contour_moments['m01'] / contour_moments['m00'])

    return cx, cy

def get_area_box(contour: np.ndarray, contour_area: float) -> float:
    """
    given a contour,
    returns its calculated area box
    """

    # get minimum area rectangle
    ((cx, cy), (width, height), theta) = cv.minAreaRect(contour)

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
    distance = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance


def get_contour_rratio(contour: np.ndarray, origin: tuple) -> float:
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


def get_contour_ellipse_feats(contour: np.ndarray) -> tuple:
    """
    given a contour, returns its "aspect"
    calculated bt the ratio between
    an ellipses major and minor axis,
    and its eccentricity, given by mathematical formula
    """
    # get the corresponding ellipse
    center, axis, theta = cv.fitEllipse(contour)

    # unpacking width/height
    width, height = axis

    # getting major/minor axes
    minor_axis = width if width < height else height
    major_axis = width if width > height else height

    # calculates aspect
    # TODO: check here for errors of 0 division
    aspect = major_axis / minor_axis

    # calculates the eccentricity
    eccentricity = m.sqrt((major_axis ** 2) - (minor_axis ** 2)) / major_axis ** 2

    return aspect, eccentricity


def get_contour_roundness(contour: np.ndarray, contour_area: float) -> float:
    """
    given a contour, returns its "roundness"
    calculated as shown in IPP
    """

    # getting contour perimeter
    contour_perimeter = cv.arcLength(contour, True)

    # calculating roundness
    roundness = (contour_perimeter ** 2) / (4 * contour_area * pi)

    return roundness

def get_morpho_features(contour: np.ndarray,
                        pixel_int: float,
                        mask_name: str,
                        area: float,
                        )-> dict:
    # getting current contour centroid coords
    centroid_x, centroid_y = get_contour_centroid(contour)

    # getting current contours area box
    area_box = get_area_box(contour, area)

    # getting current contour radius ratio
    radius_ratio = get_contour_rratio(contour, (centroid_x, centroid_y))

    # getting current contours' aspect and eccentricity, respectively
    aspect, eccentricity = get_contour_ellipse_feats(contour)

    # getting current contours' roundness
    roundness = get_contour_roundness(contour, area)

    # calculates cii as per Filippi-Chiela et al, 2012
    ii = (0.9 * aspect) - (0.87 * area_box) + (0.96 * radius_ratio) + (0.92 * roundness)

    # now moving to organizing them into a dictionary
    morpho_dict = {'image_name': mask_name,
                    'index': int(pixel_int),
                    'cx_coords': centroid_x,
                    'cy_coords': centroid_y,
                    'area': area,
                    'area_box': area_box,
                    'radius_ratio': radius_ratio,
                    'aspect': aspect,
                    'eccentricity': eccentricity,
                    'roundness': roundness,
                    'ii': ii,
                    'contour': [contour],
                    }
    return morpho_dict