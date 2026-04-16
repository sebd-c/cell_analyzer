# This module contain functions related to
# image pré-processing or auxiliary functions to bigger modules
# this module was created by Débora Sousa, sebd-c @ github
##################################################################
# imports
import cv2 as cv
import numpy as np
from copy import copy
#################################################################
# mask manipulation functions
def apply_mask(image: np.ndarray,
               mask: np.ndarray,
               pixel_intensity: int
               ) -> np.ndarray:
    """
    Given an image, a mask image and a pixel intensity for a labeled object,
    returns a copy of the image with all pixels outside the labeled object as 0
    """
    # make a copy of the image not to alter it
    masked_image = copy(image)

    # apply mask to it
    masked_image[mask != pixel_intensity] = 0

    return masked_image


def get_unique_ids(mask: np.ndarray) -> list:
    """
    Given an image of segmentation masks, returns
    a list containing all object ids.
    """
    # getting current mask unique values
    object_ids = np.unique(mask)

    # converting ids to list
    object_ids = object_ids.tolist()

    if 0 in object_ids:
        # removing zero from list (background pixel)
        object_ids.remove(0)

    # sorting list
    object_ids = sorted(object_ids)

    # returning object ids list
    return object_ids

#################################################################
# crop related functions
def make_crop(image: np.ndarray,
              x1: int,
              x2: int,
              y1: int,
              y2: int,
              max_width: int,
              max_height: int
              ) -> np.ndarray:
    """
    Given an image and 4 points in space,
    returns an unoriented crop image
    """
    # crop based on coordinates
    crop = image[y1:y2, x1:x2]
    print(f"max width is {max_width} and max height {max_height}")
    # get crop current size for padding
    crop_h, crop_w = crop.shape

    # pad if crop is smaller than desired
    pad_h = max(0, max_height - crop_h)
    pad_w = max(0, max_width - crop_w)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    pad_value = 0
    crop_padded = cv.copyMakeBorder(crop, top, bottom, left, right,
                                    borderType=cv.BORDER_CONSTANT,
                                    value=pad_value)
    print(f"unrotated:{crop_padded.shape}")
    return crop_padded



def make_crop_rotate(image: np.ndarray,
                     x1: int,
                     x2: int,
                     y1: int,
                     y2: int,
                     max_width: int,
                     max_height: int
                     ) -> np.ndarray:
    """
    Given an image and 4 points in space,
    returns an unoriented crop image
    """
    # crop based on coordinates
    crop = image[y1:y2, x1:x2]
    print(f"max width is {max_width} and max height {max_height}")
    # rotate accordingly
    # oriented_crop = rotate(crop, ROTATE_90_CLOCKWISE)

    # get crop current size for padding
    crop_h, crop_w = crop.shape

    # pad if crop is smaller than desired
    pad_h = max(0, max_width - crop_h)
    pad_w = max(0, max_height- crop_w)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    pad_value = 0
    crop_padded = cv.copyMakeBorder(crop, top, bottom, left, right,
                                    borderType=cv.BORDER_CONSTANT,
                                    value=pad_value)
    oriented_crop = cv.rotate(crop_padded, cv.ROTATE_90_CLOCKWISE)

    print(f"rotated:{oriented_crop.shape}")
    return oriented_crop

#################################################################
# image generation functions