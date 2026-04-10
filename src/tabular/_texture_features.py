# This module contain functions related to
# image pré-processing or auxiliary functions to bigger modules
# this module was created by Débora Sousa, sebd-c @ github
##################################################################
# imports
import cv2 as cv
import numpy as np
from skimage import feature as skifeat
import pandas as pd
import shapely
from src.tabular._geometric_features import get_distance
#################################################################
# pixel intensity metrics
def get_intensity_features(area: float,
                           image: np.ndarray,
                           mask_image: np.ndarray,
                           prefix: str
                           ) -> dict:

    pixint_list = image[mask_image == 1]
    pixint_list = pixint_list.flatten()

    intensities_dict = {}
    values_arr = np.asarray(pixint_list)
    total = float(np.sum(values_arr))
    intensities_dict[f"{prefix}_mean"] = float(np.mean(values_arr))
    intensities_dict[f"{prefix}_median"] = float(np.median(values_arr))
    intensities_dict[f"{prefix}_max"] = float(np.max(values_arr))
    intensities_dict[f"{prefix}_min"] = float(np.min(values_arr))
    intensities_dict[f"{prefix}_sum"] = total
    intensities_dict[f"{prefix}_int_density"] = total / area if area != 0 else np.nan

    return intensities_dict
##################################################################################
# lbp texture functions
def get_inscribed_rect_mask(contour: np.ndarray,
                            ) -> tuple:
    """

    :param img_shape:
    :param contour:
    :return:
    """
    # transform opencv contour's into shapely polygon format
    poly = shapely.Polygon(contour.reshape(-1, 2))

    # create the maximum inscribed circle
    mic = shapely.maximum_inscribed_circle(poly)

    # get mic's centroid
    mic_cx = mic.centroid.x
    mic_cy = mic.centroid.y

    # and radius
    mic_radius = get_distance((mic_cx, mic_cy), mic.coords[0])

    # get top left coordinates
    x1 = int(round(mic_cx - mic_radius * (2 ** 0.5)))
    x2 = int(round(mic_cx + mic_radius * (2 ** 0.5)))

    # get bottom right coordinates
    y1 = int(round(mic_cy - mic_radius * (2 ** 0.5)))
    y2 = int(round(mic_cy + mic_radius * (2 ** 0.5)))

    return x1, x2, y1, y2

def get_lbp_hist(image:np.ndarray,
                 mask: np.ndarray,
                 points: int,
                 radius: int,
                 n_bins=None
                 ):

    # compute LBP on the available image support and only count codes inside the mask
    lbp = skifeat.local_binary_pattern(image, points, radius, 'uniform')
    lbp_values = lbp[mask > 0]

    #
    # Determine number of bins from max LBP value
    if n_bins is None:
        n_bins = int(lbp.max() + 1)

    # Histogram
    hist, _ = np.histogram(lbp_values,
                           bins=n_bins,
                           range=(0, n_bins),
                           density=True)

    return hist


def get_lbp_metrics(hist: np.ndarray,
                    prefix: str
                    ) -> dict:
    """
    Convert an LBP histogram into multiple scalar metrics.

    Metrics computed:
    - entropy
    - energy
    - asm
    - mean_code
    - var_code
    - skewness
    - kurtosis

    Parameters
    ----------
    hist : 1D ndarray LBP histogram (normalized or counts)

    Returns
    -------
    dict
        Dictionary of scalar metrics.
    """

    # Normalize histogram
    h = hist.astype(float)
    h = h / (h.sum() + 1e-12)
    h_safe = h + 1e-12

    # Bin indices
    indices = np.arange(len(h))

    # First-order metrics
    mean_code = np.sum(indices * h)
    var_code = np.sum(((indices - mean_code) ** 2) * h)

    # Higher-order moments
    skewness = np.sum(((indices - mean_code)**3) * h) / (var_code**1.5 + 1e-12)
    kurtosis = np.sum(((indices - mean_code)**4) * h) / (var_code**2 + 1e-12)

    # LBP-specific metrics
    entropy = -np.sum(h_safe * np.log(h_safe))
    asm = np.sum(h**2)
    energy = np.sqrt(asm)

    lbp_dict = {}

    lbp_dict[f"{prefix}_lbp_entropy"] = float(entropy)
    lbp_dict[f"{prefix}_lbp_energy"] = float(energy)
    lbp_dict[f"{prefix}_lbp_asm"] = float(asm)
    lbp_dict[f"{prefix}_lbp_mean_code"] = float(mean_code)
    lbp_dict[f"{prefix}_lbp_var_code"] = float(var_code)
    lbp_dict[f"{prefix}_lbp_skewness"] = float(skewness)
    lbp_dict[f"{prefix}_lbp_kurtosis"] = float(kurtosis)

    return lbp_dict


def run_lbp_metrics(image:np.ndarray,
                    contour: np.ndarray,
                    prefix: str
                    ) -> dict:
    """

    :param image:
    :param contour:
    :return:
    """
    shape = image.shape
    mask = np.zeros(shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, color=1, thickness=-1)

    lbp_hist = get_lbp_hist(image=image,
                            mask=mask,
                            points=8,
                            radius=1,
                            )

    lbp_dict = get_lbp_metrics(hist=lbp_hist,
                               prefix=prefix)

    return lbp_dict


def smallest_binary_rotation(bits):
    n = len(bits)
    rotations = [
        bits[i:] + bits[:i]
        for i in range(n)
    ]

    code = "".join(map(str, min(rotations)))

    return code


def get_lbp_codes(image: np.ndarray,
                  mask: np.ndarray
                  ) -> list:
    """

    """
    # list of mask pixel coordinates
    ys, xs = np.where(mask > 0)
    coords = np.vstack([ys, xs]).T

    # lbp list of codes
    codes_list = []
    for coord in coords:
        # neighbors offset
        neighbor_list = [(coord[0] + 1, coord[1]), # up
                         (coord[0] - 1, coord[1]), # down
                         (coord[0], coord[1] - 1), # left
                         (coord[0], coord[1] + 1), # right
                         (coord[0] + 1, coord[1] + 1), # diag up right
                         (coord[0] - 1, coord[1] + 1), # diag bot right
                         (coord[0] + 1, coord[1] - 1), # diag up left
                         (coord[0] - 1, coord[1] - 1)] # diag bot left

        # get validity of neighborhood
        values_at_coords = mask[neighbor_list]
        points_per_value = np.bincount(values_at_coords)

        # if it's a valid pixel
        if points_per_value == 8:
            coord_bin_list = []
            # iterate over its neighbors
            for neighbor in neighbor_list:
                # to get the number code of it
                if image[coord[0], coord[1]] > image[neighbor[0], neighbor[1]]:
                    coord_bin_list.append(0)
                else:
                    coord_bin_list.append(1)

            # make it invariant to rotation
            inv_code = smallest_binary_rotation(coord_bin_list)

            # append to list of codes
            codes_list.append(inv_code)


    return codes_list


def quantize_image(image: np.ndarray, levels: int) -> np.ndarray:
    """
    Quantize an 8-bit grayscale image to a fixed number of gray levels.
    """
    # normalize img to 8-bit just in case
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # set binsize, considering that 256 will generate a lot of noise in the glcm matrix
    # I believe that humanly, for 8 bit imgs, 8 levels would suffice,
    # which means 32 for binsize is a good trying method
    binsize = 256 / levels

    # divide each pixel by bin size
    quant = (image // binsize).astype(np.int32)

    # safety case to prevent boundary cases
    quant[quant >= levels] = levels - 1
    return quant


def get_masked_glcm(image: np.ndarray,
                    mask: np.ndarray,
                    distances: list,
                    angles: list,
                    levels: int
                    ) -> np.ndarray:
    """
    Compute a masked GLCM where both pixels of the pair must be inside the object's mask
    Returns a 4D array: GLCM[level, level, distance_id, angle_id], where level is
    the quantized gray-level index for each pixel in the pair
    """
    h, w = image.shape
    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)

    # pre-quantized image
    quant = quantize_image(image, levels)

    # list of mask pixel coordinates
    ys, xs = np.where(mask > 0)
    coords = np.vstack([ys, xs]).T

    # iterate over all mask pixels
    for d_i, d in enumerate(distances):
        for a_i, angle in enumerate(angles):

            # offset for this (distance, angle)
            dy = int(round(np.sin(angle) * d))
            dx = int(round(np.cos(angle) * d))

            # to each pont get its neighbor with according offset
            for y, x in coords:
                y2 = y + dy
                x2 = x + dx

                # check image bounds and mask bounds
                if 0 <= y2 < h and 0 <= x2 < w and mask[y2, x2] > 0:
                    # if valid, get the quantized pixel version from image
                    # to set the pair of bins
                    i = quant[y, x]
                    j = quant[y2, x2]
                    # and update the glcm
                    glcm[i, j, d_i, a_i] += 1

    # Normalize each GLCM
    for d_i in range(len(distances)):
        for a_i in range(len(angles)):
            total = glcm[:, :, d_i, a_i].sum()
            if total > 0:
                glcm[:, :, d_i, a_i] /= total

    return glcm


def get_glcm_features(image: np.ndarray,
                      mask: np.ndarray,
                      levels: int,
                      distances: list,
                      angles_deg: list,
                      prefix: str
                      ) -> dict:
    """
    Compute GLCM Haralick features that are:
      - angle-invariant (mean over angles)
      - distance-specific (one feature per distance)

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image
    mask : np.ndarray
        Binary mask
    levels : int
        Number of gray levels
    distances : list
        GLCM distances (e.g. [1, 2, 3])
    angles_deg : list
        Angles in degrees

    Returns
    -------
    pd.DataFrame
    """

    # Convert angles to radians
    angles_rad = [np.deg2rad(a) for a in angles_deg]

    # Compute masked GLCM
    glcm = get_masked_glcm(image=image,
                           mask=mask,
                           distances=distances,
                           angles=angles_rad,
                           levels=levels
                           )

    properties = ["contrast",
                  "dissimilarity",
                  "homogeneity",
                  "energy",
                  "correlation",
                  "ASM",
                  "entropy"
                  ]

    features = {}

    for prop in properties:
        vals = skifeat.graycoprops(glcm, prop=prop)

        for d_i, d in enumerate(distances):
            # angle-invariant → mean over angles
            features[f"{prefix}_glcm_{prop}_d{d}"] = vals[d_i, :].mean()

    return features

