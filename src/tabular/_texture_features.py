# This module contain functions related to
# image pré-processing or auxiliary functions to bigger modules
# this module was created by Débora Sousa, sebd-c @ github
##################################################################
# imports
import cv2 as cv
import numpy as np
from skimage import feature as skifeat
from skimage import filters as skifilt
from skimage import morphology as skimorph
from skimage import measure as skimeas
from skimage import segmentation as skiseg
import pandas as pd
#################################################################
# pixel intensity metrics
def get_intensity_features(area: float,
                           image: np.ndarray,
                           mask_image: np.ndarray,
                           prefix: str
                           ) -> dict:
    """
    Compute basic intensity metrics for a
    masked object
    """
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
def get_masked_lbp(image: np.ndarray,
                   mask: np.ndarray,
                   points: int,
                   radius: int
                   ) -> np.ndarray:
    """
    Compute rotation-invariant uniform LBP codes only for the object's mask pixels,
    where all sampled neighbors must also be inside the mask
    Returns a 1D array with one LBP code (0 to points + 1) per valid mask pixel
    """
    h, w = image.shape
    img = image.astype(np.float64)
    inside_mask = mask > 0

    # list of mask pixel coordinates
    ys, xs = np.where(inside_mask)

    # circular neighbor offsets, starting at the right and going counter-clockwise
    angles = 2 * np.pi * np.arange(points) / points
    dy = np.round(-radius * np.sin(angles), 5)
    dx = np.round(radius * np.cos(angles), 5)

    # sampling coordinates for each mask pixel, shape (n_pixels, points)
    sy = ys[:, None] + dy[None, :]
    sx = xs[:, None] + dx[None, :]

    # surrounding pixels used for bilinear interpolation of each sample
    y0 = np.floor(sy).astype(int)
    y1 = np.ceil(sy).astype(int)
    x0 = np.floor(sx).astype(int)
    x1 = np.ceil(sx).astype(int)

    # check image bounds for the whole neighborhood
    valid = ((y0 >= 0) & (x0 >= 0) & (y1 < h) & (x1 < w)).all(axis=1)
    ys, xs = ys[valid], xs[valid]
    sy, sx = sy[valid], sx[valid]
    y0, y1, x0, x1 = y0[valid], y1[valid], x0[valid], x1[valid]

    # check mask bounds, every pixel used by the neighborhood must belong to the object
    valid = (inside_mask[y0, x0] & inside_mask[y0, x1] &
             inside_mask[y1, x0] & inside_mask[y1, x1]).all(axis=1)
    ys, xs = ys[valid], xs[valid]
    sy, sx = sy[valid], sx[valid]
    y0, y1, x0, x1 = y0[valid], y1[valid], x0[valid], x1[valid]

    # bilinear interpolation of the neighbor intensities
    fy = sy - y0
    fx = sx - x0
    neighbors = ((1 - fy) * (1 - fx) * img[y0, x0] +
                 (1 - fy) * fx * img[y0, x1] +
                 fy * (1 - fx) * img[y1, x0] +
                 fy * fx * img[y1, x1])

    # binary pattern: 1 where neighbor >= center pixel
    bits = neighbors >= img[ys, xs][:, None]

    # number of circular 0/1 transitions in the pattern
    transitions = np.sum(bits != np.roll(bits, -1, axis=1), axis=1)

    # uniform patterns are coded by their number of 1s, the rest share code points + 1
    codes = np.where(transitions <= 2, bits.sum(axis=1), points + 1)

    return codes


def get_lbp_hist(image:np.ndarray,
                 mask: np.ndarray,
                 points: int,
                 radius: int,
                 n_bins=None
                 ):
    """
    Compute the normalized histogram of the masked LBP codes
    """
    lbp_values = get_masked_lbp(image=image,
                                mask=mask,
                                points=points,
                                radius=radius
                                )

    # 'uniform' lbp has P + 1 uniform codes plus one bin for non-uniform ones
    if n_bins is None:
        n_bins = points + 2

    # safety case for objects too thin to have any valid pixel
    if lbp_values.size == 0:
        return np.full(n_bins, np.nan)

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
                    mask: np.ndarray,
                    prefix: str,
                    points: int = 8,
                    radius: int = 1
                    ) -> dict:
    """
    Compute LBP metrics over the whole object mask

    :param image: 2D grayscale image
    :param mask: binary mask of the object
    :param prefix: prefix for the feature names
    :param points: number of LBP sampling points
    :param radius: LBP radius
    :return: dict of LBP metrics
    """
    lbp_hist = get_lbp_hist(image=image,
                            mask=mask,
                            points=points,
                            radius=radius,
                            )

    lbp_dict = get_lbp_metrics(hist=lbp_hist,
                               prefix=prefix)

    return lbp_dict


def quantize_image(image: np.ndarray, levels: int) -> np.ndarray:
    """
    Quantize an 8-bit grayscale image to a fixed number of gray levels.
    """
    # normalize img to 8-bit just in case
    if image.dtype != np.uint8:
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

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


##################################################################################
# local texture functions
def get_mask_crop(mask: np.ndarray) -> tuple:
    """
    Get the bounding box slices of the object's mask
    """
    ys, xs = np.where(mask > 0)
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


def get_local_texture_features(image: np.ndarray,
                               mask: np.ndarray,
                               radius: int,
                               levels: int,
                               prefix: str
                               ) -> dict:
    """
    Compute local texture maps (entropy, gradient and mean) inside a disk neighborhood,
    where only mask pixels take part in each neighborhood, and summarize them over the
    pixels whose whole neighborhood is inside the object (same rule as LBP/GLCM)

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image
    mask : np.ndarray
        Binary mask
    radius : int
        Radius of the disk neighborhood
    levels : int
        Number of gray levels used for the local entropy
    prefix : str
        Prefix for the feature names

    Returns
    -------
    dict
    """
    feature_names = [f"{prefix}_local_entropy_mean",
                     f"{prefix}_local_entropy_std",
                     f"{prefix}_local_gradient_mean",
                     f"{prefix}_local_gradient_std",
                     f"{prefix}_local_mean_std"
                     ]

    # work only on the object's bounding box
    crop = get_mask_crop(mask)
    img = image[crop]
    if img.dtype != np.uint8:
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    obj_mask = (mask[crop] > 0).astype(np.uint8)

    footprint = skimorph.disk(radius)

    # pixels whose whole neighborhood is inside the object
    valid = cv.erode(obj_mask, footprint.astype(np.uint8), borderValue=0) > 0

    # safety case for objects too thin to have any valid pixel
    if not valid.any():
        return {name: np.nan for name in feature_names}

    # entropy is computed on the quantized image, since with 256 levels almost every
    # pixel of a small neighborhood is unique and entropy only measures noise
    quant = quantize_image(img, levels).astype(np.uint8)

    # local maps, the mask keeps background pixels out of each neighborhood
    entropy_map = skifilt.rank.entropy(quant, footprint, mask=obj_mask)
    gradient_map = skifilt.rank.gradient(img, footprint, mask=obj_mask)
    mean_map = skifilt.rank.mean(img, footprint, mask=obj_mask)

    entropy_values = entropy_map[valid].astype(np.float64)
    gradient_values = gradient_map[valid].astype(np.float64)
    mean_values = mean_map[valid].astype(np.float64)

    values = [entropy_values.mean(),
              entropy_values.std(),
              gradient_values.mean(),
              gradient_values.std(),
              mean_values.std()
              ]

    return {name: float(value) for name, value in zip(feature_names, values)}


##################################################################################
# spot detection functions
def get_spot_sizes(smooth: np.ndarray,
                   obj_mask: np.ndarray,
                   coords: np.ndarray,
                   median: float
                   ) -> list:
    """
    Measure each spot's size as its half-maximum area in pixels, i.e. the connected
    region around the peak brighter than median + (peak - median) / 2

    Touching spots are first split by a watershed seeded at the peaks,
    restricted to the object's mask
    """
    if len(coords) == 0:
        return []

    # one marker per spot
    markers = np.zeros(smooth.shape, dtype=np.int32)
    markers[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1)

    # split the object into one basin per spot
    basins = skiseg.watershed(-smooth, markers=markers, mask=obj_mask)

    sizes = []
    for spot_id, (y, x) in enumerate(coords, start=1):
        # half maximum level of this spot above the object's median
        half_max = median + (smooth[y, x] - median) / 2
        spot_region = (basins == spot_id) & (smooth >= half_max)

        # keep only the part connected to the peak
        regions = skimeas.label(spot_region, connectivity=2)
        sizes.append(int(np.sum(regions == regions[y, x])))

    return sizes


def get_spot_features(image: np.ndarray,
                      mask: np.ndarray,
                      prefix: str,
                      sigma: float = 1.0,
                      min_distance: int = 2,
                      k: float = 5.0,
                      invert: bool = False
                      ) -> dict:
    """
    Detect bright spots inside the object's mask as local maxima of the smoothed image
    that stand out from the object's own intensity distribution

    Smoothing uses normalized convolution, so only mask pixels are averaged,
    and a spot must be above median + k * robust std of the object's pixels

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image
    mask : np.ndarray
        Binary mask
    prefix : str
        Prefix for the feature names
    sigma : float
        Gaussian sigma used to smooth noise before finding maxima
    min_distance : int
        Minimum distance in pixels between two spots
    k : float
        Number of robust standard deviations above the median for a spot
    invert : bool
        If True, detect dark spots instead (e.g. granules in brightfield channels),
        in which case intensities and contrast refer to the inverted image

    Returns
    -------
    dict
    """
    # work only on the object's bounding box
    crop = get_mask_crop(mask)
    img = image[crop].astype(np.float64)
    obj_mask = mask[crop] > 0
    obj_mask_f = obj_mask.astype(np.float64)

    # dark spots become bright spots on the inverted image
    if invert:
        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max
        else:
            max_value = img[obj_mask].max()
        img = max_value - img

    # mask-only gaussian smoothing: blur(img * mask) / blur(mask)
    num = cv.GaussianBlur(img * obj_mask_f, (0, 0), sigma, borderType=cv.BORDER_CONSTANT)
    den = cv.GaussianBlur(obj_mask_f, (0, 0), sigma, borderType=cv.BORDER_CONSTANT)
    smooth = np.where(obj_mask, num / np.maximum(den, 1e-12), 0)

    # robust threshold from the object's own intensities
    values = smooth[obj_mask]
    median = np.median(values)
    robust_std = 1.4826 * np.median(np.abs(values - median))
    # floor of one gray level, so flat objects don't turn noise into spots
    threshold = median + k * max(robust_std, 1.0)

    # local maxima restricted to the object
    coords = skifeat.peak_local_max(smooth,
                                    min_distance=min_distance,
                                    threshold_abs=threshold,
                                    labels=obj_mask.astype(np.int32),
                                    exclude_border=False
                                    )

    spot_count = len(coords)
    obj_area = int(obj_mask.sum())

    # mean spot intensity relative to the object's median
    if spot_count > 0 and median > 0:
        spot_contrast = float(smooth[coords[:, 0], coords[:, 1]].mean() / median)
    else:
        spot_contrast = np.nan

    spots_dict = {}
    spots_dict[f"{prefix}_spot_count"] = spot_count
    spots_dict[f"{prefix}_spot_density"] = spot_count / obj_area
    spots_dict[f"{prefix}_spot_contrast"] = spot_contrast
    spot_sizes = get_spot_sizes(smooth=smooth,
                                obj_mask=obj_mask,
                                coords=coords,
                                median=median
                                )
    # wrapped in an outer list, so a one-row dataframe keeps the whole list in a single cell
    spots_dict[f"{prefix}_spot_sizes"] = [spot_sizes]

    return spots_dict
