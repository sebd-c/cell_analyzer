###########################################################################################
# imports
from os.path import join
import cv2 as cv
import numpy as np
from argparse import ArgumentParser
from src._execution_formatting import enter_to_continue
from src._execution_formatting import get_files_in_folder
from src._execution_formatting import print_progress_message
from src._execution_formatting import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing


def enhance_single_img(og_img_path: str,
                       output_path: str,
                       clip_limit: int,
                       tile_size: int,
                       lower_value: float,
                       upper_value: float
                       ) -> None:
    """
    This function takes the path to a poor quality image,
    and saves a new image enhanced by CLAHE
    """
    img = cv.imread(og_img_path, cv.IMREAD_GRAYSCALE)

    # batch-wide contrast stretch to 0-255
    rescaled_img = normalize_img(img, lower_value, upper_value)

    # saving only the stretched img if CLAHE was skipped (auto mode)
    if clip_limit is None:
        cv.imwrite(output_path, rescaled_img)
        return

    # apply histogram equalization
    clip_limit = float(clip_limit)
    tile_size = int(tile_size)
    # create a CLAHE object
    clahe = cv.createCLAHE(clipLimit=clip_limit,
                           tileGridSize=(tile_size, tile_size)
                           )

    # apply in normalized image
    cl1 = clahe.apply(rescaled_img)

    # save clahe-d img
    cv.imwrite(output_path, cl1)

    return


def get_batch_stretch_limits(input_folder: str,
                             img_extension: str
                             ) -> tuple:
    """
    Returns the (lower, upper) stretch limits extracted from the data:
    the darkest and brightest real signal over ALL imgs in the folder,
    so every img gets the same stretch and intensities stay comparable between imgs.
    Each img is 3x3 median filtered before taking its min/max, so isolated
    hot/dead pixels don't set the limits (only those few pixels get clipped).
    """
    # getting img files in respective input folder
    img_files = get_files_in_folder(path_to_folder=input_folder,
                                    extension=img_extension)

    # starting limits outside the uint8 range
    lower_value = 255
    upper_value = 0

    # looping through imgs, updating the batch-wide limits
    for img_file in img_files:
        img = cv.imread(join(input_folder, img_file), cv.IMREAD_GRAYSCALE)

        # removing isolated hot/dead pixels, keeping real structures
        filtered_img = cv.medianBlur(img, 3)

        lower_value = min(lower_value, int(filtered_img.min()))
        upper_value = max(upper_value, int(filtered_img.max()))

    return lower_value, upper_value


def normalize_img(img: np.ndarray,
                  lower_value: float,
                  upper_value: float
                  ) -> np.ndarray:
    """
    Linear contrast stretch: maps lower_value/upper_value
    to 0/255 (clipping outside), returning an uint8 img.
    The limits come from the whole batch (get_batch_stretch_limits),
    so the same stretch is applied to every img.
    """
    # scaling to 0-255, using a small epsilon to avoid division by zero
    rescaled_img = 255.0 * (img.astype(float) - lower_value) / (upper_value - lower_value + 1e-6)

    # clipping values outside the limits
    rescaled_img = np.clip(rescaled_img, 0, 255)

    # rounding (instead of truncating) so dim imgs aren't biased downwards
    return np.round(rescaled_img).astype(np.uint8)


def get_objects_and_background_masks(img: np.ndarray,
                                     noise_factor: float = 4,
                                     margin: int = 41
                                     ) -> tuple:
    """
    Given an uint8 img with objects brighter than the background,
    returns rough boolean objects and background masks.
    Objects are pixels clearly above background noise (including dim parts),
    background is everything farther than a margin from any object.
    Only used to score contrast, not as a segmentation.
    """
    # blur to keep noise out of the threshold
    blurred_img = cv.GaussianBlur(img, (0, 0), 3).astype(float)

    # robust background level and noise (median / MAD)
    background_level = np.median(blurred_img)
    noise = 1.4826 * np.median(np.abs(blurred_img - background_level))

    # objects: anything clearly above background noise
    objects_mask = blurred_img > background_level + noise_factor * noise

    # removing specks
    objects_mask = cv.morphologyEx(objects_mask.astype(np.uint8),
                                   cv.MORPH_OPEN,
                                   np.ones((5, 5), np.uint8)) > 0

    # background: away from objects, so faint object borders don't count as background
    near_objects = cv.dilate(objects_mask.astype(np.uint8),
                             np.ones((margin, margin), np.uint8)) > 0
    background_mask = ~near_objects

    return objects_mask, background_mask


def get_cnr(img: np.ndarray,
            objects_mask: np.ndarray,
            background_mask: np.ndarray
            ) -> float:
    """
    Returns the contrast-to-noise ratio between objects and background:
    (objects median - background mean) / background std.
    Median is used so the dim parts of objects (most of their pixels) drive the score.
    Rises when objects separate from background,
    falls when background noise gets amplified.
    """
    objects_px = img[objects_mask]
    background_px = img[background_mask]

    cnr = (np.median(objects_px) - np.mean(background_px)) / (np.std(background_px) + 1e-6)

    return float(cnr)


def estimate_tile_size(objects_masks: list,
                       img_width: int
                       ) -> int:
    """
    Estimates the CLAHE grid size (number of tiles per side)
    so that each tile is ~4 object diameters wide,
    based on the area-weighted median object size in the rough objects masks
    (weighted so leftover specks don't dominate).
    """
    # gathering object areas from all masks
    areas = []
    for objects_mask in objects_masks:
        _, _, stats, _ = cv.connectedComponentsWithStats(objects_mask.astype(np.uint8))
        # skipping label 0 (background)
        areas.extend(stats[1:, cv.CC_STAT_AREA])

    # falling back to opencv's default if no objects were found
    if len(areas) == 0:
        return 8

    # area-weighted median: size of the object a typical object pixel belongs to
    sorted_areas = np.sort(areas)
    cumulative_weight = np.cumsum(sorted_areas) / np.sum(sorted_areas)
    weighted_median_area = sorted_areas[np.searchsorted(cumulative_weight, 0.5)]

    # equivalent diameter of that object
    median_diameter = 2 * np.sqrt(weighted_median_area / np.pi)

    tile_size = round(img_width / (4 * median_diameter))

    return int(np.clip(tile_size, 2, 64))


def find_best_clahe_params(input_folder: str,
                           img_extension: str,
                           lower_value: float,
                           upper_value: float,
                           max_samples: int = 10
                           ) -> tuple:
    """
    Finds a single (clip_limit, tile_size) pair for the whole folder,
    maximizing the mean contrast-to-noise ratio over a sample of its imgs,
    so that all imgs are enhanced consistently.
    Returns (None, None) if no combination beats the plain stretch (no CLAHE).
    """
    # getting img files in respective input folder
    img_files = get_files_in_folder(path_to_folder=input_folder,
                                    extension=img_extension)

    # evenly spaced sample of imgs to score on
    sample_indices = np.linspace(0, len(img_files) - 1,
                                 num=min(max_samples, len(img_files)),
                                 dtype=int)
    sample_files = [img_files[index] for index in np.unique(sample_indices)]

    # reading and normalizing sample imgs, and getting their rough objects/background masks
    normalized_imgs = []
    objects_masks = []
    background_masks = []
    for img_file in sample_files:
        img = cv.imread(join(input_folder, img_file), cv.IMREAD_GRAYSCALE)
        normalized_img = normalize_img(img, lower_value, upper_value)
        objects_mask, background_mask = get_objects_and_background_masks(normalized_img)
        normalized_imgs.append(normalized_img)
        objects_masks.append(objects_mask)
        background_masks.append(background_mask)

    # estimating tile size from object size, and searching around it
    estimated_tile_size = estimate_tile_size(objects_masks=objects_masks,
                                             img_width=normalized_imgs[0].shape[1])
    tile_sizes = sorted({max(2, round(estimated_tile_size * factor))
                         for factor in (0.5, 1, 1.5, 2)})
    clip_limits = np.arange(1.0, 4.01, 0.5)

    print(f'\nsearching CLAHE params on {len(sample_files)} imgs '
          f'(estimated tile size: {estimated_tile_size})...')

    # baseline score, without CLAHE
    baseline_cnr = np.mean([get_cnr(normalized_img, objects_mask, background_mask)
                            for normalized_img, objects_mask, background_mask
                            in zip(normalized_imgs, objects_masks, background_masks)])

    # scoring every combination by mean cnr over sample imgs
    scores = []
    for tile_size in tile_sizes:
        for clip_limit in clip_limits:
            clahe = cv.createCLAHE(clipLimit=float(clip_limit),
                                   tileGridSize=(tile_size, tile_size))
            cnrs = [get_cnr(clahe.apply(normalized_img), objects_mask, background_mask)
                    for normalized_img, objects_mask, background_mask
                    in zip(normalized_imgs, objects_masks, background_masks)]
            scores.append((float(np.mean(cnrs)), float(clip_limit), tile_size))

    # printing the best combinations
    scores.sort(reverse=True)
    print(f'mean CNR without CLAHE: {baseline_cnr:.3f}')
    print('top params (mean CNR | clip limit | tile size):')
    for score, clip_limit, tile_size in scores[:5]:
        print(f'  {score:.3f} | {clip_limit} | {tile_size}')

    best_score, best_clip_limit, best_tile_size = scores[0]

    # skipping CLAHE if no combination beats the plain stretch
    if best_score <= baseline_cnr:
        print('no CLAHE params beat the baseline, skipping CLAHE')
        return None, None

    print(f'using clip limit {best_clip_limit} and tile size {best_tile_size}')

    return best_clip_limit, best_tile_size


def enhance_dir_imgs(input_folder: str,
                     output_folder: str,
                     img_extension: str,
                     clip_limit: int,
                     tile_size: int,
                     lower_value: float,
                     upper_value: float
                     ) -> None:
    """
    This function takes an input folder containing all the
    imgs to be processed, and loops through each image processing it
    """
    # getting img files in respective input folder
    img_files = get_files_in_folder(path_to_folder=input_folder,
                                    extension=img_extension)

    # iterate the processing through every img
    for file_index, img_file in enumerate(img_files, 1):
        # printing progress message
        base_string = 'generating processed img #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=len(img_files))

        # getting current image mask input/output paths
        img_input_path = join(input_folder,
                              img_file)

        # create the path to save the output path
        output_path = join(output_folder,
                           img_file)

        # running img processing func
        enhance_single_img(og_img_path=img_input_path,
                           output_path=output_path,
                           clip_limit=clip_limit,
                           tile_size=tile_size,
                           lower_value=lower_value,
                           upper_value=upper_value)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('processing complete!')

    return


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'preprocess images in grayscale'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input_folder',
                        dest='input_folder',
                        required=True,
                        help='defines path to folder containing grayscale images')

    # images extension param
    parser.add_argument('-x', '--images_extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # images extension param
    parser.add_argument('-c', '--clip-limit',
                        dest='clip_limit',
                        required=False,
                        default=2,
                        help='defines clip limit size for open CVs CLAHE')

    # images extension param
    parser.add_argument('-t', '--tile-size',
                        dest='tile_size',
                        required=False,
                        default=8,
                        help='defines tile size to grid size in open CVs CLAHE')

    # auto params flag
    parser.add_argument('-a', '--auto',
                        dest='auto',
                        action='store_true',
                        help='finds clip limit and tile size automatically '
                             '(ignores -c and -t), by maximizing objects contrast-to-noise ratio')

    # no clahe flag
    parser.add_argument('-n', '--no-clahe',
                        dest='no_clahe',
                        action='store_true',
                        help='only applies the batch-wide linear stretch (ignores -c, -t and -a). '
                             'use it when intensities must stay comparable between imgs')

    # output folder param
    parser.add_argument('-o', '--output_folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder to save processed imgs')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting images folder
    input_folder = args_dict['input_folder']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting clip limit
    clip_limit = args_dict['clip_limit']

    # getting tile size
    tile_size = args_dict['tile_size']

    # getting csv output path
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # getting a single stretch for the whole batch, so intensities stay comparable
    lower_value, upper_value = get_batch_stretch_limits(input_folder=input_folder,
                                                        img_extension=images_extension)
    print(f'\nbatch stretch: intensity {lower_value} -> 0, {upper_value} -> 255')

    # finding params automatically, if requested
    if args_dict['auto'] and not args_dict['no_clahe']:
        clip_limit, tile_size = find_best_clahe_params(input_folder=input_folder,
                                                       img_extension=images_extension,
                                                       lower_value=lower_value,
                                                       upper_value=upper_value)

    # skipping CLAHE, if requested
    if args_dict['no_clahe']:
        clip_limit, tile_size = None, None

    # CLAHE is local and non-linear, so it changes relative intensities between imgs
    if clip_limit is not None:
        print('WARNING: CLAHE will be applied, intensities will NOT be comparable between imgs '
              '(use -n/--no-clahe for quantification)')

    # running function to preprocess images in a folder
    enhance_dir_imgs(input_folder=input_folder,
                     output_folder=output_folder,
                     img_extension=images_extension,
                     clip_limit=clip_limit,
                     tile_size=tile_size,
                     lower_value=lower_value,
                     upper_value=upper_value)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
