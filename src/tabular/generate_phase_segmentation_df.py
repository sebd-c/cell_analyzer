# generate phase segmentation dfs module

print('initializing...')  # noqa

# Code destined to generating
# segmentation mask based df
# when only the phase (rgb) image
# and its mask are available.
# Masks are used as given (no relabeling,
# cleanup or size filtering), since
# they are human generated.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
import argparse as ap
import cv2 as cv
import pandas as pd
import numpy as np
from os.path import join
import tifffile
from scipy import ndimage as ndi
from src._io import make_contour_label
from src.tabular._texture_features import get_intensity_features
from src.tabular._texture_features import get_glcm_features
from src.tabular._texture_features import run_lbp_metrics
from src.tabular._texture_features import get_local_texture_features
from src.tabular._texture_features import get_spot_features
from src.tabular._geometric_features import get_morpho_features
from src.tabular.generate_segmentation_df import COLOR
from src.imaging._img_subprocess import get_unique_ids
from src._execution_formatting import enter_to_continue
from src._execution_formatting import print_progress_message
from src._execution_formatting import get_files_in_folder
from src._execution_formatting import print_execution_parameters


print('all required libraries successfully imported.')  # noqa

# objects with this many pixels or fewer are excluded
MIN_OBJECT_PIXELS = 10

# pixels kept around each object's bounding box when cropping it
# (features only use the object's own pixels, so cropping doesn't change them)
CROP_MARGIN = 1


#####################################################################
def process_single_contour(single_contour_img: np.ndarray,
                           mask_name: str,
                           pixint: float,
                           channels_dict: dict,
                           overlay_img: np.ndarray,
                           offset: tuple = (0, 0),
                           skip_texture: bool = False
                           ) -> pd.DataFrame | None:
    """
    Given an array of a single contour and the phase channels
    (both cropped around the object, offset being the (x, y) of the crop's corner),
    returns a single row df with the object's morphological,
    intensity, texture and spot features (one set per channel),
    and draws the object's label on the overlay img.
    If skip_texture, only morphological and intensity features are computed.
    Returns None for objects too small (or thin) to measure.
    """
    # converting int type
    single_contour_img = single_contour_img.astype(np.uint8)

    # excluding too small objects
    if single_contour_img.sum() <= MIN_OBJECT_PIXELS:
        print(f'\nobject {int(pixint)} of {mask_name} has {MIN_OBJECT_PIXELS} pixels or fewer, excluded')
        return None

    # finding contours in image
    # (offset puts the contour back in whole image coords)
    contours, _ = cv.findContours(single_contour_img,
                                  cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_NONE,
                                  offset=offset)

    # a label drawn in separate pieces uses its largest piece for shape features
    # (intensity and texture features still use all of its pixels)
    contour = max(contours, key=cv.contourArea)

    # getting current contour area
    area = cv.contourArea(contour)

    # holder for all features
    single_contour_dict = {}

    # calculate morphometric features to extract
    # (objects with no measurable shape, e.g. a 1 pixel wide line, are excluded)
    try:
        morpho_dict = get_morpho_features(contour=contour,
                                          pixel_int=pixint,
                                          mask_name=mask_name,
                                          area=area)
    except (ZeroDivisionError, ValueError, cv.error):
        print(f'\nobject {int(pixint)} of {mask_name} has no measurable shape, excluded')
        return None
    single_contour_dict.update(morpho_dict)

    # loop to get features per channel
    for channel_key, channel_img in channels_dict.items():
        # calculate pixel intensity features to extract
        intensity_dict = get_intensity_features(area=area,
                                                image=channel_img,
                                                mask_image=single_contour_img,
                                                prefix=channel_key
                                                )
        single_contour_dict.update(intensity_dict)

        # texture (lbp, glcm, local) and spot features are skipped if requested
        if skip_texture:
            continue

        # get lbp textural parameters
        lbp_dict = run_lbp_metrics(image=channel_img,
                                   mask=single_contour_img,
                                   prefix=channel_key
                                   )

        # get glcm textural parameters
        glcm_dict = get_glcm_features(image=channel_img,
                                      mask=single_contour_img,
                                      levels=32,
                                      distances=[3, 6, 9, 18],
                                      angles_deg=[0, 45, 90, 135],
                                      prefix=channel_key
                                      )

        # get local texture parameters
        local_dict = get_local_texture_features(image=channel_img,
                                                mask=single_contour_img,
                                                radius=3,
                                                levels=32,
                                                prefix=channel_key
                                                )

        # get spots parameters
        # all channels are brightfield, so it looks for dark spots
        spots_dict = get_spot_features(image=channel_img,
                                       mask=single_contour_img,
                                       prefix=channel_key,
                                       invert=True
                                       )

        # update single contour dict
        single_contour_dict.update(lbp_dict)
        single_contour_dict.update(glcm_dict)
        single_contour_dict.update(local_dict)
        single_contour_dict.update(spots_dict)

    # assembling contour df, i.e, making a row
    single_contour_df = pd.DataFrame(single_contour_dict, index=[0])  # noqa

    # put label
    make_contour_label(contour_index=int(pixint),
                       centroid_x=single_contour_df['cx_coords'].iloc[0],
                       centroid_y=single_contour_df['cy_coords'].iloc[0],
                       color=COLOR,
                       thickness=2,
                       img_to_label=overlay_img,
                       contour=contour,
                       )

    # returning the contour row
    return single_contour_df


def make_image_contours_df(mask_name: str,
                           mask_path: str,
                           p_img_path: str,
                           overlays_output_folder: str,
                           skip_texture: bool = False
                           ) -> pd.DataFrame | None:
    """
    Given a path to a mask and its phase (rgb) image,
    finds contours and returns a df
    containing the contours features.
    """
    # read the rgb img (cv2 reads it as bgr)
    phase_image_cv2 = cv.imread(p_img_path,
                                -1)

    # convert it to the accurate colors
    phase_image = cv.cvtColor(phase_image_cv2, cv.COLOR_BGR2RGB)

    # separate channels
    phase_red, phase_green, phase_blue = cv.split(phase_image)

    # grayscale version of the phase img (replaces the og grayscale img)
    phase_gray = cv.cvtColor(phase_image, cv.COLOR_RGB2GRAY)

    # dict to loop between channels
    # ("gray" instead of "grayscale", since it comes from the phase img, not the og one)
    channels_dict = {'gray': phase_gray,
                     'red': phase_red,
                     'green': phase_green,
                     'blue': phase_blue
                     }

    # copy of the phase img to draw outlines on
    overlay_image = phase_image_cv2.copy()

    # reading mask (used as given, each label is an object)
    mask = tifffile.imread(mask_path)

    # getting ids of objects in mask
    valid_pixint_list = get_unique_ids(mask)

    # bounding box of each object (index = id - 1)
    object_slices = ndi.find_objects(mask)

    # create an empty list to hold the single contours df
    contours_df_list = []

    # loop not to join different contours
    for pixel_intensity in valid_pixint_list:
        # cropping around the object, as features are computed per object
        rows, cols = object_slices[pixel_intensity - 1]
        rows = slice(max(rows.start - CROP_MARGIN, 0), rows.stop + CROP_MARGIN)
        cols = slice(max(cols.start - CROP_MARGIN, 0), cols.stop + CROP_MARGIN)
        crop_channels_dict = {channel_key: channel_img[rows, cols]
                              for channel_key, channel_img in channels_dict.items()}

        # single object binary img
        single_contour_img = (mask[rows, cols] == pixel_intensity).astype(np.uint8)

        contour_df = process_single_contour(single_contour_img=single_contour_img,
                                            mask_name=mask_name,
                                            pixint=pixel_intensity,
                                            channels_dict=crop_channels_dict,
                                            overlay_img=overlay_image,
                                            offset=(cols.start, rows.start),
                                            skip_texture=skip_texture
                                            )

        # appending current df to dfs list
        if contour_df is not None:
            contours_df_list.append(contour_df)

    # save image with labels
    overlays_output_path = join(overlays_output_folder, mask_name)
    cv.imwrite(overlays_output_path, overlay_image)

    # images without valid objects return None
    if not contours_df_list:
        print(f'\nno valid objects in mask {mask_name}, skipping')
        return None

    # concat the list of dfs into a single df
    concat_contours_df = pd.concat(contours_df_list, ignore_index=True)

    # returning contours df
    return concat_contours_df


def make_folder_contours_df(masks_input_folder: str,
                            phase_img_folder: str,
                            masks_img_extension: str,
                            mask_suffix: str,
                            csv_output_folder: str,
                            overlays_output_folder: str,
                            skip_texture: bool = False
                            ) -> None:
    """
    Given a path to a folder containing masks
    and a folder containing the respective phase imgs
    (linked by name, the mask name being the phase img name
    plus mask_suffix, e.g. img.tif and img_masks.tif), generates a df containing
    the wanted information, and saves the results
    in the output folder.
    """
    # getting masks files in respective input folder
    mask_files = get_files_in_folder(path_to_folder=masks_input_folder,
                                     extension=masks_img_extension)

    # analogous to phase imgs
    phase_files = get_files_in_folder(path_to_folder=phase_img_folder,
                                      extension=masks_img_extension)

    masks_files_num = len(mask_files)

    # create empty list to hold the dfs
    dfs_list = []

    # iterating over mask files
    for file_index, mask_file in enumerate(mask_files, 1):
        # printing progress message
        base_string = 'generating phase segmentation df #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=masks_files_num)

        # getting respective phase img name (mask name without the suffix)
        mask_base_name = mask_file[:-len(masks_img_extension)]
        if mask_suffix and mask_base_name.endswith(mask_suffix):
            mask_base_name = mask_base_name[:-len(mask_suffix)]
        phase_file = mask_base_name + masks_img_extension

        # skipping masks without a matching phase img
        if phase_file not in phase_files:
            print(f'\nno matching phase image ({phase_file}) for {mask_file}, skipping')
            continue

        # getting current image mask and phase img paths
        mask_input_path = join(masks_input_folder,
                               mask_file)
        phase_img_input_path = join(phase_img_folder,
                                    phase_file)

        # get image contour df and respective overlay-ed imgs
        # (named after the phase img)
        image_df = make_image_contours_df(mask_name=phase_file,
                                          mask_path=mask_input_path,
                                          p_img_path=phase_img_input_path,
                                          overlays_output_folder=overlays_output_folder,
                                          skip_texture=skip_texture
                                          )

        # append current img df to dir df
        if image_df is not None:
            dfs_list.append(image_df)

    # concatenating dfs into a single df
    contour_df = pd.concat(dfs_list, ignore_index=True)

    # create the path to save the output path
    output_path = join(csv_output_folder,
                       'contours_df.pickle')

    # saving df
    contour_df.to_pickle(output_path)

    # printing execution message
    print(f'\noutput saved to {csv_output_folder}')
    print('analysis complete!')


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'generate phase segmentation df module (rgb img + mask only)'

    # creating a parser instance
    parser = ap.ArgumentParser(description=description)

    # adding arguments to parser

    # phase input folder param
    parser.add_argument('-p', '--phase-images-folder',
                        dest='phase_folder',
                        required=True,
                        help='defines path to folder containing phase (rgb) images')

    # input masks folder param
    parser.add_argument('-m', '--masks-folder',
                        dest='masks_folder',
                        required=True,
                        help='defines path to folder containing grayscale masks')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # mask suffix param
    parser.add_argument('-ms', '--mask-suffix',
                        dest='mask_suffix',
                        required=False,
                        default='',
                        help='defines suffix added to the phase img name in the mask name '
                             '(e.g. "_masks" for img.tif and img_masks.tif)')

    # csv output folder param
    parser.add_argument('-co', '--csv_output_folder',
                        dest='csv_output_folder',
                        required=True,
                        help='defines path to output folder (csv)')

    # overlays output folder param
    parser.add_argument('-oo', '--overlays_output_folder',
                        dest='overlays_output_folder',
                        required=True,
                        help='defines path to output folder (overlays)')

    # skip texture param
    parser.add_argument('-st', '--skip-texture',
                        dest='skip_texture',
                        action='store_true',
                        help='computes only morphological and intensity features '
                             '(skips lbp, glcm, local texture and spot features)')

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

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to get the segmentation df for a folder
    make_folder_contours_df(masks_input_folder=args_dict['masks_folder'],
                            phase_img_folder=args_dict['phase_folder'],
                            masks_img_extension=args_dict['images_extension'],
                            mask_suffix=args_dict['mask_suffix'],
                            csv_output_folder=args_dict['csv_output_folder'],
                            overlays_output_folder=args_dict['overlays_output_folder'],
                            skip_texture=args_dict['skip_texture']
                            )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
