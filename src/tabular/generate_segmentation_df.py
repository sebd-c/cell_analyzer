# generate segmentation dfs module
from typing import Any

print('initializing...')  # noqa

# Code destined to generating
# segmentation mask based df.

######################################################################
# imports
# variable bgr
COLOR = (0, 255, 0)
# importing required libraries
print('importing required libraries...')  # noqa
import argparse as ap
import cv2 as cv
import pandas as pd
import numpy as np
from skimage import measure as skimeas
from skimage import morphology as skimorph
from src.tabular._texture_features import get_intensity_features
from src.tabular._geometric_features import get_morpho_features
from os.path import join
import tifffile
from src._io import make_contour_label
from src.tabular._texture_features import get_glcm_features
from src.tabular._texture_features import run_lbp_metrics
from src.imaging._img_subprocess import get_unique_ids
from src._execution_formatting import enter_to_continue
from src._execution_formatting import print_progress_message
from src._execution_formatting import get_files_in_folder
from src._execution_formatting import print_execution_parameters


print('all required libraries successfully imported.')  # noqa


#####################################################################
def process_single_contour(single_contour_img: np.ndarray,
                           mask_name: str,
                           pixint: float,
                           image: np.ndarray,
                           phase_rgb: np.ndarray,
                           phase_red: np.ndarray,
                           phase_green: np.ndarray,
                           phase_blue: np.ndarray
                           ) -> pd.DataFrame | None:
    """
    Given an array of a single contour
    :param pixint:
    :param mask_name:
    :param single_contour_img:
    :param image:
    :param phase_red:
    :param phase_green:
    :param phase_blue:
    :return:
    """

    # converting int type
    single_contour_img = single_contour_img.astype(np.uint8)

    # finding contour in image
    contour, _ = cv.findContours(single_contour_img,
                                 cv.RETR_EXTERNAL,
                                 cv.CHAIN_APPROX_NONE)

    # getting current contour area
    area = cv.contourArea(contour[0])
    print(area)

    # holder for all dicts
    single_contour_dict = {}

    # dict to loop between channels
    channels_dict = {'grayscale': image,
                     'red': phase_red,
                     'green': phase_green,
                     'blue': phase_blue
                     }
    # check if the object is valid
    # TODO: check why this is happening and if unavoidable, put in argparser
    if area > 10:
        # calculate morphometric features to extract
        morpho_dict = get_morpho_features(contour=contour[0],
                                          pixel_int=pixint,
                                          mask_name=mask_name,
                                          area=area)
        # list to update dicts per channel
        list_intensity_dict = {}
        list_lbp_dict = {}
        list_glcm_dict = {}

        # loop to get different dict channels
        for channel_key in channels_dict.keys():
            # calculate pixel intensity features to extract
            intensity_dict = get_intensity_features(area=area,
                                                    image=channels_dict[channel_key],
                                                    mask_image=single_contour_img,
                                                    prefix=str(channel_key)
                                                    )

            # get lbp textural parameters
            lbp_dict = run_lbp_metrics(image=channels_dict[channel_key],
                                       contour=contour[0],
                                       prefix=str(channel_key)
                                       )

            # get glcm textural parameters
            glcm_dict = get_glcm_features(image=channels_dict[channel_key],
                                          mask=single_contour_img,
                                          levels=32,
                                          distances=[3, 6, 9, 18],
                                          angles_deg=[0, 45, 90, 135],
                                          prefix=str(channel_key)
                                          )

            # update dicts
            list_intensity_dict.update(intensity_dict)
            list_lbp_dict.update(lbp_dict)
            list_glcm_dict.update(glcm_dict)

        # update single contour dict with
        # morpho dict and intensity and texture dicts
        single_contour_dict.update(morpho_dict)
        single_contour_dict.update(list_intensity_dict)
        single_contour_dict.update(list_lbp_dict)
        single_contour_dict.update(list_glcm_dict)

        # assembling contour df, i.e, making a row
        single_contour_df = pd.DataFrame(single_contour_dict, index=[0])  # noqa

        # put label
        make_contour_label(contour_index=int(pixint),
                           centroid_x=single_contour_df['cx_coords'].iloc[0],
                           centroid_y=single_contour_df['cy_coords'].iloc[0],
                           color=COLOR,
                           thickness=2,
                           img_to_label=phase_rgb,
                           contour=contour[0],
                           )

        # returns the contours and the list of intensities
        return single_contour_df
    return None


# module specific aux functions
def make_image_contours_df(mask_name: str,
                           mask_path: str,
                           og_img_path: str,
                           p_img_path: str,
                           overlays_output_folder: str
                           ) -> pd.DataFrame:
    """
    Given a path to a binary image,
    finds contours and
    returns data frame containing
    contours desired infos.
    """

    # read the rgb channel
    phase_image_cv2 = cv.imread(p_img_path,
                                -1)

    # convert it to the accurate colors
    phase_image = cv.cvtColor(phase_image_cv2, cv.COLOR_BGR2RGB)

    # separate channels
    phase_red, phase_green, phase_blue = cv.split(phase_image)

    # reading grayscale image
    image = cv.imread(og_img_path,
                   -1)

    # reading mask
    mask = tifffile.imread(mask_path)

    # relabel img as to separate loose pixels
    mask = skimeas.label(mask)

    # # make it uint8 for morphological operations
    mask = mask.astype(np.uint8)

    # define kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # remove small objects
    mask = skimorph.remove_small_objects(mask)

    # close small holes (same library was not used for need of int types instead of booleans)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # test output
    overlays_output_path = join(overlays_output_folder, mask_name)
    cv.imwrite(overlays_output_path, mask)

    # getting intensity range to
    # separate contours before binarizing mask
    valid_pixint_list = get_unique_ids(mask)
    # max_intensity = mask.max()  # noqa

    # getting shape for new masks arrays
    shape = mask.shape

    # create an empty list to hold the single contours df
    contours_df_list = []

    # loop not to join different contours
    for pixel_intensity in valid_pixint_list:
        # create a new blank img
        single_contour_img = np.zeros(shape)

        # putting the contour inside
        single_contour_img[mask == pixel_intensity] = 1

        flattened_contour_img = single_contour_img.flatten()
        unique_val, counts = np.unique(flattened_contour_img, return_counts=True)
        print(dict(zip(unique_val, counts)))

        # choose which way the function should be continued
        # if the parameter for phase was set,
        # means the analysis is being done in phase img,
        # and requires extraction of the rgb channels
        contour_df = process_single_contour(single_contour_img=single_contour_img,
                                            mask_name=mask_name,
                                            pixint=pixel_intensity,
                                            image=image,
                                            phase_rgb=phase_image,
                                            phase_red=phase_red,
                                            phase_green=phase_green,
                                            phase_blue=phase_blue
                                            )

        # appending current df to dfs list
        contours_df_list.append(contour_df)

    # concat the list of dfs into a single df
    concat_contours_df = pd.concat(contours_df_list, ignore_index=True)

    # save image with labels
    overlays_output_path = join(overlays_output_folder, mask_name)

    cv.imwrite(overlays_output_path, phase_image)

    # returning contours df
    return concat_contours_df


def make_folder_contours_df(masks_input_folder: str,
                            og_img_folder: str,
                            phase_img_folder: str,
                            masks_img_extension: str,
                            csv_output_folder: str,
                            overlays_output_folder: str,
                            ) -> None:
    """
    Given a path to a folder containing
    cytoplasm's masks, generates a df
    containing the wanted information,
    and saving the results
    in the output folder.
    """
    # getting masks files in respective input folder
    mask_files = get_files_in_folder(path_to_folder=masks_input_folder,
                                     extension=masks_img_extension)

    # analogous to og imgs
    og_files = get_files_in_folder(path_to_folder=og_img_folder,
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
        base_string = 'generating segmentation df #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=masks_files_num)

        # getting respective og img
        og_img_file = og_files[file_index - 1]

        # getting respective phase img
        phase_file = phase_files[file_index - 1]

        # getting current image mask input/output paths
        mask_input_path = join(masks_input_folder,
                               mask_file)

        # analogous getting current og image input/output paths
        og_img_input_path = join(og_img_folder,
                                 og_img_file
                                 )

        # analogous getting current og image input/output paths
        phase_img_input_path = join(phase_img_folder,
                                    phase_file
                                    )
        # get image contour df and respective overlay-ed imgs
        image_df = make_image_contours_df(mask_name=mask_file,
                                          mask_path=mask_input_path,
                                          og_img_path=og_img_input_path,
                                          p_img_path=phase_img_input_path,
                                          overlays_output_folder=overlays_output_folder
                                          )

        # append current img df to dir df
        dfs_list.append(image_df)

    # concatenating "dfs" from dfs lists into
    # a pandas dataframe
    contour_df = pd.concat(dfs_list, ignore_index=True)

    # create the path to save the output path
    output_path = join(csv_output_folder,
                       'contours_df.pickle')

    # saving df
    contour_df.to_pickle(output_path)

    # printing execution message
    print(f'output saved to {csv_output_folder}')
    print('analysis complete!')

    return


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'generate segmentation df module'

    # creating a parser instance
    parser = ap.ArgumentParser(description=description)

    # adding arguments to parser

    # grayscale input folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing original clahed images')
    # phase input folder param
    parser.add_argument('-p', '--phase-images-folder',
                        dest='phase_folder',
                        required=True,
                        help='defines path to folder containing phase images')

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

    # # overlays output folder param
    # parser.add_argument('-st', '--segmentation_type',
    #                     dest='segmentation_type',
    #                     required=True,
    #                     help='defines which type of segmentation it is. "nuc" or "cyto"')

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

    # getting masks folder
    masks_folder = args_dict['masks_folder']

    # getting images folder
    images_folder = args_dict['images_folder']

    # phase images folder
    phase_folder = args_dict['phase_folder']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting csv output path
    csv_output_folder = args_dict['csv_output_folder']

    # getting overlays output path
    overlays_output_folder = args_dict['overlays_output_folder']

    # # getting type of segmentation
    # segmentation_type = args_dict['segmentation_type']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to get the segmentation df for a folder
    make_folder_contours_df(masks_input_folder=masks_folder,
                            og_img_folder=images_folder,
                            phase_img_folder=phase_folder,
                            masks_img_extension=images_extension,
                            csv_output_folder=csv_output_folder,
                            overlays_output_folder=overlays_output_folder,
                            )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
