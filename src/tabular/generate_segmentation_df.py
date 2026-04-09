# generate segmentation dfs module
import numpy as np
from sqlalchemy.util import to_list

print('initializing...')  # noqa

# Code destined to generating
# segmentation mask based df.

######################################################################
# imports

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
from src.utils.aux_funcs import enter_to_continue, mask_image
from src.utils.aux_funcs import get_contour_centroid
from src.utils.aux_funcs import get_area_box
from src.utils.aux_funcs import get_contour_rratio
from src.utils.aux_funcs import get_contour_ellipse_feats
from src.utils.aux_funcs import get_contour_roundness
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import make_contour_label
from src.utils.aux_funcs import get_unique_ids
from src.tabular._texture_features import run_lbp_metrics
from src.tabular._texture_features import get_glcm_features


print('all required libraries successfully imported.')  # noqa


#####################################################################
def get_df_features(single_contour_image: np.ndarray,
                    contour: np.ndarray,
                    pixel_int: float,
                    mask_name: str,
                    pixint_list: list,
                    phase_red_list: list or None = None,
                    phase_green_list: list or None = None,
                    phase_blue_list: list or None = None
                    ) -> pd.DataFrame:
    """
    Given a single contour,
    calculates and adresses desired values to columns
    and returns a single row dataframe of that
    :param contour: contour from cv2
    :param pixel_int: pixel intensity from the original mask, will be used as an indexer
    :param mask_name: image name, used to address and properly fill the df
    :param flag: comes from the processing function, 0 if used in grayscale and 1 if used in phase
    :param pixint_list: list of pixel intensities of the grayscale channel
    :param phase_red_list: list of pixel intensities of the R channel from the RGB (phase img)
    :param phase_green_list: list of pixel intensities of the G channel from the RGB (phase img)
    :param phase_blue_list: list of pixel intensities of the B channel from the RGB (phase img)
    :return: single row df respective to the contour passed
    """

    # getting current contour area
    area = cv.contourArea(contour)
    print(area)

    # TODO: check why this is happening and if unavoidable, put in argparser
    if area > 10:

        # calculate morphometric features to extract
        morpho_dict = get_morpho_features(contour=contour,
                                          pixel_int=pixel_int,
                                          mask_name=mask_name,
                                          area=area)

        # calculate pixel intensity features to extract
        intensity_dict = get_intensity_features(area=area,
                                                image=pixint_list,
                                                mask_image=phase_red_list,
                                                prefix=phase_green_list,
                                                )
        # get textural parameters of contour in different phase channel
        single_lbp_df = run_lbp_metrics(image=phase_red,
                                        contour=contour[0]
                                        )

        single_glcm_df = get_glcm_features(image=phase_red,
                                           mask=single_contour_img,
                                           levels=32,
                                           distances=[3, 6, 9, 18],
                                           angles_deg=[0, 45, 90, 135])
        morpho_dict.update(intensity_dict)

    else:
        contour_dict = {'image_name': mask_name,
                        'contour_index': int(pixel_int),
                        'cx_coords': 0,
                        'cy_coords': 0,
                        'area': -1,
                        'area_box': -1,
                        'radius_ratio': -1,
                        'aspect': -1,
                        'eccentricity': -1,
                        'roundness': -1,
                        'ii': -1,
                        'contour': [contour],
                        }
    # assembling contour df, i.e, making a row
    contour_df = DataFrame(contour_dict, index=[0])  # noqa

    # returns single row of dataframe
    return contour_df


def process_contour_phase(single_contour_img: np.ndarray,
                          mask_name: str,
                          pixint: float,
                          image: np.ndarray,
                          phase_red: np.ndarray,
                          phase_green: np.ndarray,
                          phase_blue: np.ndarray
                          ) -> pd.DataFrame:
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
    area = cv.contourArea(contour)
    print(area)

    channels_dict = {'grayscale': image,
                     'red': phase_red,
                     'green': phase_green,
                     'blue': phase_blue
                     }
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

        # update morpho dict with intensity and texture dicts
        morpho_dict.update(list_intensity_dict)
        morpho_dict.update(list_lbp_dict)
        morpho_dict.update(list_glcm_dict)

    # loop in single contour to extract parameters
    single_contour_df = get_df_features(single_contour_image=single_contour_img,
                                        contour=contour[0],
                                        pixel_int=pixint,
                                        mask_name=mask_name,
                                        pixint_list=og_intensity,
                                        phase_red_list=phase_red_intensity,
                                        phase_green_list=phase_green_intensity,
                                        phase_blue_list=phase_blue_intensity
                                        )
    rows_to_delete = single_contour_df[single_contour_df['area']==-1].index
    single_contour_df.drop(rows_to_delete, inplace=True)



    concat_metrics_df = pd.concat([single_contour_df, single_lbp_df, single_glcm_df],
                               axis=1)

    if not len(single_contour_df) == 0:
        # put label
        make_contour_label(contour_index=int(pixint),
                           centroid_x=single_contour_df['cx_coords'].iloc[0],
                           centroid_y=single_contour_df['cy_coords'].iloc[0],
                           color=255,
                           thickness=2,
                           img_to_label=image,
                           contour=contour[0],
                           )
    else:
        pass

        # put label
        # make_contour_label(contour_index=int(pixint),
        #                    centroid_x=single_contour_df['cx_coords'].iloc[0],
        #                    centroid_y=single_contour_df['cy_coords'].iloc[0],
        #                    color=255,
        #                    thickness=2,
        #                    img_to_label=image,
        #                    contour=contour[0],
        #                    )

    # returns the contours and the list of intensities
    return concat_metrics_df


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
        contour_df = process_contour_phase(single_contour_img=single_contour_img,
                                           mask_name=mask_name,
                                           pixint=pixel_intensity,
                                           image=image,
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

    cv.imwrite(overlays_output_path, image)

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
