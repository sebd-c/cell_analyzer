# generate segmentation dfs module
import numpy as np

print('initializing...')  # noqa

# Code destined to generating
# segmentation mask based df.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from argparse import ArgumentParser
from cv2 import imread
from cv2 import contourArea
from cv2 import findContours
from cv2 import split
from cv2 import imwrite
from cv2 import COLOR_BGR2RGB
from cv2 import cvtColor
from cv2 import CHAIN_APPROX_NONE
from cv2 import RETR_EXTERNAL
from numpy import uint8 as np_uint8
from numpy import ndarray
from pandas import concat
from pandas import DataFrame
from os.path import join
from numpy import max
from numpy import min
from numpy import mean
from numpy import median
from numpy import sum
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_contour_centroid
from src.utils.aux_funcs import get_area_box
from src.utils.aux_funcs import get_contour_rratio
from src.utils.aux_funcs import get_contour_ellipse_feats
from src.utils.aux_funcs import get_contour_roundness
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import make_contour_label

print('all required libraries successfully imported.')  # noqa


#####################################################################
def get_parameters_df(contour: ndarray,
                      pixel_int: float,
                      mask_name: str,
                      flag: int,
                      pixint_list: list,
                      phase_red_list: list or None = None,
                      phase_green_list: list or None = None,
                      phase_blue_list: list or None = None
                      ) -> DataFrame:
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
    area = contourArea(contour)

    # TODO: check why this is happening and if unavoidable, put in argparser
    if area > 10:

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
        contour_dict = {'image_name': mask_name,
                        'contour_index': int(pixel_int),
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

        # lastly, we define what pixel intensities are going to be added
        # if the function was called by the processing of a phase img
        if flag == 1:
            # getting pixel intensity values for RGB and adding them to the dict
            contour_dict['grayscale_mean'] = mean(pixint_list)
            contour_dict['grayscale_median'] = median(pixint_list)
            contour_dict['grayscale_max'] = max(pixint_list)
            contour_dict['grayscale_min'] = min(pixint_list)
            contour_dict['grayscale_sum'] = sum(pixint_list)
            contour_dict['grayscale_int_density'] = contour_dict['grayscale_sum'] / area

            contour_dict['red_mean'] = mean(phase_red_list)
            contour_dict['red_median'] = median(phase_red_list)
            contour_dict['red_max'] = max(phase_red_list)
            contour_dict['red_min'] = min(phase_red_list)
            contour_dict['red_sum'] = sum(phase_red_list)
            contour_dict['red_int_density'] = contour_dict['red_sum'] / area

            contour_dict['green_mean'] = mean(phase_green_list)
            contour_dict['green_median'] = median(phase_green_list)
            contour_dict['green_max'] = max(phase_green_list)
            contour_dict['green_min'] = min(phase_green_list)
            contour_dict['green_sum'] = sum(phase_green_list)
            contour_dict['green_int_density'] = contour_dict['green_sum'] / area

            contour_dict['blue_mean'] = mean(phase_blue_list)
            contour_dict['blue_median'] = median(phase_blue_list)
            contour_dict['blue_max'] = max(phase_blue_list)
            contour_dict['blue_min'] = min(phase_blue_list)
            contour_dict['blue_sum'] = sum(phase_blue_list)
            contour_dict['blue_int_density'] = contour_dict['blue_sum'] / area

        # if the function was called by the processing of a simple channel img
        #  simply calculate for that grayscale
        elif flag == 0:
            contour_dict['grayscale_mean'] = mean(pixint_list)
            contour_dict['grayscale_median'] = median(pixint_list)
            contour_dict['grayscale_max'] = max(pixint_list)
            contour_dict['grayscale_min'] = min(pixint_list)
            contour_dict['grayscale_sum'] = sum(pixint_list)
            contour_dict['grayscale_int_density'] = contour_dict['grayscale_sum'] / area

    else:
        contour_dict = {'image_name': mask_name,
                        'contour_index': -1,
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


def process_contour_phase(single_contour_img: ndarray,
                          mask_name: str,
                          pixint: float,
                          image: ndarray,
                          phase_red: ndarray,
                          phase_green: ndarray,
                          phase_blue: ndarray
                          ) -> DataFrame:
    """
    :param og_phase:
    :param pixint:
    :param mask_name:
    :param single_contour_img:
    :param image:
    :param phase_red:
    :param phase_green:
    :param phase_blue:
    :return:
    """

    # getting pixel intensity for each channel
    phase_red_intensity = phase_red[single_contour_img == 1]
    phase_green_intensity = phase_green[single_contour_img == 1]
    phase_blue_intensity = phase_blue[single_contour_img == 1]
    og_intensity = image[single_contour_img == 1]

    # converting int type
    single_contour_img = single_contour_img.astype(np_uint8)

    # finding contour in image
    contour, _ = findContours(single_contour_img, RETR_EXTERNAL, CHAIN_APPROX_NONE)

    # print(contour)
    # print(contour[0])
    # exit()
    single_contour_df = get_parameters_df(contour=contour[0],
                                          pixel_int=pixint,
                                          mask_name=mask_name,
                                          flag=1,
                                          pixint_list=og_intensity,
                                          phase_red_list=phase_red_intensity,
                                          phase_green_list=phase_green_intensity,
                                          phase_blue_list=phase_blue_intensity
                                          )
    rows_to_delete = single_contour_df[single_contour_df['area']==-1].index
    single_contour_df.drop(rows_to_delete, inplace=True)

    print(single_contour_df.shape)

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

    # returns the contours and the list of intensities
    return single_contour_df


# module specific aux functions
def make_image_contours_df(mask_name: str,
                           mask_path: str,
                           og_img_path: str,
                           p_img_path: str,
                           overlays_output_folder: str
                           ) -> DataFrame:
    """
    Given a path to a binary image,
    finds contours and
    returns data frame containing
    contours desired infos.
    """

    # read the rgb channel
    phase_image_cv2 = imread(p_img_path,
                             -1)

    # convert it to the accurate colors
    phase_image = cvtColor(phase_image_cv2, COLOR_BGR2RGB)

    # separate channels
    phase_red, phase_green, phase_blue = split(phase_image)

    # reading mask and image
    mask = imread(mask_path,
                  -1)

    image = imread(og_img_path,
                   -1)

    # separating contours before binarizing mask
    max_intensity = mask.max()  # noqa

    # getting shape for new masks arrays
    shape = mask.shape

    # create an empty list to hold the single contours df
    contours_df_list = []

    # loop not to join different contours
    for pixel_intensity in range(1, max_intensity + 1):
        # create a new blank img
        single_contour_img = np.zeros(shape)

        # putting the contour inside
        single_contour_img[mask == pixel_intensity] = 1

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
    concat_contours_df = concat(contours_df_list, ignore_index=True)

    # save image with labels
    overlays_output_path = join(overlays_output_folder, mask_name)

    imwrite(overlays_output_path, image)

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
        # get image contour df and respective overlayed imgs
        image_df = make_image_contours_df(mask_name=mask_file,
                                          mask_path=mask_input_path,
                                          og_img_path=og_img_input_path,
                                          p_img_path=phase_img_input_path,
                                          overlays_output_folder=overlays_output_folder
                                          )

        # append curr img df to dir df
        dfs_list.append(image_df)

    # concating "dfs" from dfs lists into
    # a pandas dataframe
    contour_df = concat(dfs_list, ignore_index=True)

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
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # grayscale input folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing original images')
    # phase input folder param
    parser.add_argument('-p', '--phase-images-folder',
                        dest='phase_folder',
                        required=True,
                        help='defines path to folder containing phase images')

    # input masks folder param
    parser.add_argument('-m', '--masks-folder',
                        dest='masks_folder',
                        required=True,
                        help='defines path to folder containing cellpose masks outputs')

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
    #                     help='defines wich type of segmentation it is. "nuc" or "cyto"')

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
