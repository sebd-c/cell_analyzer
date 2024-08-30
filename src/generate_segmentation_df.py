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
from cv2 import drawContours
from cv2 import imwrite
from cv2 import CHAIN_APPROX_NONE
from cv2 import RETR_EXTERNAL
from numpy import uint8 as np_uint8
from pandas import concat
from pandas import DataFrame
from os.path import join
from numpy import max
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_contour_centroid
from src.utils.aux_funcs import get_area_box
from src.utils.aux_funcs import get_contour_rratio
from src.utils.aux_funcs import get_contour_ellipse_feats
from src.utils.aux_funcs import get_contour_roundness
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_execution_parameters

print('all required libraries successfully imported.')  # noqa


#####################################################################
# module specific aux functions

def make_image_contours_df(mask_name: str,
                           mask_path: str,
                           og_img_path: str,
                           overlays_output_folder: str
                           ) -> DataFrame:
    """
    Given a path to a binary image,
    finds contours and returns data
    frame containing contours coords.
    """
    # reading mask and image
    mask = imread(mask_path,
                  -1)

    image = imread(og_img_path,
                   -1)

    # separating contours before binarizing mask
    max_intensity = mask.max()

    # getting shape for new masks arrays
    shape = mask.shape

    # empty list to hold contours
    contours = []
     # todo nuclei segmentation
    # loop not to "join different contours"
    for pixel_intensity in range(1, max_intensity + 1):
        # create a new contour
        single_contour = np.zeros(shape)

        # putting the contour inside
        single_contour[mask == pixel_intensity] = 1

        # converting int type
        single_contour = single_contour.astype(np_uint8)

        # finding contours in image
        contour, _ = findContours(single_contour, RETR_EXTERNAL, CHAIN_APPROX_NONE)

        # putting contour in list
        contours.append(contour[0])

    # gets enumerate to have indexes
    contours_enumerate = enumerate(contours, 1)

    # create dfs list holder
    contours_df_list = []

    # loop inside an image
    # to work with that images' contours
    for contour_index, contour in contours_enumerate:

        # getting current contour area
        area = contourArea(contour)

        # TODO: check why this is happening and if unavoidable, put in argparser
        if area < 10:
            continue

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
        cii = (0.9 * aspect) - (0.87 * area_box) + (0.96 * radius_ratio) + (0.92 * roundness)

        # by the end of that loop, you now have a list of one contour features
        # now moving to organizing them into a dictionary
        contour_dict = {'image_name': mask_name,
                        'contour_index': contour_index,
                        'cx_coords': centroid_x,
                        'cy_coords': centroid_y,
                        'area': area,
                        'area_box': area_box,
                        'radius_ratio': radius_ratio,
                        'aspect': aspect,
                        'eccentricity': eccentricity,
                        'roundness': roundness,
                        'cii': cii
                        }

        # assembling contour df, i.e, making a row
        contour_df = DataFrame(contour_dict, index=[0])

        # appending current df to dfs list
        contours_df_list.append(contour_df)

    # concating contour df into bigger df
    # a pandas dataframe
    concat_contours_df = concat(contours_df_list, ignore_index=True)

    # drawing contours in img
    overlayed_image = drawContours(image,
                                   contours,
                                   -1,
                                   (0, 255, 0),
                                   thickness=2)

    # saving layered img
    overlays_output_path = join(overlays_output_folder, mask_name)
    imwrite(overlays_output_path, overlayed_image)

    # returning contours df
    return concat_contours_df


def make_folder_contours_df(masks_input_folder: str,
                            og_imgs_input_folder: str,
                            masks_img_extension: str,
                            csv_output_folder: str,
                            overlays_output_folder: str
                            ) -> None:
    """
    Given a path to a folder containing
    cytoplasms masks, generates a df
    containing the wanted information,
    and saving the results
    in the output folder.
    """
    # getting masks files in respective input folder
    mask_files = get_files_in_folder(path_to_folder=masks_input_folder,
                                     extension=masks_img_extension)

    masks_files_num = len(mask_files)

    # create empty list to hold the dfs
    dfs_list = []

    # iterating over files
    for file_index, mask_file in enumerate(mask_files, 1):
        # printing progress message
        base_string = 'generating segmentation df #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=masks_files_num)

        # getting current image mask input/output paths
        mask_input_path = join(masks_input_folder,
                               mask_file)

        # create img file name
        og_img_file = mask_file.replace('_cp_masks', '')

        # analogous getting current og image input/output paths
        og_img_input_path = join(og_imgs_input_folder,
                                 og_img_file)

        # get image contour df and respective overlayed imgs
        image_df = make_image_contours_df(mask_name=mask_file,
                                          mask_path=mask_input_path,
                                          og_img_path=og_img_input_path,
                                          overlays_output_folder=overlays_output_folder)

        # append curr img df to dir df
        dfs_list.append(image_df)

    # concating "dfs" from dfs lists into
    # a pandas dataframe
    contour_df = concat(dfs_list, ignore_index=True)

    # create the path to save the output path
    output_path = join(csv_output_folder,
                       'contours_df.csv')

    # saving df
    contour_df.to_csv(output_path, index=False)

    # printing execution message
    print(f'output saved to {csv_output_folder}')
    print('analysis complete!')


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

    # input folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing original images')

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
                        help='defines path to output folder (csvs)')

    # overlays output folder param
    parser.add_argument('-oo', '--overlays_output_folder',
                        dest='overlays_output_folder',
                        required=True,
                        help='defines path to output folder (overlays)')

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

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting csv output path
    csv_output_folder = args_dict['csv_output_folder']

    # getting overlays output path
    overlays_output_folder = args_dict['overlays_output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to get the segmentation df for a folder
    make_folder_contours_df(masks_input_folder=masks_folder,
                            og_imgs_input_folder=images_folder,
                            masks_img_extension=images_extension,
                            csv_output_folder=csv_output_folder,
                            overlays_output_folder=overlays_output_folder)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
