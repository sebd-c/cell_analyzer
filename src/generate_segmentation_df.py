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
from cv2 import pointPolygonTest
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
    finds contours and
    returns data frame containing
    contours desired infos.
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
        ii = (0.9 * aspect) - (0.87 * area_box) + (0.96 * radius_ratio) + (0.92 * roundness)

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
                        'ii': ii,
                        'contour': contour
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


def link_cytnuc(cyt_df: DataFrame, nuc_df: DataFrame) -> DataFrame:
    """
    given a df from cytoplasm segmentation,
    and its respective nuclei df segmentation,
    associates each nucleus with its parent cytoplasm
    into a new df
    """

    # make a list as placeholder while making new linked df
    linked_dfs_list = []

    # loop of nucleus through the cytoplasm df
    for nucleus_index, nuc_row in nuc_df.iterrows():
        for cyto_index, cyto_row in cyt_df.iterrows():
            # checking if it's a match parent cytoplasm
            # and if it is, 0 or 1,
            if pointPolygonTest(cyto_row['contour'],
                                (nuc_row['cx_coords'], nuc_row['cy_coords']),
                                measureDist=False) > -1:
                # if the nucleus is nested in the contour,
                # start a new df, with both df information
                linked_dict = {'image_name': cyto_row['image_name'],
                               'cyto_id': cyto_row['contour_index'],
                               'cyto_cx': cyto_row['cx_coords'],
                               'cyto_cy': cyto_row['cy_coords'],
                               'cyto_area': cyto_row['area'],
                               'cyto_arbox': cyto_row['area_box'],
                               'cyto_radra': cyto_row['radius_ratio'],
                               'cyto_asp': cyto_row['aspect'],
                               'cyto_ecc': cyto_row['eccentricity'],
                               'cyto_rou': cyto_row['roundness'],
                               'cii': cyto_row['ii'],
                               'cyto_contour': cyto_row['contour'],
                               'nuc_id': nuc_row['contour_index'],
                               'nuc_cx': nuc_row['cx_coords'],
                               'nuc_cy': nuc_row['cy_coords'],
                               'nuc_area': nuc_row['area'],
                               'nuc_arbox': nuc_row['area_box'],
                               'nuc_radra': nuc_row['radius_ratio'],
                               'nuc_asp': nuc_row['aspect'],
                               'nuc_ecc': nuc_row['eccentricity'],
                               'nuc_rou': nuc_row['roundness'],
                               'nii': nuc_row['ii'],
                               'nuc_contour': nuc_row['contour'],
                               }
                # since you found the parent cytoplasm,
                # you need to break the loop to move into the other nuclei
                break

        # make the new dictionary into a temporary one row df
        linked_df = DataFrame(linked_dict, index=[0])

        # append the newly made df into a list
        linked_dfs_list.append(linked_df)
    # concating contour df into bigger df
    # a pandas dataframe
    concat_linked_df = concat(linked_dfs_list, ignore_index=True)

    return concat_linked_df


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
