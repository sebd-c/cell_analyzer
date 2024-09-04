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
from cv2 import putText
from cv2 import LINE_8
from cv2 import CHAIN_APPROX_NONE
from cv2 import RETR_EXTERNAL
from cv2 import FONT_HERSHEY_COMPLEX
from numpy import uint8 as np_uint8
from pandas import concat
from pandas import DataFrame
from pandas import read_csv
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
from src.utils.merge_channels import merge_multiple_images

print('all required libraries successfully imported.')  # noqa


#####################################################################
# module specific aux functions

def link_cytnuc(cyt_df: DataFrame,
                nuc_df: DataFrame,
                output_path: str) -> None:
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
                # make the new dictionary into a temporary one row df
                linked_df = DataFrame(linked_dict, index=[0])

                # append the newly made df into a list
                linked_dfs_list.append(linked_df)

                # since you found the parent cytoplasm,
                # you need to break the loop to move into the other nuclei
                break

    # concating contour df into bigger df
    # a pandas dataframe
    concat_linked_df = concat(linked_dfs_list, ignore_index=True)

    # saving new df
    concat_linked_df.to_csv(output_path, index=False)

    return


def make_folder_cytnuc(cyt_csv_input_path: str,
                       nuc_csv_input_path: str,
                       nuc_overlayed_input_folder: str,
                       cyto_overlayed_input_folder: str,
                       img_extension: str,
                       csv_output_folder: str,
                       joined_overlays_output_folder: str,
                       ) -> None:
    """
    Given a path to a cytoplasm df
    a nuclei df, the parent images w overlays
    of both, joins this two dfs
    relating each nucleus to a parent
    and saves the results in the output folder,
    as well as a merged version of the overlayed images.
    """
    # merging nuclei and cytoplasm imgs
    merge_multiple_images(nuclei_folder=nuc_overlayed_input_folder,
                          cyto_folder=cyto_overlayed_input_folder,
                          output_folder=joined_overlays_output_folder,
                          img_extension=img_extension)

    # read the .csv tables of both making them into a DataFrame object
    cyto_df = read_csv(cyt_csv_input_path)

    nuc_df = read_csv(nuc_csv_input_path)

    # create the path to save the output path
    output_path = join(csv_output_folder,
                       'contours_cytnuc.csv')

    # linking the dfs
    link_cytnuc(cyt_df=cyto_df,
                nuc_df=nuc_df,
                output_path=output_path)

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
