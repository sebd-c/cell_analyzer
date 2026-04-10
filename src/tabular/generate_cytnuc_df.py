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
import cv2 as cv
# from cv2 import pointPolygonTest
import pandas as pd
# from pandas import concat
# from pandas import DataFrame
# from pandas import read_pickle
from os.path import join
from src._execution_formatting import print_execution_parameters
from src.utils.merge_channels import merge_multiple_images

print('all required libraries successfully imported.')  # noqa


#####################################################################
# module specific aux functions

def string_to_contours_array(input_string: str) -> np.ndarray:
    """
    Converts a formatted string representing cv2.findContours output into a numpy array.

    Args:
        input_string (str): The input string containing points in the format:
                           "[[[x1 y1]] [[x2 y2]] ... [[xn yn]]]".

    Returns:
        np.ndarray: A numpy array of shape (n, 2) representing contour points.
    """
    try:
        # Clean up the input string to remove unwanted characters
        clean_string = input_string.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()

        # Replace multiple spaces with a single space
        clean_string = ' '.join(clean_string.split())

        # Remove the outermost brackets and split into individual points
        clean_string = clean_string.strip('[]')
        points = clean_string.split('] [')

        # Parse each point into a list of integers
        contour_array = []
        for point in points:
            # Strip any leftover brackets and split into coordinates
            coords = point.replace('[', '').replace(']', '').split()
            contour_array.append([int(coord) for coord in coords])

        return np.array(contour_array)
    except Exception as e:
        raise ValueError(f"Failed to parse input string into a numpy array: {e}")


def link_cytnuc(cyt_df: pd.DataFrame,
                nuc_df: pd.DataFrame,
                output_path: str) -> None:
    """
    given a df from cytoplasm segmentation,
    and its respective nuclei df segmentation,
    associates each nucleus with its parent cytoplasm
    into a new df
    """

    # make a list as placeholder while making new linked df
    linked_dfs_list = []

    # change image names so they match
    cyt_df['image_name'] = cyt_df['image_name'].replace('green', '', regex=True)
    nuc_df['image_name'] = nuc_df['image_name'].replace('red', '', regex=True)

    # loop of nucleus through the cytoplasm df
    for nucleus_index, nuc_row in nuc_df.iterrows():

        # conditional to loop only in the cytoplasms
        # that have matching img name as the nucleus
        cyt_df_img = cyt_df[cyt_df['image_name'] == nuc_row['image_name']]

        for cyto_index, cyto_row in cyt_df_img.iterrows():
            # get current cyto
            cyto_contour = cyto_row['contour']

            # test if a point is inside an object
            if cv.pointPolygonTest(cyto_contour,
                                # if it is, it'll be a match parent cytoplasm
                                # measureDist 0 or 1
                                (nuc_row['cx_coords'], nuc_row['cy_coords']),
                                measureDist=False) > -1:
                # if (nuc_row['cx_coords'], nuc_row['cy_coords']) in cyto_row['pixel_coords_list']:
                # if the nucleus is nested in the contour,

                # having the tested pointed out it's a parent,
                # begin making a row for the new joint df
                #TODO: add flag d cyto/nuc no inicio pra
                # concatenar verticalmente sem essa baixaria
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
                               'cyto_contour': [cyto_row['contour']],
                               'cyto_grayscale_mean': cyto_row['grayscale_mean'],
                               'cyto_grayscale_median': cyto_row['grayscale_median'],
                               'cyto_grayscale_max': cyto_row['grayscale_max'],
                               'cyto_grayscale_min':cyto_row['grayscale_min'],
                               'cyto_grayscale_sum': cyto_row['grayscale_sum'],
                               'cyto_grayscale_int_density': cyto_row['grayscale_int_density'],
                               'cyto_red_mean': cyto_row['red_mean'],
                               'cyto_red_median': cyto_row['red_median'],
                               'cyto_red_max': cyto_row['red_max'],
                               'cyto_red_min': cyto_row['red_min'],
                               'cyto_red_sum': cyto_row['red_sum'],
                               'cyto_red_int_density': cyto_row['red_int_density'],
                               'cyto_green_mean': cyto_row['green_mean'],
                               'cyto_green_median': cyto_row['green_median'],
                               'cyto_green_max': cyto_row['green_max'],
                               'cyto_green_min': cyto_row['green_min'],
                               'cyto_green_sum': cyto_row['green_sum'],
                               'cyto_green_int_density': cyto_row['green_int_density'],
                               'cyto_blue_mean': cyto_row['blue_mean'],
                               'cyto_blue_median': cyto_row['blue_median'],
                               'cyto_blue_max': cyto_row['blue_max'],
                               'cyto_blue_min': cyto_row['blue_min'],
                               'cyto_blue_sum': cyto_row['blue_sum'],
                               'cyto_blue_int_density': cyto_row['blue_int_density'],
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
                               'nuc_contour': [nuc_row['contour']],
                               'nuc_grayscale_mean': nuc_row['grayscale_mean'],
                               'nuc_grayscale_median': nuc_row['grayscale_median'],
                               'nuc_grayscale_max': nuc_row['grayscale_max'],
                               'nuc_grayscale_min': nuc_row['grayscale_min'],
                               'nuc_grayscale_sum': nuc_row['grayscale_sum'],
                               'nuc_grayscale_int_density': nuc_row['grayscale_int_density'],
                               'nuc_red_mean': nuc_row['red_mean'],
                               'nuc_red_median': nuc_row['red_median'],
                               'nuc_red_max': nuc_row['red_max'],
                               'nuc_red_min': nuc_row['red_min'],
                               'nuc_red_sum': nuc_row['red_sum'],
                               'nuc_red_int_density': nuc_row['red_int_density'],
                               'nuc_green_mean': nuc_row['green_mean'],
                               'nuc_green_median': nuc_row['green_median'],
                               'nuc_green_max': nuc_row['green_max'],
                               'nuc_green_min': nuc_row['green_min'],
                               'nuc_green_sum': nuc_row['green_sum'],
                               'nuc_green_int_density': nuc_row['green_int_density'],
                               'nuc_blue_mean': nuc_row['blue_mean'],
                               'nuc_blue_median': nuc_row['blue_median'],
                               'nuc_blue_max': nuc_row['blue_max'],
                               'nuc_blue_min': nuc_row['blue_min'],
                               'nuc_blue_sum': nuc_row['blue_sum'],
                               'nuc_blue_int_density': nuc_row['blue_int_density']
                               # 'xgal_e': cyto_row['xgal_e'],
                               # 'xgal_d': cyto_row['xgal_d'],
                               # 'xgal_h': cyto_row['xgal_h'],
                               # 's_status_e': cyto_row['s_status_e'],
                               # 's_status_d': cyto_row['s_status_d'],
                               # 's_status_h': cyto_row['s_status_h'],
                               # 'cons_xgal': cyto_row['cons_xgal'],
                               # 'cons_sstatus': cyto_row['cons_sstatus'],
                               # 'label': cyto_row['label']
                               }

                # make the new dictionary into a temporary one row df
                linked_df = pd.DataFrame(linked_dict, index=[0])

                # append the newly made df into a list
                linked_dfs_list.append(linked_df)

                # since you found the parent cytoplasm,
                # you need to break the loop to move into the other nuclei
                break

    # concatenating contour df into bigger df
    # a pandas dataframe
    concat_linked_df = pd.concat(linked_dfs_list, ignore_index=True)

    # saving new df
    concat_linked_df.to_pickle(output_path)

    return


def make_cytnuc_output(cyt_csv_input_path: str,
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
    cyto_df = pd.read_pickle(cyt_csv_input_path)

    nuc_df = pd.read_pickle(nuc_csv_input_path)

    # create the path to save the output path
    output_path = join(csv_output_folder,
                       'contours_cytnuc.pickle')

    # linking the dfs
    link_cytnuc(cyt_df=cyto_df,
                nuc_df=nuc_df,
                output_path=output_path)

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
    parser.add_argument('-cf', '--cyto_images_folder',
                        dest='cyto_images_folder',
                        required=True,
                        help='defines path to folder containing overlayed cytoplasm images')

    # input folder param
    parser.add_argument('-nf', '--nuc_images_folder',
                        dest='nuc_images_folder',
                        required=True,
                        help='defines path to folder containing overlayed nuclei images')

    # images extension param
    parser.add_argument('-x', '--images_extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # csv cytoplasm input folder param
    parser.add_argument('-cic', '--cyto_input_csv',
                        dest='cyto_input_csv',
                        required=True,
                        help='defines path to input cytoplasm csv')

    # overlays output folder param
    parser.add_argument('-nic', '--nuc_input_csv',
                        dest='nuc_input_csv',
                        required=True,
                        help='defines path to output folder (overlays)')

    # overlays output folder param
    parser.add_argument('-oo', '--overlays_output_folder',
                        dest='overlays_output_folder',
                        required=True,
                        help='defines path to output folder (overlays)')

    # overlays output folder param
    parser.add_argument('-co', '--csv-output-folder',
                        dest='csv_output_folder',
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

    # getting cytoplasm overlayed images
    cyto_images_folder = args_dict['cyto_images_folder']

    # getting nuclei overlayed images
    nuc_images_folder = args_dict['nuc_images_folder']

    # getting cytoplasm csv
    cyto_input_csv = args_dict['cyto_input_csv']

    # getting nuclei csv
    nuc_input_csv = args_dict['nuc_input_csv']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting csv output path
    csv_output_folder = args_dict['csv_output_folder']

    # getting overlays output path
    overlays_output_folder = args_dict['overlays_output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # runnning join
    make_cytnuc_output(cyt_csv_input_path=cyto_input_csv,
                       nuc_csv_input_path=nuc_input_csv,
                       nuc_overlayed_input_folder=nuc_images_folder,
                       cyto_overlayed_input_folder=cyto_images_folder,
                       img_extension=images_extension,
                       csv_output_folder=csv_output_folder,
                       joined_overlays_output_folder=overlays_output_folder,
                       )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
