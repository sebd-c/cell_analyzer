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
from cv2 import pointPolygonTest
from pandas import concat
from pandas import DataFrame
from pandas import read_pickle
from os.path import join
from os import mkdir
from src.utils.merge_channels import merge_multiple_images
from src.utils.aux_funcs import make_dir_list
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_execution_parameters

print('all required libraries successfully imported.')  # noqa


#####################################################################
# module specific aux functions

def organize_crop(crop_path: str,
                  df: DataFrame,
                  output_path_list: list
                  ) -> None:
    """
    given an object crop,
    and a df containing its id, origin img and label
    creates a copy of the crop img into designated folder
    """

    # make a list as placeholder while making new linked df
    linked_dfs_list = []

    label_dict = {0: 'normal', 1: 'quiescent', 2: 'fully_senescent', 3: 'senescent_like'}

    # class names
    class_names = ['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']

    # change image names so they match
    cyt_df['image_name'] = cyt_df['image_name'].replace('green', '', regex=True)
    nuc_df['image_name'] = nuc_df['image_name'].replace('red', '', regex=True)

    # loop of nucleus through the cytoplasm df
    for nucleus_index, nuc_row in nuc_df.iterrows():

        # conditional to loop only in the cytoplasms
        # that have matching img name as the nucleus
        cyt_df_img = cyt_df[cyt_df['image_name'] == nuc_row['image_name']]

        for cyto_index, cyto_row in cyt_df_img.iterrows():
            # remember to put the contour into the df
            # when saved in a csv, the contour turns into string
            cyto_contour = cyto_row['contour']

            # str_unextracted_cyto = string_to_contours_array(input_string=str_unextracted_cyto)

            # # we adapted it with a surplus of brackets
            # # before proceeding, you need to remove that surplus of brackets
            # # TODO: alterar tudo pra .pickle ou achar uma conversão melhor pra isso
            # front = str_unextracted_cyto.find('[')
            # str_unextracted_cyto = str_unextracted_cyto[:front] + str_unextracted_cyto[front + 1:]
            # rear = str_unextracted_cyto.rfind(']')
            # str_unextracted_cyto = str_unextracted_cyto[:rear] + str_unextracted_cyto[rear + 1:]
            #
            # # after having adapt the string to array, we can proceed
            # unextracted_cyto = array(str_unextracted_cyto)
            # extracted_cyto = UMat(unextracted_cyto)

            # now use this cv2 function to
            # test if a point is inside an object
            if pointPolygonTest(cyto_contour,
                                # if it is, it'll be a match parent cytoplasm
                                # measureDist 0 or 1
                                (nuc_row['cx_coords'], nuc_row['cy_coords']),
                                measureDist=False) > -1:
                # if (nuc_row['cx_coords'], nuc_row['cy_coords']) in cyto_row['pixel_coords_list']:
                # if the nucleus is nested in the contour,

                # having the tested pointed out it's a parent,
                # begin making a row for the new joint df
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
    concat_linked_df.to_pickle(output_path)

    return concat_linked_df


def organize_crops_folder(images_input_folder: str,
                          image_extension: str,
                          df_path: str,
                          output_folder: list,
                          ) -> None:
    """
    Given a path to a folder containing
    grayscale crop images of objects and
    a dataframe containing each crop's
    id, origin image and class label,
    generates a copy of the crop in the desired directory.
    """

    # read csv
    df = read_pickle(df_path)

    # getting image files in respective input folder
    crop_files = get_files_in_folder(path_to_folder=images_input_folder,
                                      extension=image_extension)

    crop_files_num = len(crop_files)

    # iterating over image files
    for file_index, crop_file in enumerate(crop_files, 1):
        # printing progress message
        base_string = 'generating crop #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=crop_files_num)

        # getting current image mask input path
        image_input_path = join(images_input_folder,
                                crop_file)

        # evaluate class by crop
        make_image_crops(image_path=image_input_path,
                         image_name=image_file,
                         df=df,
                         max_width=max_width,
                         max_height=max_height,
                         output_folder=output_folder)

    # printing execution message
    print(f'output saved to {output_folder}')
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
    parser.add_argument('-i', '--images_folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing grayscale crop images')

    # images extension param
    parser.add_argument('-x', '--images_extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # input df param
    parser.add_argument('-d', '--df-path',
                        dest='df_path',
                        required=True,
                        help='defines path to dataframe containing class labeled objects')

    # input folder param
    parser.add_argument('-cl', '--class-list',
                        nargs='+',
                        dest='class_list',
                        required=True,
                        help='list of strings that define the name of the labels')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folders (each crop in its class directory)')

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

    # getting grayscale crops folder
    images_folder = args_dict['images_folder']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting dataframe with ids + labels
    df_path = args_dict['df_path']

    # getting classes list
    class_list = args_dict['class_list']

    # output folder path
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()
    output_path_list = make_dir_list(class_list=class_list,
                                     output_folder=output_folder)
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
