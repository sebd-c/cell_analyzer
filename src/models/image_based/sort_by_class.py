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
from cv2 import imwrite
from pandas import DataFrame
from pandas import read_pickle
from os.path import join
from src.utils.aux_funcs import make_dir_list
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_execution_parameters

print('all required libraries successfully imported.')  # noqa


#####################################################################
# module specific aux functions

def organize_crop(crop_path: str,
                  crop_name: str,
                  df: DataFrame,
                  output_path_dict: dict
                  ) -> None:
    """
    given an object crop,
    and a df containing its id, origin img and label,
    a dict of folder name and path,
    creates a copy of the crop img into designated folder
    """

    # reading crop image
    crop = imread(crop_path,
                   -1)

    # label dict for translation from df to directory name
    label_dict = {0: 'normal', 1: 'quiescent', 2: 'fully_senescent', 3: 'senescent_like'}

    # refactor df
    df['label'] = df['label'].replace(label_dict)

    # create crop name split for identification in df
    crop_name_split = crop_name.split('_')

    # identify crop in df to get label
    crop_row = df[(df['contour_index'] == crop_name_split[0]) & (df['image_name'] == crop_name_split[1])]

    # getting crop label
    crop_label = crop_row['label']

    # getting output path
    crop_output_path = output_path_dict[crop_label]

    # save in specified dir
    imwrite(crop_output_path, crop)

    return


def organize_crops_folder(images_input_folder: str,
                          image_extension: str,
                          df_path: str,
                          output_path_dict: dict,
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
        organize_crop(crop_path=image_input_path,
                      crop_name=crop_file,
                      df=df,
                      output_path_dict=output_path_dict)

    # printing execution message
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
    output_path_dict = make_dir_list(class_list=class_list,
                                     output_folder=output_folder)
    # runnning
    organize_crops_folder(images_input_folder=images_folder,
                          image_extension=images_extension,
                          df_path=df_path,
                          output_path_dict=output_path_dict
                          )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
