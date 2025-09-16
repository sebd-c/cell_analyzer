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
from cv2 import boundingRect
from cv2 import drawContours
from cv2 import split
from cv2 import imwrite
from cv2 import rotate
from cv2 import ROTATE_90_CLOCKWISE
from cv2 import ROTATE_90_COUNTERCLOCKWISE
from numpy import uint8 as np_uint8
from numpy import ndarray
from numpy import pad
from pandas import concat, read_pickle
from pandas import DataFrame
from os.path import join
from numpy import max
from numpy import min
from numpy import mean
from numpy import median
from numpy import sum
from numpy import unique
from numpy import isin
import tifffile
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
from src.utils.aux_funcs import get_unique_ids
from src.utils.aux_funcs import get_coords
from src.utils.aux_funcs import get_boundaries_mask


print('all required libraries successfully imported.')  # noqa


#####################################################################

# module specific aux functions
def make_image_crops(cyto_path: str,
                     nuc_path: str,
                     phase_path: str,
                     image_name: str,
                     df: DataFrame,
                     max_width: int or float,
                     max_height: int or float,
                     output_folder: str
                     ) -> None:
    """
    Given a path to a grayscale image,
    a dataframe containing each object's centroid,
    id, origin image and contour,
    generates crops of all of that image's objects,
    and saves the results in the output folder.
    """

    # reading grayscale image
    cyto_image = imread(cyto_path,
                        -1)

    # reading grayscale image
    nuc_image = imread(nuc_path,
                        -1)

    # reading grayscale image
    phase_image = imread(phase_path,
                        -1)

    # define a pixel intensity
    pixel_intensity = 255

    # getting img shape
    shape = cyto_image.shape

    # segment the df to loop only on this image's objects
    image_df = df[df['image_name'] == image_name]

    # unique df
    unique_df = image_df.drop_duplicates(subset=['image_name', 'cyto_id'], keep='first')

    duplicated_cyto_df = image_df[image_df.duplicated(subset=['image_name', 'cyto_id'], keep=False)]

    # loop not to join different contours
    for index, row in unique_df.iterrows():
        # noise cleaning process outside object
        # creates mask to hold cytoplasm contour
        cyto_mask = np.zeros(shape, dtype=np.uint8)

        # filling mask image
        drawContours(cyto_mask, row['cyto_contour'], 0, pixel_intensity, -1)

        # creates mask to hold nucleus contour
        nuc_mask = np.zeros(shape, dtype=np.uint8)

        # checks if there is more than one nucleus
        if (row[['image_name', 'cyto_id']] == duplicated_cyto_df[['image_name', 'cyto_id']]).all(axis=1).any():
            # if so, get all nucleus to be considered
            select_nucleus_df = duplicated_cyto_df[(duplicated_cyto_df['image_name'] == row['image_name']) & (duplicated_cyto_df['cyto_id'] == row['cyto_id'])]

            # loop making contours
            for nuc_index, nuc_row in select_nucleus_df.iterrows():
                # filling mask image
                drawContours(nuc_mask, row['nuc_contour'], 0, pixel_intensity, -1)

        # apply mask to make clean image
        clean_image = cyto_image[cyto_mask != pixel_intensity] = 0

        # flipping conditional
        _, _, w, h = boundingRect(row['contour'])

        # if the object is in another rotation, flip it
        if w > h:

            # vertex settings per object
            x1 = row['cx_coords'].iloc[0] - max_height / 2
            x2 = row['cx_coords'].iloc[0] + max_height / 2
            y1 = row['cy_coords'].iloc[0] - max_width / 2
            y2 = row['cy_coords'].iloc[0] + max_width / 2

            # image cropping
            crop = clean_image[y1:y2, x1:x2]

            # fix crop orientation
            oriented_crop = rotate(crop, ROTATE_90_CLOCKWISE)

        else:

            # vertex settings per object
            x1 = row['cx_coords'].iloc[0] - max_width / 2
            x2 = row['cx_coords'].iloc[0] + max_width / 2
            y1 = row['cy_coords'].iloc[0] - max_height / 2
            y2 = row['cy_coords'].iloc[0] + max_height / 2

            # image cropping
            crop = clean_image[y1:y2, x1:x2]

            # fix crop orientation
            oriented_crop = crop

        # add padding
        pad(oriented_crop, ((5, 5), (5, 5)), mode='constant', constant_values=0)

        # making crop name for saving
        crop_name = row['cyto_id'] + '_' + image_name

        # defining crop output path
        crop_output_path = join(output_folder, crop_name)

        # saving crop
        imwrite(crop_output_path, oriented_crop)

    return


def make_folder_crops(cyto_input_folder: str,
                      nuc_input_folder: str,
                      phase_input_folder: str,
                      image_extension: str,
                      df_path: str,
                      output_folder: str,
                      ) -> None:
    """
    Given a path to a folder containing
    grayscale images of an object and
    a dataframe containing each object's centroid,
    id, origin image and contour,
    generates crops of all objects,
    and saves the results in the output folder.
    """

    # read csv
    df = read_pickle(df_path)

    # variables to save width/height values
    max_height = float('-inf')
    max_width = float('-inf')

    # defining crop shape
    for index, row in df.iterrows():

        # use open cv bb to get min rect shape
        x, y, w, h = boundingRect(row['contour'])

        # as a big object in height or width are the same
        # set one variable to always be the bigger one (some objects will be "flipped" in the end)
        width = w if w < h else h
        height = w if w > h else h

        # compare with maxes saved
        max_width = max_width if max_width > width else width
        max_height = max_height if max_height > height else height

    # getting image files in respective input folder
    cyto_files = get_files_in_folder(path_to_folder=cyto_input_folder,
                                      extension=image_extension)

    # getting image files in respective input folder
    nuc_files = get_files_in_folder(path_to_folder=nuc_input_folder,
                                     extension=image_extension)

    # getting image files in respective input folder
    phase_files = get_files_in_folder(path_to_folder=phase_input_folder,
                                    extension=image_extension)

    image_files_num = len(cyto_files)

    # iterating over image files
    for file_index, cyto_file in enumerate(cyto_files, 1):
        # printing progress message
        base_string = 'generating crop image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=image_files_num)

        # getting current image mask input/output paths
        cyto_input_path = join(cyto_input_folder,
                                cyto_file)

        # getting respective og img
        nuc_file = nuc_files[file_index - 1]

        # getting respective phase img
        phase_file = phase_files[file_index - 1]

        # analogous getting current og image input/output paths
        nuc_input_path = join(nuc_input_folder,
                              nuc_file)

        # analogous getting current og image input/output paths
        phase_input_path = join(phase_input_folder,
                                    phase_file)

        # make and save image crops
        make_image_crops(cyto_path=cyto_input_path,
                         nuc_path=nuc_input_path,
                         phase_path=phase_input_path,
                         image_name=cyto_file,
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
    description = 'generate crops module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # grayscale input folder param
    parser.add_argument('-c', '--cyto-folder',
                        dest='cyto_folder',
                        required=True,
                        help='defines path to folder containing original grayscale cytoplasm images')

    # grayscale input folder param
    parser.add_argument('-n', '--nuclei-folder',
                        dest='nuclei_folder',
                        required=True,
                        help='defines path to folder containing original nuclei grayscale images')

    # grayscale input folder param
    parser.add_argument('-p', '--phase-folder',
                        dest='phase_folder',
                        required=True,
                        help='defines path to folder containing phase grayscale-ed images')

    # segmentation dataframe input folder param
    parser.add_argument('-d', '--df-path',
                        dest='df_path',
                        required=True,
                        help='defines path to folder containing full df')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # crops output folder param
    parser.add_argument('-co', '--cyto-output-folder',
                        dest='cyto_output_folder',
                        required=True,
                        help='defines path to output folder (cyto crops)')

    # crops output folder param
    parser.add_argument('-no', '--nuclei-output-folder',
                        dest='nuclei_output_folder',
                        required=True,
                        help='defines path to output folder (nuc crops)')

    # crops output folder param
    parser.add_argument('-po', '--phase-output-folder',
                        dest='phase_output_folder',
                        required=True,
                        help='defines path to output folder (phase crops)')

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

    # getting cytoplasm folder
    cyto_folder = args_dict['cyto_folder']

    # getting nuclei folder
    nuclei_folder = args_dict['nuclei_folder']

    # getting cytoplasm folder
    phase_folder = args_dict['phase_folder']

    # path to segmentation dataframe
    df_path = args_dict['df_path']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting output folder
    output_folder = args_dict['output_folder']

    # # getting type of segmentation
    # segmentation_type = args_dict['segmentation_type']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input

    # running function to get the segmentation df for a folder
    make_folder_crops(cyto_input_folder=cyto_folder,
                      image_extension=images_extension,
                      df_path=df_path,
                      output_folder=output_folder,
                      )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
