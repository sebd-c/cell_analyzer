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
def make_image_crops(image_path: str,
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
    image = imread(image_path,
                   -1)

    # define a pixel intensity
    pixel_intensity = 255

    # getting img shape
    shape = image.shape

    # segment the df to loop only on this image's objects
    image_df = df[df['image_name'] == image_name]

    # loop not to join different contours
    for index, row in image_df.iterrows():
        # noise cleaning process outside object
        # creates mask to hold contour
        mask = np.zeros(shape, dtype=np.uint8)

        # filling mask image
        drawContours(mask, row['contour'], 0, pixel_intensity, -1)

        # apply mask to make clean image
        clean_image = image[mask != pixel_intensity] = 0

        # flipping conditional
        _, _, w, h = boundingRect(row['contour'])

        # if the object is in another rotation, flip it
        if w > h:
            cropping_image = rotate(clean_image, ROTATE_90_CLOCKWISE)
        else:
            cropping_image = clean_image

        # vertex settings per object
        x1 = row['cx_coords'].iloc[0] - max_width/2
        x2 = row['cx_coords'].iloc[0] + max_width/2
        y1 = row['cy_coords'].iloc[0] - max_height/2
        y2 = row['cy_coords'].iloc[0] + max_height/2


        # image cropping
        crop = cropping_image[y1:y2, x1:x2]

        # making crop name for saving
        crop_name = row['contour_index'] + '_' + image_name

        # defining crop output path
        crop_output_path = join(output_folder, crop_name)

        # saving crop
        imwrite(crop_output_path, crop)

    return


def make_folder_crops(images_input_folder: str,
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
    image_files = get_files_in_folder(path_to_folder=images_input_folder,
                                      extension=image_extension)

    image_files_num = len(image_files)

    # iterating over image files
    for file_index, image_file in enumerate(image_files, 1):
        # printing progress message
        base_string = 'generating crop #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=image_files_num)

        # getting current image mask input/output paths
        image_input_path = join(images_input_folder,
                                image_file)

        # make and save image crops
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
    description = 'generate crops module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # grayscale input folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing original grayscale images')

    # segmentation dataframe input folder param
    parser.add_argument('-d', '--df-path',
                        dest='df_path',
                        required=True,
                        help='defines path to folder containing phase images')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # crops output folder param
    parser.add_argument('-o', '--output_folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder (crops)')

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

    # getting images folder
    images_folder = args_dict['images_folder']

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
    enter_to_continue()

    # running function to get the segmentation df for a folder
    make_folder_crops(images_input_folder=masks_folder,
                      og_img_folder=images_folder,
                      phase_img_folder=phase_folder,
                      image_extension=images_extension,
                      csv_output_folder=csv_output_folder,
                      output_folder=overlays_output_folder,
                      )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
