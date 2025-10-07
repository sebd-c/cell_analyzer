# generate binary masks module

print('initializing...')  # noqa

# Code destined to generating
# binary masks based on a pixel
# intensity threshold.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from numpy import stack
from os.path import join
from numpy import ndarray
from numpy import uint8 as np_uint8
from numpy import zeros as np_zeroes
from cv2 import imread
from cv2 import GaussianBlur
from cv2 import findContours
from cv2 import drawContours
from cv2 import RETR_EXTERNAL
from cv2 import BORDER_DEFAULT
from cv2 import CHAIN_APPROX_NONE
from cv2 import contourArea
from skimage.io import imsave as sk_imsave
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################

######################################################################
# defining auxiliary functions


def mask_single_img(image: ndarray,
                    lower_threshold: int,
                    upper_threshold: int,
                    min_size: int,
                    max_size: int,
                    kernel_size: int
                    ) -> ndarray:
    """
    Given a 2d array, returns masked
    image, based on given parameters.
    """
    # getting image dimensions
    image_dimensions = image.shape
    img_height, img_width = image_dimensions

    # blurring image (good for rounding objects)
    image = GaussianBlur(image,
                         (kernel_size, kernel_size),
                         BORDER_DEFAULT)

    # applying upper threshold (discards pixels with higher intensity than threshold)
    image[image > upper_threshold] = 0

    # applying lower threshold
    image[image < lower_threshold] = 0
    image[image >= lower_threshold] = 255

    # converting image to 8bit
    image = image.astype(np_uint8)

    # finding contours
    contours, _ = findContours(image,
                               RETR_EXTERNAL,
                               CHAIN_APPROX_NONE)

    # resetting image (I'll redraw only contours within specified size parameters)
    image = np_zeroes(image_dimensions)

    # iterating over contours
    for contour in contours:

        # getting current contour area
        contour_area = contourArea(contour)

        # getting contour area check bool
        area_check = (contour_area >= min_size) and (contour_area <= max_size)

        # checking current contour area
        if area_check:

            # adding current contour to blank image
            drawContours(image=image,  # noqa
                         contours=[contour],
                         contourIdx=-1,
                         color=255,
                         thickness=-1)

    # returning filtered mask image
    return image


def generate_binary_mask(input_path: str,
                         lower_threshold: int,
                         upper_threshold: int,
                         min_size: int,
                         max_size: int,
                         kernel_size: int,
                         output_path: str,
                         ) -> None:
    """
    Given a path to an image, a pixel intensity
    threshold, reads image as grayscale and
    binarizes image, applying GaussianBlur
    with given kernel size, saving binary
    mask to given output path.
    """
    # reading current image
    image = imread(input_path,
                   -1)

    # defining placeholder value for image mask
    image_mask = None


    # getting image mask
    image_mask = mask_single_img(image=image,
                                 kernel_size=kernel_size,
                                 lower_threshold=lower_threshold,
                                 upper_threshold=upper_threshold,
                                 min_size=min_size,
                                 max_size=max_size)

    # converting image to 8bit
    image_mask = image_mask.astype(np_uint8)

    # saving image
    sk_imsave(output_path,
              image_mask,
              check_contrast=False)


def generate_binary_masks(input_folder: str,
                          images_extension: str,
                          lower_threshold: int,
                          upper_threshold: int,
                          min_size: int,
                          max_size: int,
                          kernel_size: int,
                          output_folder: str,
                          ) -> None:
    """
    Given a path to a folder containing cell
    fluorescent images, generates binary
    masks, based on given thresholds.
    """
    # getting images in input folder
    images = get_files_in_folder(path_to_folder=input_folder,
                                 extension=images_extension)
    # get iterating size
    image_files_num = len(images)

    # iterating over images in input folder
    for image_index, image_name in enumerate(images, 1):

        # updating progress tracker attributes
        base_string = 'generating segmentation df #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=image_index,
                               total=image_files_num)

        # getting current image input/output paths
        current_input_path = join(input_folder,
                                  image_name)
        current_output_path = join(output_folder,
                                   image_name)

        # generating binary mask for current image
        generate_binary_mask(input_path=current_input_path,
                             lower_threshold=lower_threshold,
                             upper_threshold=upper_threshold,
                             min_size=min_size,
                             max_size=max_size,
                             kernel_size=kernel_size,
                             output_path=current_output_path
                             )


def parse_and_run(args_dict: dict,
                  ) -> None:
    """
    Extracts args from args_dict
    and runs module function.
    """


######################################################################
# argument parsing related functions

def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "generate binary masks based on fluorescence images"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        type=str,
                        required=True,
                        help='defines path to folder containing fluorescence images')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        type=str,
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folder')

    # kernel size param
    parser.add_argument('-k', '--kernel-size',
                        dest='kernel_size',
                        type=int,
                        required=False,
                        default=5,
                        help='defines kernel size for GaussianBlur')

    # lower threshold param
    parser.add_argument('-lt', '--lower-threshold',
                        dest='lower_threshold',
                        type=int,
                        required=False,
                        default=30,
                        help='defines minimum pixel intensity threshold to be used in mask generation')

    # upper threshold param
    parser.add_argument('-ut', '--upper-threshold',
                        dest='upper_threshold',
                        type=int,
                        required=False,
                        default=65536,  # maximum value for 16-bit image
                        help='defines maximum pixel intensity threshold to be used in mask generation')

    # min size param
    parser.add_argument('-mi', '--min-size',
                        dest='min_size',
                        type=int,
                        default=0,
                        help='defines minimum size (in pixels) for segmented objects')

    # max size param
    parser.add_argument('-ma', '--max-size',
                        dest='max_size',
                        type=int,
                        default=100000,
                        help='defines maximum size (in pixels) for segmented objects')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        type=str,
                        required=True,
                        help='defines output folder (folder that will contain binary masks)')

    # skip enter param
    parser.add_argument('-s', '--skip-enter',
                        dest='skip_enter',
                        action='store_true',
                        required=False,
                        help='defines whether to suppress "Enter to continue" input before execution')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


# defining main function
def main():
    """Runs main code."""

    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting lower threshold
    lower_threshold = args_dict['lower_threshold']

    # getting upper threshold
    upper_threshold = args_dict['upper_threshold']

    # getting min size
    min_size = args_dict['min_size']

    # getting max size
    max_size = args_dict['max_size']

    # getting blur kernel size
    kernel_size = args_dict['kernel_size']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_binary_masks function
    generate_binary_masks(input_folder=input_folder,
                          images_extension=images_extension,
                          lower_threshold=lower_threshold,
                          upper_threshold=upper_threshold,
                          min_size=min_size,
                          max_size=max_size,
                          kernel_size=kernel_size,
                          output_folder=output_folder
                          )

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
