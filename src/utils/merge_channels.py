# merge channels module

print('initializing...')  # noqa

# Code destined to merging channels
# for forNMA integration.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from cv2 import imwrite
from cv2 import addWeighted
from cv2 import IMREAD_GRAYSCALE
from os.path import join
from numpy import add as np_add
from argparse import ArgumentParser
from numpy import uint8 as np_uint8
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_files_in_folder

print('all required libraries successfully imported.')  # noqa


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'merge channels module - merge red/green data from RGB "as displayed" Incucyte images'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # red images folder path param
    parser.add_argument('-r', '--red-folder',
                        dest='red_folder',
                        type=str,
                        help='defines path to folder containing red images (8-bit .tif)',
                        required=True)

    # red images folder path param
    parser.add_argument('-g', '--green-folder',
                        dest='green_folder',
                        type=str,
                        help='defines path to folder containing green images (8-bit .tif)',
                        required=True)

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        type=str,
                        help='defines path to folder which will contain merged (8-bit .tif) images',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


######################################################################
# defining auxiliary functions


def merge_single_image(nuc_path: str,
                       cyto_path: str,
                       output_path: str
                       ) -> None:
    """
    Given a path to nuclei/cyto images,
    merges them into single array,
    saving it to given output path.
    """

    # reading red/green images
    nucleus = imread(nuc_path, IMREAD_GRAYSCALE)
    cytoplasm = imread(cyto_path, IMREAD_GRAYSCALE)

    # add(or blend) the images
    merged_img = addWeighted(nucleus, 0.5, cytoplasm, 0.5, 0)

    # saving merged image
    imwrite(output_path,
            merged_img)

    # # getting image halves (important to join them later and max still be 255)
    # red_half = red / 2
    # green_half = green / 2
    #
    # # merging images
    # merged = np_add(red_half, green_half)
    #
    # # converting int type
    # merged = merged.astype(np_uint8)

    return


def merge_multiple_images(nuclei_folder: str,
                          cyto_folder: str,
                          output_folder: str,
                          img_extension: str
                          ) -> None:
    """
    Given a path to folders containing
    red/green images, merges images into
    single array, saving output image in
    given output folder.
    """
    # getting images in input folder
    print('getting images in input folder...')
    nuclei_images = get_files_in_folder(path_to_folder=nuclei_folder,
                                        extension=img_extension)

    # getting images total
    images_num = len(nuclei_images)

    # starting counter for current image index
    current_image_index = 1

    # iterating over images
    for nuclei_image_name in nuclei_images:
        # printing execution message
        progress_base_string = 'merging image #INDEX# of #TOTAL#'
        print_progress_message(base_string=progress_base_string,
                               index=current_image_index,
                               total=images_num)

        # getting current nuclei image input paths
        nucleus_path = join(nuclei_folder,
                            nuclei_image_name)

        # make conversion for cytoplasm path
        cyt_image_name = nuclei_image_name.replace('red', 'green')

        # getting current cytoplasm image input paths
        cyto_path = join(cyto_folder,
                         cyt_image_name)

        # make conversion for joined channels path
        joined_image_name = nuclei_image_name.replace('red', 'joined')

        # getting current image save path
        save_path = join(output_folder,
                         joined_image_name)

        # running merge_single_image function
        merge_single_image(nuc_path=nucleus_path,
                           cyto_path=cyto_path,
                           output_path=save_path)

        # updating current image index
        current_image_index += 1

    # printing execution message
    f_string = f'all {images_num} images merged!'
    print(f_string)

    return


######################################################################
# defining main function


def main():
    """
    Gets arguments from cli and runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting red images folder param
    red_folder = args_dict['red_folder']

    # getting green images folder param
    green_folder = args_dict['green_folder']

    # getting output images folder param
    output_folder = args_dict['output_folder']

    # TODO: adicionar param de img extension

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running merge_channels function
    merge_multiple_images(output_folder=output_folder,
                          nuclei_folder=red_folder,
                          cyto_folder=green_folder,
                          img_extension='.tif')


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
