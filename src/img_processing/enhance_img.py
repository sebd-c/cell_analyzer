###########################################################################################
# imports
from os.path import join
from cv2 import imread
from cv2 import imwrite
from cv2 import createCLAHE
from cv2 import IMREAD_GRAYSCALE
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing


def enhance_single_img(og_img_path: str,
                       output_path: str,
                       img_extension: str
                       ) -> None:
    """
    This function takes the path to a poor quality image,
    and saves a new image enhanced by CLAHE
    """
    img = imread(og_img_path, IMREAD_GRAYSCALE)

    # create a CLAHE object (Arguments are optional).
    clahe = createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)

    imwrite(output_path, cl1)

    return


def enhance_dir_imgs(input_folder: str,
                     output_folder: str,
                     img_extension: str
                     ) -> None:
    """
    This function takes an input folder containing all the
    imgs to be processed, and loops through each image processing it
    """
    # getting img files in respective input folder
    img_files = get_files_in_folder(path_to_folder=input_folder,
                                    extension=img_extension)

    # iterate the processing through every img
    for file_index, img_file in enumerate(img_files, 1):
        # printing progress message
        base_string = 'generating processed img #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=len(img_files))

        # getting current image mask input/output paths
        img_input_path = join(input_folder,
                              img_file)

        # create the path to save the output path
        output_path = join(output_folder,
                           img_file)

        # runnning img processing func
        enhance_single_img(og_img_path=img_input_path,
                           output_path=output_path)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('processing complete!')

    return


def run_cellpose():
    pass


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'preprocess images in grayscale'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input_folder',
                        dest='input_folder',
                        required=True,
                        help='defines path to folder containing grayscale images')

    # images extension param
    parser.add_argument('-x', '--images_extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # output folder param
    parser.add_argument('-o', '--output_folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder to save processed imgs')

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

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting csv output path
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    enhance_dir_imgs(input_folder=images_folder,
                     output_folder=output_folder,
                     img_extension=images_extension)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
