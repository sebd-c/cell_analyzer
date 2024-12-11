# # fix model for df showing in terminal
# identities_matrix_df = pd.DataFrame(identities_matrix,
#                                             columns=seqs_names_list_ds_01,
#                                             index=seqs_names_list_ds_01)
#         with pd.option_context('display.max_rows', None, 'display.max_columns',
#                                None):  # more options can be specified also
#             print(identities_matrix_df)
###########################################################################################
# imports
from argparse import ArgumentParser
from cv2 import imread
from cv2 import contourArea
from cv2 import findContours
from cv2 import split
from cv2 import imwrite
from cv2 import COLOR_BGR2RGB
from cv2 import cvtColor
from cv2 import CHAIN_APPROX_NONE
from cv2 import RETR_EXTERNAL
from numpy import uint8 as np_uint8
from numpy import ndarray
from pandas import concat
from pandas import DataFrame
from os.path import join
from numpy import max
from numpy import min
from numpy import mean
from numpy import median
from numpy import sum
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


################################################################################################
# module of aux functions related to img preprocessing


def make_image_contours_df(mask_name: str,
                           mask_path: str,
                           og_img_path: str,
                           p_img_path: str,
                           overlays_output_folder: str
                           ) -> DataFrame:
    """
    Given a path to a binary image,
    finds contours and
    returns data frame containing
    contours desired infos.
    """

    # read the rgb channel
    phase_image_cv2 = imread(p_img_path,
                             -1)

    # convert it to the accurate colors
    phase_image = cvtColor(phase_image_cv2, COLOR_BGR2RGB)

    # separate channels
    phase_red, phase_green, phase_blue = split(phase_image)

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'adds manual labeled columns to the df'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input csv path
    parser.add_argument('-i', '--input-dataframe',
                        dest='input_dataframe',
                        required=True,
                        help='defines path to input csv containing all infos')

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
    input_dataframe = args_dict['input_dataframe']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    plot_reds(input_dataframe)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
