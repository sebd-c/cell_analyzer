# generate segmentation dfs module

print('initializing...')  # noqa

# Code destined to generating
# segmentation mask based df.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import get_contours_df
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate segmentation df module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help='defines path to folder containing cellpose masks outputs')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=False,
                        default='.tif',
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output file (.csv)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def add_autophagy_col(df: DataFrame,
                      images_folder: str,
                      foci_masks_folder: str,
                      images_extension: str,
                      foci_min_area: int,
                      ring_expansion: float
                      ) -> None:
    """
    Given a crops info df, and paths to
    foci masks, adds autophagy col
    based on foci count/area.
    """
    # defining col name
    col_name = 'class'

    # emptying class col
    df[col_name] = None

    # getting rows num
    rows_num = len(df)

    # getting df rows
    df_rows = df.iterrows()

    # defining starter for current row index
    current_row_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing progress message
        base_string = 'adding autophagy col to row #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row crop info
        crop_name = row_data['crop_name']
        width = row_data['width']
        height = row_data['height']

        # getting current row crop name with extension
        crop_name_w_extension = f'{crop_name}{images_extension}'

        # getting current row crop image/foci mask path
        crop_path = join(images_folder,
                         crop_name_w_extension)
        foci_mask_path = join(foci_masks_folder,
                              crop_name_w_extension)

        # getting current crop contours df
        contours_df = get_contours_df(image_name=crop_name,
                                      image_path=foci_mask_path)

        # filtering df by min foci area
        contours_df = contours_df[contours_df['area'] >= foci_min_area]

        # getting current crop foci count/area
        foci_count = len(contours_df)
        foci_area_col = contours_df['area']
        foci_area_mean = foci_area_col.mean()

        # getting current crop autophagy level
        # current_class = get_autophagy_level(foci_count=foci_count,
        #                                     foci_area_mean=foci_area_mean,
        #                                     foci_count_threshold=FOCI_COUNT_THRESHOLD,
        #                                     foci_area_mean_threshold=FOCI_AREA_MEAN_THRESHOLD)
        # TODO: remove once test completed!
        current_ratio = '>1' if current_ratio >= 1 else '<1'
        current_class = f'{foci_count}_{current_ratio}'

        # updating current row data
        df.at[row_index, col_name] = current_class

        # updating current row index
        current_row_index += 1


def generate_segmentation_df(input_folder: str,
                             images_extension: str,
                             output_path: str,
                             ) -> None:
    """
    Given a path to a folder containing binary images
    containing cytoplasm segmentation masks, extracts
    data from masks, saving analysis data frame to
    given output path.
    """


    # saving segmentation df
    segmentation_df.to_csv(output_path,
                         index=False)

    # printing execution message
    print(f'output saved to {output_path}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_segmentation_df function
    generate_segmentation_df(crops_file=crops_file,
                             images_folder=images_folder,
                             foci_masks_folder=foci_masks_folder,
                             images_extension=images_extension,
                             output_path=output_path,
                             foci_min_area=foci_min_area,
                             ring_expansion=ring_expansion)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
