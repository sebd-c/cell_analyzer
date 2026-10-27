# # fix model for df showing in terminal
# identities_matrix_df = pd.DataFrame(identities_matrix,
#                                             columns=seqs_names_list_ds_01,
#                                             index=seqs_names_list_ds_01)
#         with pd.option_context('display.max_rows', None, 'display.max_columns',
#                                None):  # more options can be specified also
#             print(identities_matrix_df)
###########################################################################################
# imports
from os.path import join
from os.path import basename
from os.path import splitext
from pandas import read_pickle
from argparse import ArgumentParser
from src._execution_formatting import enter_to_continue
from src._execution_formatting import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing


def add_group_column(infos_df_path: str,
                     output_folder: str,
                     ) -> None:
    """
    This function takes the path to a df,
    and saves a new df with a "group" column,
    taken from the [1] element of the image name split by "_"
    (e.g. 01_f+c+_t50.tif -> f+c+)
    """
    # read dataframe
    infos_df = read_pickle(infos_df_path)

    # getting the group from the image name
    infos_df['group'] = infos_df['cyto_image_name'].str.split('_').str[1]

    # create the path to save the output path
    input_name = splitext(basename(infos_df_path))[0]
    output_path = join(output_folder,
                       f'{input_name}_grouped.pickle')

    # saving df
    infos_df.to_pickle(output_path)

    # printing execution message
    print(f'output saved to {output_path}')
    print('analysis complete!')


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'adds a "group" column to the df, taken from the image name'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input csv path
    parser.add_argument('-i', '--full-dataframe',
                        dest='input_dataframe',
                        required=True,
                        help='defines path to input pickle containing all infos')

    # output path
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='output folder')

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

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to add the group column
    add_group_column(infos_df_path=input_dataframe,
                     output_folder=output_folder
                     )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
