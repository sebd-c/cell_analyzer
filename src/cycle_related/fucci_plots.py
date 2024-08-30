print('importing required libraries...')  # noqa
from argparse import ArgumentParser
from pandas import DataFrame
from pandas import read_csv
from src.utils.aux_funcs import print_execution_parameters

print('all required libraries successfully imported.')  # noqa


#####################################################################
# cycle related functions


def get_cell_cycle_simple(df_path: str
                          ) -> DataFrame:
    """
    Given a fornma-like df, returns it
    with a new cell cycle column, based on
    which median pixel intensity was higher.
    """

    # reading df
    base_df = read_csv(df_path)

    # add empty column
    base_df['cell_cycle'] = None

    # iterate over df whilst adding cell cycle class
    for row_index, row_data in base_df.iterrows():

        # defining placeholder variables
        cell_cycle = None
        red_median = row_data['Median_red']
        green_median = row_data['Median_green']

        # checking if red median is higher
        if red_median >= green_median and red_median > 0:

            # then, cell cycle must be 'G1' (red)
            cell_cycle = 'G1'

        # if green value higher than red
        elif row_data['Median_green'] > row_data['Median_red']:

            # then, cell cycle must be 'G2' (green)
            cell_cycle = 'G2'

        else:

            # then, cell cycle must be S, unlikely
            cell_cycle = None

    # returning updated df
    return base_df


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'updates a fornma-like table in order to add cycle column'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input_path',
                        dest='input_path',
                        required=True,
                        help='defines path to fornma-like .csv containing original segmented nucleus df')

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

    # getting input folder
    input_path = args_dict['input_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running function to update the df
    get_cell_cycle_simple(df_path=input_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
