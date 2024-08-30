print('importing required libraries...')  # noqa
from argparse import ArgumentParser
from pandas import DataFrame
from pandas import read_csv
from os.path import join
from src.utils.aux_funcs import print_execution_parameters

print('all required libraries successfully imported.')  # noqa


#####################################################################
# cycle related functions


def get_cell_cycle_simple(df_path: str,
                          output_folder: str) -> None:
    """
    Given a fornma-like df, returns it
    with a new cell cycle column, based on
    which median pixel intensity was higher.
    """

    # reading df
    base_df = read_csv(df_path)

    # copy df
    updated_df = base_df

    # add empty column
    # for cycle
    updated_df['cell_cycle'] = None

    # for treatment
    updated_df['treatment'] = None

    # iterate over df whilst adding cell cycle class
    for row_index, row_data in updated_df.iterrows():

        # defining placeholder variables
        cycle_state = None
        red_median = row_data['Median_red']
        green_median = row_data['Median_green']

        # checking if red median is higher
        if red_median > green_median and red_median > 0:

            # then, cell cycle must be 'G1' (red)
            cycle_state = 'G1'

        # if green value higher than red
        elif row_data['Median_green'] > row_data['Median_red']:

            # then, cell cycle must be 'G2' (green)
            cycle_state = 'G2'

        # if equal, then S?
        elif row_data['Median_green'] == row_data['Median_red']:

        # then, cell cycle must be 'G2' (green)
        cycle_state = 'S'

        else:

            # then it's probably an image technical problem
            cycle_state = '-'

        # add it to the csv
        updated_df.at[row_index, 'cell_cycle'] = cycle_state

        # treatment part
        # get image name
        image_name = row_data['Image_name_red']

        # get the letter of the well
        # gettin a list of the parts from the name
        split_name = image_name.split("_")

        # getting the part which contains the well letter
        numbered_well = split_name[1]

        # taking out the number so the if is easier
        well = numbered_well[0]

        # TODO: voltar aqui e deixar isso em estrutura de dict

        # if well A, ctr
        if well == 'A':
            treatment_condition = 'ctr'

        # else is treatment
        else:
            treatment_condition = 'tmz'

        # add it to the csv
        updated_df.at[row_index, 'treatment'] = treatment_condition

    for condition, group in updated_df.groupby('treatment'):

        file_name = f'updated_output_{condition}.csv'

        # create the path to save the output path
        output_path = join(output_folder,
                           file_name)

        # saving df
        group.to_csv(output_path, index=False)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('analysis complete!')


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
    # output path param
    parser.add_argument('-o', '--output_folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output file (.csv)')

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

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running function to update the df
    get_cell_cycle_simple(df_path=input_path,
                          output_folder=output_folder)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
