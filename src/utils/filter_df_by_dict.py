###########################################################################################
# imports
from os.path import join
from pandas import DataFrame
from pandas import merge
from pandas import concat
from yaml import safe_load
from pandas import read_csv
from pandas import read_pickle
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing


def filter_df(unfiltered_df: DataFrame,
              dict_path: str,
              ) -> DataFrame:
    """
    This function takes the path to an unfiltered df,
    and saves a new df after using a filter dictionary
    to exclude the objects that were wrongfully segmented
    """
    # open parameters file
    with open(dict_path, 'r') as open_file:
        filter_params = safe_load(open_file)

    # turn it into a dict
    filter_dict = filter_params['data_dict']

    # make the dictionary into a df
    # params_df = DataFrame.from_dict(filter_dict)
    params_df = DataFrame([(k, v) for k, vs in filter_dict.items() for v in vs],
                          columns=['image_name', 'contour_index'])

    # now that we have both the complete df and the
    # filter the df using a left outer join
    filtered_df = unfiltered_df.merge(params_df,
                                      on=['image_name', 'contour_index'],
                                      how='left',
                                      indicator=True
                                      )
    # followed by the exclusion of the rows that were shared
    filtered_df = filtered_df[filtered_df['_merge'] == 'left_only']

    # exclude the newly created column for cleanness
    filtered_df = filtered_df.drop(columns='_merge')

    # returns the filtered df
    return filtered_df


def filter_data(tto_input_path: str,
                ctr_input_path: str,
                ctr_filter_path: str,
                tto_filter_path: str,
                output_folder: str
                ) -> None:
    """
    This function takes an input folder containing all the
    imgs to be processed, and loops through each image processing it
    """

    # reading treatment df
    tmz_df = read_pickle(tto_input_path)

    # adding treatment col
    tmz_df['tto'] = 'tmz'

    # reading control df
    ctr_df = read_pickle(ctr_input_path)

    # adding treatment col
    ctr_df['tto'] = 'ctr'

    # applying filter function
    # on ctr
    filtered_ctr = filter_df(ctr_df, ctr_filter_path)

    # on tto
    filtered_tto = filter_df(tmz_df, tto_filter_path)

    # concatenate both dataframes
    concat_df = concat([filtered_tto, filtered_ctr])

    # create the path to save the output path
    output_path = join(output_folder,
                       'filtered_df.pickle')

    # saving df
    concat_df.to_pickle(output_path, index=False)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('processing complete!')

    return


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'filters the dfs from ctr and tmz to modeling proccess'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input csv path for treatment
    parser.add_argument('-t', '--tto_input_path',
                        dest='tto_input_path',
                        required=True,
                        help='defines path to input csv path for treatment')

    # input csv path for control
    parser.add_argument('-c', '--ctr_input_path',
                        dest='ctr_input_path',
                        required=True,
                        help='defines path to input csv path for ctr')

    # input csv path for treatment
    parser.add_argument('-tp', '--tto_parameter_path',
                        dest='tto_parameter_path',
                        required=True,
                        help='defines path to input parameters in yaml format for treatment')

    # input csv path for control
    parser.add_argument('-cp', '--ctr_parameter_path',
                        dest='ctr_parameter_path',
                        required=True,
                        help='defines path to input parameters in yaml format for control')

    # output folder param
    parser.add_argument('-o', '--output_folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder to save the filtered df')

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
    tto_input_path = args_dict['tto_input_path']

    # getting images extension
    ctr_input_path = args_dict['ctr_input_path']

    # getting tmz parameters
    tto_parameter_path = args_dict['tto_parameter_path']

    # getting ctr parameters
    ctr_parameter_path = args_dict['ctr_parameter_path']

    # getting csv output path
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    filter_data(tto_input_path=tto_input_path,
                ctr_input_path=ctr_input_path,
                ctr_filter_path=ctr_parameter_path,
                tto_filter_path=tto_parameter_path,
                output_folder=output_folder
                )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
