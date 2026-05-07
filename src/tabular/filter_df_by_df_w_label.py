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
from pandas import read_pickle
from pandas import read_csv
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing


def merge_label_df(infos_df_path: str,
                   labels_df_path: str,
                   output_folder: str,
                   ) -> None:
    """
    This function takes the path to an unfiltered df,
    and saves a new df after using a filter dictionary
    to exclude the objects that were wrongfully segmented
    """
    # read dataframes
    infos_df = read_pickle(infos_df_path)

    labels_df = read_csv(labels_df_path)

    # now that we have both the complete df and the
    # filter the df using a left outer join
    labeled_df = infos_df.merge(labels_df,
                                on=['cyto_image', 'cyto_index'],
                                how='left',
                                indicator=True
                                )
    # followed by the exclusion of the rows that were not shared
    labeled_df = labeled_df[labeled_df['_merge'] == 'both']

    # exclude the newly created column for cleanness
    labeled_df = labeled_df.drop(columns='_merge')

    # after obtaining the consensus features, we further obtain the stablished classes
    # label classes defined by
    # if not xgal and not senescent like -> normal/growing cell (0, 0) = 0
    # if xgal and not senescent like -> quiescent (1, 0) = 1
    # if xgal and senescent like -> full senescent phenotype (1, 1) = 2
    # if not xgal and senescent like -> blocked lisossomal senescence (0, 1) = 3

    # first sum up the two consensus columns
    labeled_df['label'] = labeled_df['cons_xgal'] + labeled_df['cons_morph_status']

    # then transform the result into a label
    # we'll be using most of the sum's results, only changing one of the onesies
    labeled_df.loc[(labeled_df['label'] == 1) & (labeled_df['cons_morph_status'] == 1), 'label'] = 3

    # create the path to save the output path
    output_path = join(output_folder,
                       'refiltered_labeled_cytnuc_contours.pickle')

    # saving df
    labeled_df.to_pickle(output_path)

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
    description = 'adds manual labeled columns to the df'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input csv path
    parser.add_argument('-i', '--full-dataframe',
                        dest='input_dataframe',
                        required=True,
                        help='defines path to input csv containing all infos')

    # input labels csv path
    parser.add_argument('-l', '--label-dataframe',
                        dest='label_dataframe',
                        required=True,
                        help='defines path to input csv containing manual labels')

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

    # getting images extension
    label_dataframe = args_dict['label_dataframe']

    # getting images extension
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    merge_label_df(infos_df_path=input_dataframe,
                   labels_df_path=label_dataframe,
                   output_folder=output_folder
                   )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
