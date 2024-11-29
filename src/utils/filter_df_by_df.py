# # fix model for df showing in terminal
# identities_matrix_df = pd.DataFrame(identities_matrix,
#                                             columns=seqs_names_list_ds_01,
#                                             index=seqs_names_list_ds_01)
#         with pd.option_context('display.max_rows', None, 'display.max_columns',
#                                None):  # more options can be specified also
#             print(identities_matrix_df)
###########################################################################################
# imports
from seaborn import scatterplot
import matplotlib.pyplot as plt
from pandas import DataFrame
from os.path import join
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
    infos_df = read_csv(infos_df_path)

    labels_df = read_csv(labels_df_path)

    # now that we have both the complete df and the
    # filter the df using a left outer join
    labeled_df = infos_df.merge(labels_df,
                                on=['image_name', 'contour_index'],
                                how='left',
                                indicator=True
                                )
    # followed by the exclusion of the rows that were not shared
    labeled_df = labeled_df[labeled_df['_merge'] == 'both']

    # exclude the newly created column for cleanness
    labeled_df = labeled_df.drop(columns='_merge')

    labeled_df['cons_xgal'] = None
    labeled_df['cons_sstatus'] = None
    labeled_df['labeled_class'] = None

    # xgal_e	s_status_e	xgal_d	s_status_d	xgal_h	s_status_h
    for row_index, row in labeled_df.iterrows():
        print(row['xgal_d'])
        print(row['xgal_e'])
        print(row['xgal_h'])
        print(row['xgal_d'] + row['xgal_e'] + row['xgal_h'])
        # get consensus xgal
        if row['xgal_d'] + row['xgal_e'] + row['xgal_h'] == 0 or 1:
            # one or less annotator marked an object as xgal negative
            labeled_df.loc[row_index, 'cons_xgal'] = 0
            # labeled_df.at[row_index, 'cons_xgal'] = 0
            print(row['cons_xgal'])
        elif row['xgal_d'] + row['xgal_e'] + row['xgal_h'] == 2 or 3:
            # two or more annotators marked an object as xgal positive
            labeled_df.loc[row_index, 'cons_xgal'] = 1
            print(row['cons_xgal'])
        exit()
        # get consensus senescent status
        if row['s_status_d'] + row['s_status_e'] + row['s_status_h'] == 0 or 1:
            # one or less annotator marked an object as not senescent like
            labeled_df.at[row_index, 'cons_sstatus'] = 0
        elif row['xgal_d'] + row['xgal_e'] + row['xgal_h'] == 2 or 3:
            # two or more annotators marked an object  senescent like
            labeled_df.at[row_index, 'cons_sstatus'] = 1
        
        # after obtaining the consensus features, we further obtain the stablished classes 
        # label classes defined by
        # if not xgal and not senescent like -> normal/growing cell = 0
        # if xgal and not senescent like -> quiescent = 1
        # if not xgal and senescent like -> blocked lisossomal senescence = 2
        # if xgal and senescent like -> full senescent phenotype = 3
        # make consensus columns
        # for each row
        if row['cons_xgal'] == 0 and row['cons_sstatus'] == 0:
            # condition attributed to growing tumor cells
            labeled_df.at[row_index, 'label'] = 0
        elif row['cons_xgal'] == 1 and row['cons_sstatus'] == 0:
            # condition attributed to quiescent cells
            labeled_df.at[row_index, 'label'] = 1
        elif row['cons_xgal'] == 0 and row['cons_sstatus'] == 1:
            # condition attributed to senescent like cells wo xgal
            labeled_df.at[row_index, 'label'] = 2
        elif row['cons_xgal'] == 1 and row['cons_sstatus'] == 1:
            # condition attributed to full senescent phenotype cells
            labeled_df.at[row_index, 'label'] = 3

    # create the path to save the output path
    output_path = join(output_folder,
                       'labeled_contours.csv')

    # saving df
    labeled_df.to_csv(output_path, index=False)

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
    parser.add_argument('-d', '--full-dataframe',
                        dest='full_dataframe',
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
    full_dataframe = args_dict['full_dataframe']

    # getting images extension
    label_dataframe = args_dict['label_dataframe']

    # getting images extension
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    merge_label_df(infos_df_path=full_dataframe,
                   labels_df_path=label_dataframe,
                   output_folder=output_folder
                   )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
