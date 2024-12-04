# generate segmentation dfs module
import numpy as np

print('initializing...')  # noqa

# Code destined to generating
# segmentation mask based df.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from argparse import ArgumentParser
from cv2 import pointPolygonTest
from pandas import concat
from pandas import DataFrame
from pandas import read_pickle
from os.path import join
from src.utils.aux_funcs import print_execution_parameters
from src.utils.merge_channels import merge_multiple_images

print('all required libraries successfully imported.')  # noqa


#####################################################################
# module specific aux functions

def link_tx_dfs(drug_df: DataFrame,
                ctr_df: DataFrame,
                label_01: str,
                label_02: str,
                ) -> DataFrame:
    """
    given a df from cytoplasm segmentation,
    and its respective nuclei df segmentation,
    associates each nucleus with its parent cytoplasm
    into a new df
    """

    # add new id columns to each group
    drug_df['tx'] = label_01

    ctr_df['tx'] = label_02

    concat_tx_df = concat([drug_df, ctr_df], ignore_index=True)

    # make object identification column
    concat_tx_df['obj_id'] = concat_tx_df.index

    return concat_tx_df


def make_tx_output(drug_csv_input_path: str,
                   ctr_csv_input_path: str,
                   label_01: str,
                   label_02: str,
                   output_folder: str
                   ) -> None:
    """
    Given a path to a cytoplasm df
    a nuclei df, the parent images w overlays
    of both, joins this two dfs
    relating each nucleus to a parent
    and saves the results in the output folder,
    as well as a merged version of the overlayed images.
    """

    # read the .csv tables of both making them into a DataFrame object
    drug_df = read_pickle(drug_csv_input_path)

    ctr_df = read_pickle(ctr_csv_input_path)

    # create the path to save the output path
    output_path = join(output_folder,
                       'joined_tx_df.pickle'
                       )

    # linking the dfs
    linked_df = link_tx_dfs(drug_df=drug_df,
                            ctr_df=ctr_df,
                            label_01=label_01,
                            label_02=label_02
                            )

    # saving output
    linked_df.to_pickle(output_path)

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'join 2 different treatments dataframes and returns an output df'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param (drug df)
    parser.add_argument('-d', '--drug_dataframe_path',
                        dest='drug_dataframe_path',
                        required=True,
                        help='defines path to folder containing .pickle of tratment condition')

    # input folder param (drug df)
    parser.add_argument('-c', '--ctr_dataframe_path',
                        dest='ctr_dataframe_path',
                        required=True,
                        help='defines path to folder containing overlayed nuclei images')

    # label for drug df
    parser.add_argument('-l1', '--first_label',
                        dest='drug_label',
                        required=True,
                        help='defines str label for drug condition')

    # label for ctr df
    parser.add_argument('-l2', '--second_label',
                        dest='ctr_label',
                        required=True,
                        help='defines str label for ctr condition')

    # overlays output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder (overlays)')

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

    # getting drug df path
    drug_csv_input_path = args_dict['drug_dataframe_path']

    # getting ctr df path
    ctr_csv_input_path = args_dict['ctr_dataframe_path']

    # getting drug label
    label_01 = args_dict['drug_label']

    # getting ctr label
    label_02 = args_dict['ctr_label']

    # getting overlays output path
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # runnning join
    make_tx_output(drug_csv_input_path=drug_csv_input_path,
                   ctr_csv_input_path=ctr_csv_input_path,
                   label_01=label_01,
                   label_02=label_02,
                   output_folder=output_folder
                    )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
