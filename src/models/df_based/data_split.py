# imports module
from os.path import join
from argparse import ArgumentParser
from pandas import read_pickle
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import enter_to_continue


##########################################################################################
# auxiliary functions


def split_data(input_path: str,
               output_folder: str
               ) -> None:
    # read data
    data = read_pickle(input_path)

    # set columns of interest
    feature_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                    'cyto_ecc', 'cyto_rou', 'cii', 'nuc_area', 'nuc_arbox',
                    'nuc_radra', 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii',
                    ]

    # drop the ground truth column
    X = data.drop(['label'], axis=1)
    X = X[feature_cols]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42,
                                                        stratify=y)


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'runs a decision tree model'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input csv path
    parser.add_argument('-i', '--input-path',
                        dest='input_path',
                        required=True,
                        help='defines path to input csv cell informations')

    # output folder
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output files')

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

    # getting input csv path
    input_path = args_dict['input_path']

    # getting images extension
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running function to preprocess images in a folder
    run_dt_model_senescence(input_path=input_path,
                            output_folder=output_folder
                            )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
