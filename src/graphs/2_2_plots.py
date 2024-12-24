# # fix model for df showing in terminal
# identities_matrix_df = pd.DataFrame(identities_matrix,
#                                             columns=seqs_names_list_ds_01,
#                                             index=seqs_names_list_ds_01)
#         with pd.option_context('display.max_rows', None, 'display.max_columns',
#                                None):  # more options can be specified also
#             print(identities_matrix_df)
###########################################################################################
# imports

from itertools import combinations
from seaborn import scatterplot
from seaborn import color_palette
import matplotlib.pyplot as plt
from pandas import read_pickle
from sklearn.metrics import silhouette_score
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing


def plot_distributions(input_path: str
                       ) -> None:
    # read data
    data = read_pickle(input_path)

    # set columns of interest
    feature_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                    'cyto_ecc', 'cyto_rou', 'cii', 'nuc_area', 'nuc_arbox',
                    'nuc_radra', 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii',
                    ]

    axis_dict = {'cyto_area': 'Area',
                 'cyto_arbox': 'Area box',
                 'cyto_radra': 'Radius ratio',
                 'cyto_asp': 'Aspect',
                 'cyto_ecc': 'Eccentricity',
                 'cyto_rou': 'Roundess',
                 'cii': 'CII',
                 'nuc_area': 'Area',
                 'nuc_arbox': 'Area box',
                 'nuc_radra': 'Radius ratio',
                 'nuc_asp': 'Aspect',
                 'nuc_ecc': 'Eccentricity',
                 'nuc_rou': 'Roundess',
                 'nii': 'NII'
                 }

    comb = combinations(feature_cols, 2)

    for i in list(comb):
        X = data[[i[0], i[1]]]
        labels = data['label']

        # Compute silhouette score
        sil_score = silhouette_score(X, labels)

        print(f"Feature: {i[0]}")
        print(f"Feature: {i[1]}")
        print(f"Silhouette score: {sil_score}")

        ax = scatterplot(data=data,
                         x=data[i[0]],
                         y=data[i[1]],
                         hue='label',
                         palette=color_palette(['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']),
                         hue_order=['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']
                         )
        plt.xlabel(axis_dict[i[0]])
        plt.ylabel(axis_dict[i[1]])
        plt.grid(False)
        plt.text(1, 1,
                 f"Silhouette score: {sil_score: .3f}",
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax.transAxes)
        plt.show()
        plt.savefig(f"{i[0]}_{i[1]}.pdf")

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
    # plot_reds(input_dataframe)
    plot_distributions(input_dataframe)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
