# # fix model for df showing in terminal
# identities_matrix_df = pd.DataFrame(identities_matrix,
#                                             columns=seqs_names_list_ds_01,
#                                             index=seqs_names_list_ds_01)
#         with pd.option_context('display.max_rows', None, 'display.max_columns',
#                                None):  # more options can be specified also
#             print(identities_matrix_df)
###########################################################################################
# imports
from seaborn import displot
from seaborn import scatterplot
from seaborn import kdeplot
from seaborn import color_palette
import matplotlib.pyplot as plt
from pandas import read_pickle
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing

def plot_distributions(input_path: str
                       ) -> None:
    # read data
    data = read_pickle(input_path)

    # list of features to plot
    feature_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                    'cyto_ecc', 'cyto_rou', 'cii', 'nuc_area', 'nuc_arbox',
                    'nuc_radra', 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii'
                    ]
    label_dict = {0: 'Normal', 1: 'Quiescent', 2: 'Fully Senescent', 3: 'Senescent-like'}

    data.replace({'label': label_dict}, inplace=True)

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
    scatterplot(data=data,
                x='cyto_area',
                y='nuc_area',
                hue='label',
                palette=color_palette(['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']),
                hue_order=['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']
                )
    plt.xlabel('Cytoplasmic Area')
    plt.ylabel('Nuclear Area')
    plt.xlim(0, 200000)
    plt.grid(False)
    plt.show()

    scatterplot(data=data,
                x='cyto_ecc',
                y='nuc_ecc',
                hue='label',
                palette=color_palette(['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']),
                hue_order=['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']
                )
    plt.xlabel('Cytoplasmic Eccentricity')
    plt.ylabel('Nuclear Eccentricity')
    plt.grid(False)
    plt.show()

    scatterplot(data=data,
                x='cyto_area',
                y='cyto_ecc',
                hue='label',
                palette=color_palette(['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']),
                hue_order=['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']
                )
    plt.xlabel('Cytoplasmic Area')
    plt.ylabel('Cytoplasmic Eccentricity')
    plt.grid(False)
    plt.show()

    scatterplot(data=data,
                x='nuc_area',
                y='nuc_ecc',
                hue='label',
                palette=color_palette(['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']),
                hue_order=['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']
                )
    plt.xlabel('Nuclear Area')
    plt.ylabel('Nuclear Eccentricity')
    plt.grid(False)
    plt.show()

    # # Iterating through axes and names
    # for feature in feature_cols:
    #     ax = kdeplot(data, x=feature, hue='label')
    #     plt.xlabel(axis_dict[feature])
    #
    #     plt.grid(False)
    #     plt.show()


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
