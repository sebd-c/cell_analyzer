# # fix model for df showing in terminal
# identities_matrix_df = pd.DataFrame(identities_matrix,
#                                             columns=seqs_names_list_ds_01,
#                                             index=seqs_names_list_ds_01)
#         with pd.option_context('display.max_rows', None, 'display.max_columns',
#                                None):  # more options can be specified also
#             print(identities_matrix_df)
###########################################################################################
# imports
from seaborn import violinplot
import matplotlib.pyplot as plt
from pandas import read_pickle
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing

def plot_violins(input_path: str
                 ) -> None:
    # read data
    data = read_pickle(input_path)

    # split data to plot into different plotters
    tmz = data[data['tx'] == 'tmz']
    ctr = data[data['tx'] == 'ctr']

    features = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                'cyto_ecc', 'cyto_rou', 'cii', 'nuc_area', 'nuc_arbox',
                'nuc_radra', 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii',
                'label', 'cons_xgal', 'cons_sstatus'
                ]
    # plotting
    for feature in features:
        violinplot(data=data, y=feature, hue=data['tx'], inner="point")
        plt.title(f"{caminho}/{nome_arquivo}")
        plt.xlabel('Cytoplasm Irregularity Index')
        plt.ylabel('Cytoplasm Area')
        plt.grid(False)
        plt.show()
        plt.show()
        plt.close()


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
    plot_violins(input_dataframe)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
