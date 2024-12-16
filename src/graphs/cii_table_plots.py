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
from pandas import read_csv
from pandas import read_pickle
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing


def plot_reds(input_path: str
              ) -> None:
    # read data
    data = read_pickle(input_path)
    scatterplot(data=data, x="cyto_red_max", y="nuc_red_max", hue="tx")
    plt.xlabel('Max Red Pixel Value from Cytoplasm')
    plt.ylabel('Max Red Pixel Value from Cytoplasm')
    plt.grid(False)
    plt.show()
    plt.show()
    plt.close()


def plot_cellmorph(input_path: str
                   ) -> None:
    # read data
    data = read_pickle(input_path)

    # split data to plot into different graphs
    tmz = data[data['tx'] == 'tmz']
    ctr = data[data['tx'] == 'ctr']

    # plotting
    plot = scatterplot(data=tmz,
                       x='cii',
                       y='cyto_area',
                       hue='nii',
                       palette='dark:#5A9_r',
                       size='nuc_area')
    # Set axis limits
    plt.xlim(0, 200)  # Set x-axis limits
    plt.ylim(0, 200000)  # Set y-axis limits
    plt.title('CellMorph TMZ')
    plt.xlabel('Cytoplasm Irregularity Index')
    plt.ylabel('Cytoplasm Area')
    plt.grid(False)
    plt.show()
    plt.show()
    plt.close()

    plot = scatterplot(data=ctr,
                       x='cii',
                       y='cyto_area',
                       hue='nii',
                       palette='dark:#5A9_r',
                       size='nuc_area')
    # Set axis limits
    plt.xlim(0, 200)  # Set x-axis limits
    plt.ylim(0, 200000)  # Set y-axis limits
    plt.title('CellMorph CTR')
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
    plot_cellmorph(input_dataframe)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
