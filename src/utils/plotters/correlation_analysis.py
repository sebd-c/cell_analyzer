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
from seaborn import color_palette
import matplotlib.pyplot as plt
from pandas import read_pickle
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters
from sklearn.manifold import TSNE


################################################################################################
# module of aux functions related to img preprocessing


def plot_tsne(input_path: str
              ) -> None:
    # read data
    data = read_pickle(input_path)

    data_wo_labels = data.drop(columns=['image_name', 'cons_xgal', 'cons_sstatus', 'label', 'tx', 'obj_id'])

    data_embedded = TSNE(n_components=2,
                         learning_rate='auto',
                         init='random',
                         perplexity=3,
                         verbose=1).fit_transform(data_wo_labels)

    plt.figure(figsize=(16, 10))
    scatterplot(
        x=data_embedded[:, 0], y=data_embedded[:, 1],
        hue=data['label'],
        palette=color_palette("bright"),
        data=data,
        alpha=0.3
    )
    plt.legend(title='Classes', labels=['Normal', 'Quiescent', 'Fully senescent', 'Senescent-like'])
    plt.show()
    plt.close()


def plot_distributions(input_path: str
                       ) -> None:
    # read data
    data = read_pickle(input_path)

    # split data to plot into different plotters
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


# sns.displot(penguins, x="flipper_length_mm", hue="species")


# def plot_pixint_per_gt(df: DataFrame) -> None:
#     scatterplot(data=df, x="ii", y="area", hue="xgal")
#     plt.xlabel('Irregularity Index')
#     plt.ylabel('Area')
#     plt.grid(False)
#     plt.show()
#     plt.show()
#     plt.close()


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
    plot_tsne(input_dataframe)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
