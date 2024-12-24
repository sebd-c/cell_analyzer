# # fix model for df showing in terminal
# identities_matrix_df = pd.DataFrame(identities_matrix,
#                                             columns=seqs_names_list_ds_01,
#                                             index=seqs_names_list_ds_01)
#         with pd.option_context('display.max_rows', None, 'display.max_columns',
#                                None):  # more options can be specified also
#             print(identities_matrix_df)
###########################################################################################
# imports
from scipy.stats import kruskal
import scikit_posthocs as sp
from pandas import read_pickle
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing

def run_kruskal_wallis_h(input_path: str
                         ) -> None:
    # read data
    data = read_pickle(input_path)

    # add circularity
    data['cyto_circ'] = 1 / data['cyto_rou']
    data['nuc_circ'] = 1 / data['nuc_rou']

    # list of features to plot
    feature_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                    'cyto_ecc', 'cyto_rou', 'cyto_circ', 'cii', 'nuc_area', 'nuc_arbox',
                    'nuc_radra', 'nuc_asp', 'nuc_circ', 'nuc_ecc', 'nuc_rou', 'nii'
                    ]
    # label_dict = {0: 'Normal', 1: 'Quiescent', 2: 'Fully Senescent', 3: 'Senescent-like'}

    # data.replace({'label': label_dict}, inplace=True)

    # Loop through each feature and perform the Kruskal-Wallis H test
    results = {}
    for feature in feature_cols:
        # Group data by the label and extract feature values
        groups = data.groupby('label')[feature]
        grouped_values = [group for _, group in groups]

        # Perform Kruskal-Wallis H test
        stat, p = kruskal(*grouped_values)

        # Store results
        results[feature] = {'statistic': stat, 'p_value': p}

    # Display the results
    for feature, result in results.items():
        print(f"Feature: {feature}")
        print(f"  Kruskal-Wallis H statistic: {result['statistic']}")
        print(f"  P-value: {result['p_value']}")
        if result['p_value'] < 0.05:
            print(f"  Significant differences found between the groups.")
            # Prepare data for Dunn's test
            dunn_data = data[[feature, 'label']]

            # Perform Dunn's test
            dunn_result = sp.posthoc_dunn(dunn_data, val_col=feature, group_col='label', p_adjust='bonferroni')

            # Print Dunn's test results
            print("  Dunn's test results (p-values):")
            print(dunn_result)
        else:
            print(f"  No significant differences found between the groups.")


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
    run_kruskal_wallis_h(input_dataframe)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
