# imports module
from os.path import join
from argparse import ArgumentParser
from umap import UMAP
from numpy import array
from os.path import join
from numpy import ndarray
from pandas import DataFrame
from pandas import read_pickle
from plotly.express import scatter
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import enter_to_continue


##########################################################################################
# auxiliary functions
def add_handcrafted_features_col(df: DataFrame) -> None:
    """
    Given an encodings df, adds
    handcrafted features col.
    """
    # defining new col name
    handcrafted_features_col = 'handcrafted_features'

    # defining cols to be used as handcrafted features
    # TODO: update this function to use add_col from aux_funcs
    features_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                    'cyto_ecc', 'cyto_rou', 'cii', 'nuc_area', 'nuc_arbox',
                    'nuc_radra', 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii']

    # defining placeholder value for new col
    df[handcrafted_features_col] = None

    # getting df rows
    df_rows = df.iterrows()

    # iterating over df rows
    for row_index, row_data in df_rows:

        # defining placeholder value for feature vector list
        feature_vector_list = []

        # iterating over feature cols
        for feature_col in features_cols:

            # getting current feature
            current_feature = row_data[feature_col]

            # appending current feature to feature vector list
            feature_vector_list.append(current_feature)

        # converting data types
        feature_vector_arr = array(feature_vector_list)

        # updating row data
        df.at[row_index, handcrafted_features_col] = feature_vector_arr


def get_encodings_arr(df: DataFrame,
                      col: str
                      ) -> ndarray:
    """
    Given an encodings df, returns
    encodings array, based on given
    column.
    """
    # getting current column
    encodings_col = df[col]

    # extracting encodings from df
    encodings = encodings_col.tolist()

    # converting data types
    encodings_arr = array(encodings)

    # returning encodings array
    return encodings_arr


def add_umap_cols(df: DataFrame,
                  col: str
                  ) -> None:
    """
    Given an encodings df, adds
    UMAP columns, based on given
    column.
    """
    # getting encodings array
    encodings_arr = get_encodings_arr(df=df,
                                      col=col)

    # creating umap object
    umap_2d = UMAP(n_components=2,
                   n_neighbors=15,
                   min_dist=0.1,
                   metric='braycurtis',
                   random_state=42)

    # getting 3d projection
    proj_2d = umap_2d.fit_transform(encodings_arr)

    # transposing array
    umap_results = proj_2d.transpose()

    # extracting values from umap results
    umap1, umap2 = umap_results

    # adding umap cols
    df['UMAP1'] = umap1
    df['UMAP2'] = umap2


def plot_encodings_clusters(input_path: str,
                            encodings_col: str,
                            color_col: str,
                            output_folder: str,
                            ) -> None:
    """
    Given a path to an encodings df,
    plots encodings umap clusters.
    """
    # reading df
    df = read_pickle(input_path)

    # checking encodings col (mnist/cifar/nuclei)
    if encodings_col == 'handcrafted_features':

        # adding handcrafted features col
        add_handcrafted_features_col(df=df)

    # adding umap cols
    add_umap_cols(df=df,
                  col=encodings_col)

    # getting cols list
    cols = df.columns

    # getting image name col
    image_name_col = 'crop_name' if 'crop_name' in cols else 'image_name'

    # plotting data
    fig = scatter(data_frame=df,
                  x='UMAP1',
                  y='UMAP2',
                  color=color_col,
                  hover_data=[image_name_col])

    # updating marker size
    fig.update_traces(marker_size=10)

    # saving plot
    save_name = f'{encodings_col}_umap.html'
    save_path = join(output_folder,
                     save_name)
    fig.write_html(save_path)


#####################################################################
# argument parsing related functions

def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "plot encodings clusters module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input path param
    parser.add_argument('-i', '--input-path',
                        dest='input_path',
                        type=str,
                        required=True,
                        help='defines path to encodings df [.pickle]')

    # encodings col param
    parser.add_argument('-e', '--encodings_col',
                        dest='encodings_col',
                        type=str,
                        required=False,
                        default='encodings',
                        help='defines column to use as encodings in plot')

    # color col param
    parser.add_argument('-c', '--color_col',
                        dest='color_col',
                        type=str,
                        required=False,
                        default='legend',
                        help='defines column to use as color in plot')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        type=str,
                        required=True,
                        help='defines output folder (folder that will contain plots)')

    # skip enter param
    parser.add_argument('-s', '--skip-enter',
                        dest='skip_enter',
                        action='store_true',
                        required=False,
                        help='defines whether to suppress "Enter to continue" input before execution')

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

    # getting input path
    input_path = args_dict['input_path']

    # getting encodings col
    encodings_col = args_dict['encodings_col']

    # getting color col
    color_col = args_dict['color_col']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running plot_overlap_data function
    plot_encodings_clusters(input_path=input_path,
                            encodings_col=encodings_col,
                            color_col=color_col,
                            output_folder=output_folder,
                            progress_tracker=progress_tracker)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
