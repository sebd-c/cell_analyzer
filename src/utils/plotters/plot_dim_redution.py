# imports module
from os.path import join
from argparse import ArgumentParser
from umap import UMAP
from numpy import ndarray
import plotly.express as px
from pandas import DataFrame
from pandas import read_pickle
from seaborn import scatterplot
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import enter_to_continue


##########################################################################################
# auxiliary functions
def run_pca(data: DataFrame,
            feature_cols: list,
            n_components: int,
            random_state: int
            ) -> ndarray:
    # getting data from df
    features = data[feature_cols]

    # scaling data (converting values to z-scores)
    features = StandardScaler().fit_transform(features)

    # getting PCA based on given components num
    pca = PCA(n_components=n_components,
              random_state=random_state)

    # getting principal components
    principal_components = pca.fit_transform(features)

    sil_score = silhouette_score(principal_components, data['label'])
    print(sil_score)

    # getting explained variation per component
    var_per_components = pca.explained_variance_ratio_
    x_var, y_var, z_var = var_per_components
    x_var_percent = round(x_var * 100, 3)
    y_var_percent = round(y_var * 100, 3)
    z_var_percent = round(y_var * 100, 3)

    # getting x, y values
    x = principal_components[:, 0]
    y = principal_components[:, 1]
    z = principal_components[:, 2]

    # adding cols to df
    data['x'] = x
    data['y'] = y
    data['z'] = z

    # expected classifications
    # defining update dict
    label_dict = {0: 'Normal',
                  1: 'Quiescent',
                  2: 'Fully senescent',
                  3: 'Senescent-like'}

    # updating labels col
    data.replace({'label': label_dict},
                 inplace=True)

    # creating plot pc1 x pc2
    scatterplot(data=data,
                x='x',
                y='y',
                hue='label')

    # defining plot axis labels
    x_label = f'PC1 ({x_var_percent}%)'
    y_label = f'PC2 ({y_var_percent}%)'

    # updating plot axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # showing plot
    plt.show()

    plt.savefig('pca.svg')
    plt.savefig('pca.pdf')
  #   # creating plot pc1 x pc3
  #   scatterplot(data=data,
  #               x='x',
  #               y='z',
  #               hue='label')

  #   # defining plot axis labels
  #   x_label = f'PC1 ({x_var_percent}%)'
  #   y_label = f'PC3 ({z_var_percent}%)'

  #   # updating plot axis labels
  #   plt.xlabel(x_label)
  #   plt.ylabel(y_label)

  #   # showing plot
  #   plt.show()

  # # creating plot pc2 x pc3
  #   scatterplot(data=data,
  #               x='y',
  #               y='z',
  #               hue='label')

  #   # defining plot axis labels
  #   x_label = f'PC2 ({y_var_percent}%)'
  #   y_label = f'PC3 ({z_var_percent}%)'

  #   # updating plot axis labels
  #   plt.xlabel(x_label)
  #   plt.ylabel(y_label)

  #   # showing plot
  #   plt.show()

# running PCA
run_pca(data=DATA,
        feature_cols=FEATURE_COLS,
        n_components=3,
        random_state=SEED)


def run_tsne(data: DataFrame,
             feature_cols: list,
             n_components: int,
             perplexity: int,
             max_iter: int,
             random_state: int
             ) -> ndarray:
    # getting data from df
    features = data[feature_cols]

    # scaling data (converting values to z-scores)
    features = StandardScaler().fit_transform(features)

    # getting TSNE based on given components num
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                max_iter=max_iter,
                random_state=random_state)

    # getting principal components
    principal_components = tsne.fit_transform(features)

    kl_val = tsne.kl_divergence_
    print(f"KL div: {kl_val:.2f}")

    # Step 3: Apply clustering on t-SNE-transformed data
    kmeans = KMeans(n_clusters=4, random_state=42)
    y_pred = kmeans.fit_predict(principal_components)

    # Step 4: Calculate silhouette metrics
    global_sil_score = silhouette_score(principal_components, y_pred)
    print(f"Global Silhouette Score: {global_sil_score:.2f}")

    # getting x, y values
    x = principal_components[:, 0]
    y = principal_components[:, 1]

    # adding cols to df
    data['x'] = x
    data['y'] = y

    # defining update dict
    label_dict = {0: 'Normal',
                  1: 'Quiescent',
                  2: 'Fully senescent',
                  3: 'Senescent-like'}

    # updating labels col
    data.replace({'label': label_dict},
                 inplace=True)

    # creating plot
    ax = scatterplot(data=data,
                     x='x',
                     y='y',
                     hue='label')

    # defining plot axis labels
    x_label = f'T-SNE 1'
    y_label = f'T-SNE 2'

    # updating plot axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # showing plot
    plt.show()

    plt.savefig('tsne.svg')
    plt.savefig('tsne.pdf')

# running TSNE
run_tsne(data=DATA,
         feature_cols=FEATURE_COLS,
         n_components=2,
         perplexity=40,
         max_iter=300,
         random_state=SEED)

def run_umap(data: DataFrame,
             feature_cols: list,
             random_state: int
             ) -> ndarray:
    # getting data from df
    features = data[feature_cols]

    # scaling data (converting values to z-scores)
    features = StandardScaler().fit_transform(features)

    # getting UMAP based on given components num
    umap = UMAP(random_state=random_state)

    # getting principal components
    principal_components = umap.fit_transform(features)

    # getting x, y values
    x = principal_components[:, 0]
    y = principal_components[:, 1]

    # adding cols to df
    data['x'] = x
    data['y'] = y

    # defining update dict
    label_dict = {0: 'Normal',
                  1: 'Quiescent',
                  2: 'Fully senescent',
                  3: 'Senescent-like'}

    # updating labels col
    data.replace({'label': label_dict},
                 inplace=True)

    # creating plot
    scatterplot(data=data,
                x='x',
                y='y',
                hue='label')

    # defining plot axis labels
    x_label = f'UMAP 1'
    y_label = f'UMAP 2'

    # updating plot axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # showing plot
    plt.show()

    plt.savefig('umap.svg')
    plt.savefig('umap.pdf')
# running TSNE
run_umap(data=DATA,
         feature_cols=FEATURE_COLS,
         random_state=SEED)

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
