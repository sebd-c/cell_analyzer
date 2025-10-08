# imports module
from os.path import join
from argparse import ArgumentParser
from pandas import read_pickle
from pandas import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn as sns
from seaborn import scatterplot
from seaborn import color_palette
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree
from six import StringIO
import pydotplus
from sklearn.model_selection import train_test_split
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import enter_to_continue


##########################################################################################
# auxiliary functions


def run_kmeans(input_path: str,
               output_folder: str
               ) -> None:
    # read data
    data = read_pickle(input_path)

    # set columns of interest
    feature_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                    'cyto_ecc', 'cyto_rou', 'cii', 'nuc_area', 'nuc_arbox',
                    'nuc_radra', 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii',
                    ]
    # cyto cols
    feature_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                    'cyto_ecc', 'cyto_rou', 'cii'
                    ]
    # nuc cols
    # feature_cols = ['nuc_area', 'nuc_arbox', 'nuc_radra',
    #                 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii'
    #                ]
    # feature_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp', 'cyto_ecc',
    #                 'cyto_rou', 'cii', 'cyto_contour', 'cyto_grayscale_mean', 'cyto_grayscale_median',
    #                 'cyto_grayscale_max', 'cyto_grayscale_min', 'cyto_grayscale_sum',
    #                 'cyto_grayscale_int_density', 'cyto_red_mean', 'cyto_red_median',
    #                 'cyto_red_max', 'cyto_red_min', 'cyto_red_sum', 'cyto_red_int_density',
    #                 'cyto_green_mean', 'cyto_green_median', 'cyto_green_max',
    #                 'cyto_green_min', 'cyto_green_sum', 'cyto_green_int_density',
    #                 'cyto_blue_mean', 'cyto_blue_median', 'cyto_blue_max', 'cyto_blue_min',
    #                 'cyto_blue_sum', 'cyto_blue_int_density', 'nuc_area', 'nuc_arbox',
    #                 'nuc_radra', 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii', 'nuc_contour',
    #                 'nuc_grayscale_mean', 'nuc_grayscale_median', 'nuc_grayscale_max',
    #                 'nuc_grayscale_min', 'nuc_grayscale_sum', 'nuc_grayscale_int_density',
    #                 'nuc_red_mean', 'nuc_red_median', 'nuc_red_max', 'nuc_red_min',
    #                 'nuc_red_sum', 'nuc_red_int_density', 'nuc_green_mean', 'nuc_green_median',
    #                 'nuc_green_max', 'nuc_green_min', 'nuc_green_sum', 'nuc_green_int_density',
    #                 'nuc_blue_mean', 'nuc_blue_median', 'nuc_blue_max', 'nuc_blue_min', 'nuc_blue_sum',
    #                 'nuc_blue_int_density']

    label_dict = {0: 'Normal', 1: 'Quiescent', 2: 'Fully Senescent', 3: 'Senescent-like'}

    # data.replace({'label': label_dict}, inplace=True)

    # drop the ground truth column
    X = data.drop(['label'], axis=1)
    X = X[feature_cols]

    # use scaler to "unbias" weight of features
    ms = MinMaxScaler()

    # normalize data
    X = ms.fit_transform(X)

    # define y data
    y = data['label']

    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                     test_size=0.3,
    #                                                     random_state=42,
    #                                                     stratify=y)

    # cs = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    #     kmeans.fit(X)
    #     cs.append(kmeans.inertia_)
    # plt.plot(range(1, 11), cs)
    # plt.title('The Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('CS')
    # plt.show()

    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)

    kmeans.fit(X)

    y_kmeans = kmeans.predict(X)

    plt.scatter(X[:, 1], X[:, 0], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
    # class names
    class_names = ['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']


def run_hdbscan(input_path: str,
                output_folder: str
                )-> None:
    """
    run hdbscan algorithm
    """

    # read data
    data = read_pickle(input_path)

    # set columns of interest
    feature_cols = ['cyto_area', 'cyto_arbox', 'cyto_radra', 'cyto_asp',
                    'cyto_ecc', 'cyto_rou', 'cii', 'nuc_area', 'nuc_arbox',
                    'nuc_radra', 'nuc_asp', 'nuc_ecc', 'nuc_rou', 'nii',
                    ]

    label_dict = {0: 'Normal', 1: 'Quiescent', 2: 'Fully Senescent', 3: 'Senescent-like'}

    data.replace({'label': label_dict}, inplace=True)

    # drop the ground truth column
    X = data.drop(['label'], axis=1)
    X = X[feature_cols]

    # scaling data (converting values to z-scores)
    X = StandardScaler().fit_transform(X)

    clusterer = HDBSCAN(min_cluster_size=20)
    cluster_labels = clusterer.fit_predict(X)

    # creating umap object
    umap_2d = UMAP(n_components=2,
                   n_neighbors=15,
                   min_dist=0.1,
                   metric='canberra',
                   random_state=42)

    # getting 3d projection
    proj_2d = umap_2d.fit_transform(X)

    # transposing array
    umap_results = proj_2d.transpose()

    # extracting values from umap results
    umap1, umap2 = umap_results

    # adding umap cols
    data['UMAP1'] = umap1
    data['UMAP2'] = umap2

    # ax = scatterplot(data=data,
    #                  x=data['UMAP1'],
    #                  y=data['UMAP2'],
    #                  hue='label',
    #                  palette=color_palette(['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']),
    #                  hue_order=['Normal', 'Quiescent', 'Fully Senescent', 'Senescent-like']
    #                  )
    # plt.xlabel('UMAP1')
    # plt.ylabel('UMAP2')
    # plt.grid(False)
    #
    # plt.show()

    clustered = (cluster_labels >= 0)
    # plt.scatter(data['UMAP1'],
    #             data['UMAP2'],
    #             c=data['label'],
    #             s=0.1,
    #             alpha=0.5)
    plt.scatter(data['UMAP1'],
                data['UMAP2'],
                c=cluster_labels,
                s=0.1,
                cmap='viridis')
    plt.show()

    return

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
    # run_kmeans(input_path=input_path,
    #            output_folder=output_folder
    #            )

    run_hdbscan(input_path=input_path,
                output_folder=output_folder
               )


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
