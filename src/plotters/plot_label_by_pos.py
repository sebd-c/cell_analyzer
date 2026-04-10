###########################################################################################
# imports
from os.path import join

import matplotlib
import pandas
import numpy as np

from collections import defaultdict
from random import uniform
from pandas import DataFrame
from pandas import read_pickle
from matplotlib import pyplot as plt
from seaborn import histplot
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_files_in_folder
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters


################################################################################################
# module of aux functions related to img preprocessing
def get_closest_points_between_contours(contour_a, contour_b):
    """
    Find the closest pair of points between two different OpenCV contours.

    Accepted contour formats:
    - (N, 1, 2) -> standard OpenCV contour
    - (N, 2)

    Returns:
        point_a (tuple[float, float]),
        point_b (tuple[float, float]),
        min_distance (float)
    """
    a = np.asarray(contour_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(contour_b, dtype=np.float64).reshape(-1, 2)

    if a.size == 0 or b.size == 0:
        raise ValueError("Both contours must have at least one point.")

    # Pairwise squared distances between all points from contour A and contour B.
    diff = a[:, None, :] - b[None, :, :]
    dist_sq = np.einsum("ijk,ijk->ij", diff, diff)

    idx_a, idx_b = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    point_a = (float(a[idx_a, 0]), float(a[idx_a, 1]))
    point_b = (float(b[idx_b, 0]), float(b[idx_b, 1]))
    min_distance = float(np.sqrt(dist_sq[idx_a, idx_b]))

    return point_a, point_b, min_distance


def get_pair_dists(input_path: str) -> dict:
    """
    given a dataframe containing all
    """
    # read df
    df = read_pickle(input_path)

    # dict to hold distances by normalized class pair, e.g. (0, 2) == (2, 0)
    pair_dists = defaultdict(list)

    # iterate for the 1st object
    for i in range(len(df)):

        row_i = df.iloc[i]

        # get og obj label
        label_i = int(row_i['label'])

        # get all "unseen" rows
        rows_below = df.iloc[i + 1:]

        # separate by image
        rows_below = rows_below[rows_below['image_name'] == row_i['image_name']]

        print(f'contour {i} of {len(df)}')
        # and iterate for the second object
        for j, row_j in rows_below.iterrows():
            # get contours
            contour_i = row_i['cyto_contour']
            contour_j = row_j['cyto_contour']

            # get the other object's label
            label_j = int(row_j['label'])

            # make and sort the combination
            label_comb = tuple(sorted((label_i, label_j)))
            
            # calculate dist
            _, _, min_distance = get_closest_points_between_contours(contour_a=contour_i,
                                                                     contour_b=contour_j)
            pair_dists[label_comb].append(min_distance)

    print(pair_dists.keys())
    return pair_dists


def plot_dists(pair_dists: dict) -> None:
    """

    """
    def _norm_label(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return str(x)

    if not pair_dists:
        raise ValueError("pair_dists is empty.")

    items = [(pair, dists) for pair, dists in pair_dists.items() if dists]
    if not items:
        raise ValueError("pair_dists contains no distances to plot.")

    n = len(items)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (pair, dists) in zip(axes, items):
        pair_label = f"{_norm_label(pair[0])}_{_norm_label(pair[1])}"
        ax.hist(dists, bins=10, alpha=0.75)
        ax.set_title(pair_label)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Frequency")

    # hide any unused subplots
    for ax in axes[len(items):]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'returns a histplot of the distribution of classes by distance'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input_df_path',
                        dest='input_df_path',
                        required=True,
                        help='dataframe that contains all cell infos')

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
    input_df_path = args_dict['input_df_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running
    pair_dists_dict = get_pair_dists(input_path=input_df_path)

    plot_dists(pair_dists_dict)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
