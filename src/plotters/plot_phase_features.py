###########################################################################################
# imports
import os
from os.path import join
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed, only saving figs
import matplotlib.pyplot as plt
import seaborn as sns
from src.plotters import plot_cytnuc_features as cytnuc_plotter
from src.plotters.plot_cytnuc_features import MORPHO_FEATURES
from src.plotters.plot_cytnuc_features import BOUNDED_FEATURES
from src.plotters.plot_cytnuc_features import get_feature_category
from src.plotters.plot_cytnuc_features import get_group_order
from src.plotters.plot_cytnuc_features import get_group_palette
from src.plotters.plot_cytnuc_features import use_log_scale
from src.plotters.plot_cytnuc_features import get_features
from src.plotters.plot_cytnuc_features import plot_distributions
from src.plotters.plot_cytnuc_features import plot_scatters
from src.plotters.plot_cytnuc_features import save_fig
from src._execution_formatting import enter_to_continue
from src._execution_formatting import print_execution_parameters
from src._execution_formatting import print_progress_message


################################################################################################
# module of plots made from the phase segmentation df (generate_phase_segmentation_df output),
# reusing the cytnuc plotter functions (objects are treated as nuclei, i.e. "nuc_" features)

# compartment prefix given to the phase df columns
PREFIX = 'nuc_'

# group parts, read from the image name (e.g. C_12_7.2_N_BS-1.tif, P20_5.3_..., P 20 2.4 N BS-1.tif)
GROUP_PATTERN = r'^([CP])[ _]?(\d+)'
TREATMENT_NAMES = {'C': 'Controle', 'P': 'Fotodinâmica'}

# group names shown in plots (in this order, in two lines so axis labels don't overlap)
# and their colors (light = 12 weeks, dark = 20 weeks)
GROUP_COLORS = {'Controle\n12 semanas': '#6baed6',
                'Controle\n20 semanas': '#08519c',
                'Fotodinâmica\n12 semanas': '#fd8d3c',
                'Fotodinâmica\n20 semanas': '#a63603'}

# 2d scatter plots: (x, y, use one row per cytoplasm)
SCATTER_PAIRS = [('nuc_eccentricity', 'nuc_area', False),
                 ('nuc_roundness', 'nuc_area', False),
                 ('nuc_ii', 'nuc_area', False),
                 ('nuc_aspect', 'nuc_roundness', False)]

# the cytnuc plotter functions order and color groups from its module constants
cytnuc_plotter.GROUP_NAMES = {name: name for name in GROUP_COLORS}
cytnuc_plotter.GROUP_COLORS = GROUP_COLORS


def prepare_phase_df(phase_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefixes the phase df columns with "nuc_" (so the cytnuc plotter functions
    can use them) and adds treatment, weeks and group columns
    from the image name (e.g. C_12_... -> Controle, 12, "Controle\n12 semanas").
    """
    phase_df = phase_df.add_prefix(PREFIX)

    group_parts = phase_df[f'{PREFIX}image_name'].str.extract(GROUP_PATTERN)
    phase_df['treatment'] = group_parts[0].map(TREATMENT_NAMES)
    phase_df['weeks'] = pd.to_numeric(group_parts[1])
    phase_df['group'] = phase_df['treatment'] + '\n' + group_parts[1] + ' semanas'

    # images whose name doesn't follow the pattern are left out
    missing_group = phase_df['group'].isna()
    if missing_group.any():
        print(f'no group found for: {sorted(phase_df.loc[missing_group, f"{PREFIX}image_name"].unique())}')

    return phase_df[~missing_group]


#####################################################################
# per group normalized kde plots


def plot_normalized_kdes(phase_df: pd.DataFrame,
                         features: list,
                         group_column: str,
                         kde_normalized_folder: str,
                         file_format: str
                         ) -> None:
    """
    For every feature, saves a kde plot with each group normalized on its own
    (every curve has area 1), so small groups are as readable as large ones
    and the groups' shapes can be compared regardless of their sizes.
    """
    group_order = get_group_order(phase_df, group_column)

    for feature_index, feature in enumerate(features, 1):
        # printing progress message
        base_string = 'plotting normalized kdes #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=feature_index,
                               total=len(features))

        # creating category subfolder (morphological, <compartment>_<channel>_intensity...)
        category_folder = join(kde_normalized_folder, get_feature_category(feature))
        os.makedirs(category_folder, exist_ok=True)

        # same log scale and clipping rules as the other kde plots
        log_scale = bool(use_log_scale(phase_df[feature], feature))
        clip = (0, None) if not log_scale and phase_df[feature].min() >= 0 else None
        upper_bound = BOUNDED_FEATURES.get(feature.split('_', 1)[1])
        if clip is not None and upper_bound is not None:
            clip = (0, upper_bound)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.kdeplot(data=phase_df,
                    x=feature,
                    hue=group_column,
                    hue_order=group_order,
                    palette=get_group_palette(group_order),
                    common_norm=False,
                    log_scale=log_scale,
                    clip=clip,
                    warn_singular=False,
                    ax=ax)
        ax.set_title(feature)
        ax.set_ylabel('Density (each group normalized)')
        if log_scale:
            ax.set_xlabel(f'{feature} (log)')
        save_fig(fig, category_folder, feature, file_format)

    print()


#####################################################################
# correlation plots


def plot_correlations(phase_df: pd.DataFrame,
                      features: list,
                      correlation_folder: str,
                      file_format: str,
                      min_abs_correlation: float = 0.8
                      ) -> None:
    """
    Saves:
    - an annotated correlation matrix of the morphological features;
    - a clustered correlation matrix of all features;
    - a csv with every pair of features with |r| >= min_abs_correlation.
    Spearman correlation is used, as many features are skewed.
    """
    # morphological features (readable)
    morpho_features = [f'{PREFIX}{feature}' for feature in MORPHO_FEATURES if f'{PREFIX}{feature}' in features]
    morpho_corr = phase_df[morpho_features].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(morpho_corr,
                vmin=-1,
                vmax=1,
                cmap='vlag',
                annot=True,
                fmt='.2f',
                annot_kws={'size': 8},
                square=True,
                ax=ax)
    ax.set_title('Morphological features correlation (Spearman)')
    save_fig(fig, correlation_folder, 'morpho_correlation', file_format)

    # all features (clustering needs no missing values)
    corr = phase_df[features].corr(method='spearman')
    size = max(10, 0.12 * len(features))
    cluster_grid = sns.clustermap(corr.fillna(0),
                                  vmin=-1,
                                  vmax=1,
                                  cmap='vlag',
                                  figsize=(size, size),
                                  xticklabels=True,
                                  yticklabels=True,
                                  dendrogram_ratio=0.08)
    cluster_grid.ax_heatmap.tick_params(labelsize=6)
    cluster_grid.figure.suptitle('Features correlation (Spearman, clustered)', y=1.01)
    save_fig(cluster_grid.figure, correlation_folder, 'all_correlation_clustered', file_format)

    # saving each correlated pair once (upper triangle), strongest first
    upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(upper_mask).stack().reset_index()
    pairs.columns = ['feature_1', 'feature_2', 'spearman_r']
    pairs = pairs[pairs['spearman_r'].abs() >= min_abs_correlation]
    pairs = pairs.reindex(pairs['spearman_r'].abs().sort_values(ascending=False).index)
    pairs.to_csv(join(correlation_folder, 'correlated_pairs.csv'), index=False)


#####################################################################
# main plotting function


def make_plots(input_df_path: str,
               output_folder: str,
               file_format: str
               ) -> None:
    """
    Reads the phase df and saves all plots,
    each kind in its own (newly created) subfolder.
    """
    # reading df
    phase_df = prepare_phase_df(pd.read_pickle(input_df_path))

    # creating output folders
    folders = {name: join(output_folder, name)
               for name in ('kde', 'kde_density', 'kde_normalized', 'violin', 'correlation', 'scatter')}
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # getting features (constant ones are skipped)
    features = get_features(phase_df, PREFIX)
    print(f'{len(features)} features, objects per group:')
    print(phase_df['group'].value_counts().to_string())

    # plotting
    plot_distributions(cytnuc_df=phase_df,
                       features=features,
                       group_column='group',
                       kde_folder=folders['kde'],
                       kde_density_folder=folders['kde_density'],
                       violin_folder=folders['violin'],
                       file_format=file_format)

    plot_normalized_kdes(phase_df=phase_df,
                         features=features,
                         group_column='group',
                         kde_normalized_folder=folders['kde_normalized'],
                         file_format=file_format)

    print('plotting correlations...')
    plot_correlations(phase_df=phase_df,
                      features=features,
                      correlation_folder=folders['correlation'],
                      file_format=file_format)

    print('plotting scatters...')
    scatter_pairs = [pair for pair in SCATTER_PAIRS if pair[0] in features and pair[1] in features]
    plot_scatters(cytnuc_df=phase_df,
                  group_column='group',
                  scatter_folder=folders['scatter'],
                  file_format=file_format,
                  scatter_pairs=scatter_pairs)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('plotting complete!')


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    """
    # defining program description
    description = 'plot phase segmentation df features module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser
    parser.add_argument('-i', '--input-dataframe',
                        dest='input_dataframe',
                        required=True,
                        help='defines path to input phase df pickle (contours_df.pickle)')

    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder (subfolders are created inside it)')

    parser.add_argument('-f', '--file-format',
                        dest='file_format',
                        required=False,
                        default='png',
                        help='defines the plots file format (png, svg, pdf)')

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

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running plotting function
    make_plots(input_df_path=args_dict['input_dataframe'],
               output_folder=args_dict['output_folder'],
               file_format=args_dict['file_format'])


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
