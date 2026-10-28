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
from matplotlib.ticker import LogFormatter
from src._execution_formatting import enter_to_continue
from src._execution_formatting import print_progress_message
from src._execution_formatting import print_execution_parameters


################################################################################################
# module of plots made from the cytoplasm x nuclei df (generate_cytnuc_df output)

# keys identifying a single cytoplasm in the cytnuc df
CYTO_KEYS = ['cyto_image_name', 'cyto_index']

# numeric columns that are ids/positions, not features
NON_FEATURE_SUFFIXES = ('_index', '_cx_coords', '_cy_coords')

# morphological features, used in the readable correlation matrix
MORPHO_FEATURES = ['area', 'area_box', 'radius_ratio', 'aspect',
                   'eccentricity', 'roundness', 'ii']

# feature name markers, used to split plots in subfolders
INTENSITY_FEATURES = ('mean', 'median', 'max', 'min', 'sum', 'int_density')
TEXTURE_MARKERS = ('glcm_', 'lbp_', 'local_')
SPOT_MARKER = 'spot_'

# group names shown in plots (in this order)
GROUP_NAMES = {'f-c-': 'Normal', 'f-c+': 'Quiescent', 'f+c+': 'Fully Senescent', 'f+c-': 'Senescent-like'}

# fixed color per group (matplotlib tab10 green, blue, orange, red)
GROUP_COLORS = {'Normal': '#2ca02c', 'Quiescent': '#1f77b4', 'Fully Senescent': '#ff7f0e', 'Senescent-like': '#d62728'}

# names used in plot labels
COMPARTMENT_NAMES = {'cyto': 'Cytoplasmic', 'nuc': 'Nuclear'}
FEATURE_NAMES = {'ii': 'irregularity index'}

# features with a known upper bound (kde is clipped at it)
BOUNDED_FEATURES = {'eccentricity': 1}

# features always plotted in log scale in the 2d scatter plots
LOG_SCALE_FEATURES = ('ii',)

# 2d scatter plots: (x, y, use one row per cytoplasm)
SCATTER_PAIRS = [('cyto_area', 'cyto_eccentricity', True),
                 ('nuc_area', 'nuc_eccentricity', False),
                 ('cyto_area', 'nuc_area', False),
                 ('cyto_eccentricity', 'nuc_eccentricity', False),
                 ('cyto_area', 'cyto_ii', True),
                 ('nuc_area', 'nuc_ii', False)]


class PlainLogFormatter(LogFormatter):
    """
    Log axis tick labels as plain numbers (e.g. 20000 instead of 2e+04).
    """
    def _num_to_string(self, x, vmin, vmax):
        return f'{x:.0f}' if x >= 1 else f'{x:g}'

    def set_locs(self, locs=None):
        super().set_locs(locs)

        # axes spanning more than 1 decade: only label 2 and 5 between decades, so labels don't collide
        vmin, vmax = sorted(self.axis.get_view_interval())
        if vmin > 0 and np.log10(vmax / vmin) > 1 and self._sublabels is not None:
            self._sublabels = {1, 2, 5}


def add_group_column(cytnuc_df: pd.DataFrame,
                     group_column: str
                     ) -> pd.DataFrame:
    """
    Makes sure the df has a group column. If it doesn't,
    creates it from the [1] element of the cytoplasm image name split by "_"
    (e.g. 01_f+c+_t50.tif -> f+c+)
    """
    if group_column not in cytnuc_df.columns:
        cytnuc_df[group_column] = cytnuc_df['cyto_image_name'].str.split('_').str[1]

        group_dict = {'f-c-': 'Normal', 'f-c+': 'Quiescent', 'f+c+': 'Fully Senescent', 'f+c-': 'Senescent-like'}

        cytnuc_df.replace({'group': group_dict}, inplace=True)

    # renaming groups also when the df already had a group column
    cytnuc_df[group_column] = cytnuc_df[group_column].replace(GROUP_NAMES)

    return cytnuc_df


def get_group_palette(group_order: list) -> dict:
    """
    Returns {group: color}, with the fixed GROUP_COLORS for known groups
    and the remaining tab10 colors for any other group.
    """
    spare_colors = [color for color in plt.get_cmap('tab10').colors
                    if matplotlib.colors.to_hex(color) not in GROUP_COLORS.values()]
    other_groups = [group for group in group_order if group not in GROUP_COLORS]

    palette = {group: GROUP_COLORS[group] for group in group_order if group in GROUP_COLORS}
    palette.update({group: spare_colors[index % len(spare_colors)]
                    for index, group in enumerate(other_groups)})

    return palette


def get_group_order(cytnuc_df: pd.DataFrame,
                    group_column: str
                    ) -> list:
    """
    Returns the groups in GROUP_NAMES order (so colors and order are the same
    in every plot), followed by any other group, alphabetically.
    """
    groups = cytnuc_df[group_column].dropna().unique()
    known_groups = [group for group in GROUP_NAMES.values() if group in groups]
    other_groups = sorted(group for group in groups if group not in known_groups)

    return known_groups + other_groups


def get_cyto_df(cytnuc_df: pd.DataFrame) -> pd.DataFrame:
    """
    The cytnuc df has one row per nucleus, so cytoplasms with
    multiple nuclei are repeated. Returns one row per cytoplasm,
    so cytoplasmic features aren't counted more than once.
    """
    return cytnuc_df.drop_duplicates(subset=CYTO_KEYS)


def get_features(df: pd.DataFrame,
                 prefix: str
                 ) -> list:
    """
    Returns the numeric feature columns of a compartment (prefix "cyto_" or "nuc_"),
    skipping ids/positions and features that are constant (nothing to plot).
    """
    numeric_columns = df.select_dtypes(include='number').columns

    features = [column
                for column in numeric_columns
                if column.startswith(prefix)
                and not column.endswith(NON_FEATURE_SUFFIXES)
                and df[column].nunique(dropna=True) > 1]

    return features


def get_plot_df(cytnuc_df: pd.DataFrame,
                feature: str
                ) -> pd.DataFrame:
    """
    Returns the df to plot a feature from:
    one row per cytoplasm for cytoplasmic features,
    one row per nucleus for nuclear features.
    """
    if feature.startswith('cyto_'):
        return get_cyto_df(cytnuc_df)

    return cytnuc_df


def get_feature_category(feature: str) -> str:
    """
    Returns the subfolder a feature's plots go to:
    "morphological", "spots", "other",
    "<compartment>_<channel>_intensity" (e.g. cyto_grayscale_intensity)
    or "<compartment>_<channel>_texture" (e.g. cyto_grayscale_texture).
    Feature names follow <compartment>_<feature> or <compartment>_<channel>_<feature>.
    """
    compartment, feature_name = feature.split('_', 1)

    if feature_name in MORPHO_FEATURES:
        return 'morphological'

    # remaining features start with the channel
    channel, channel_feature = feature_name.split('_', 1) if '_' in feature_name else (feature_name, '')

    if channel_feature in INTENSITY_FEATURES:
        return f'{compartment}_{channel}_intensity'

    if channel_feature.startswith(TEXTURE_MARKERS):
        return f'{compartment}_{channel}_texture'

    if channel_feature.startswith(SPOT_MARKER):
        return 'spots'

    return 'other'


def save_fig(fig: plt.Figure,
             output_folder: str,
             file_name: str,
             file_format: str
             ) -> None:
    """
    Saves and closes a figure (closing keeps memory low in bulk plotting).
    """
    fig.savefig(join(output_folder, f'{file_name}.{file_format}'),
                dpi=150,
                bbox_inches='tight')
    plt.close(fig)


#####################################################################
# bulk distribution plots


def plot_distributions(cytnuc_df: pd.DataFrame,
                       features: list,
                       group_column: str,
                       kde_folder: str,
                       kde_density_folder: str,
                       violin_folder: str,
                       file_format: str
                       ) -> None:
    """
    For every feature, saves a count kde plot (histogram + kde scaled to counts),
    a density kde plot (scaled by group size) and a violin plot, split by group.
    """
    group_order = get_group_order(cytnuc_df, group_column)

    for feature_index, feature in enumerate(features, 1):
        # printing progress message
        base_string = 'plotting distributions #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=feature_index,
                               total=len(features))

        plot_df = get_plot_df(cytnuc_df, feature)

        # creating category subfolders (morphological, intensity, <compartment>_<channel>_texture...)
        category = get_feature_category(feature)
        kde_category_folder = join(kde_folder, category)
        kde_density_category_folder = join(kde_density_folder, category)
        violin_category_folder = join(violin_folder, category)
        os.makedirs(kde_category_folder, exist_ok=True)
        os.makedirs(kde_density_category_folder, exist_ok=True)
        os.makedirs(violin_category_folder, exist_ok=True)

        # skewed positive features (e.g. areas) are plotted in log scale,
        # so the kde can't spill below 0 and tails aren't cut abruptly
        log_scale = bool(use_log_scale(plot_df[feature], feature))

        # non-negative features in linear scale have the kde clipped at 0
        # (and bounded features, e.g. eccentricity, at their max too)
        clip = (0, None) if not log_scale and plot_df[feature].min() >= 0 else None
        upper_bound = BOUNDED_FEATURES.get(feature.split('_', 1)[1])
        if clip is not None and upper_bound is not None:
            clip = (0, upper_bound)

        # count kde plot (histogram outline + kde line, both in absolute counts)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(data=plot_df,
                     x=feature,
                     hue=group_column,
                     hue_order=group_order,
                     palette=get_group_palette(group_order),
                     stat='count',
                     element='step',
                     fill=False,
                     kde=True,
                     kde_kws={'clip': clip},
                     common_norm=False,
                     log_scale=log_scale,
                     ax=ax)
        ax.set_title(feature)
        if log_scale:
            ax.set_xlabel(f'{feature} (log)')
        save_fig(fig, kde_category_folder, feature, file_format)

        # density kde plot (normalized over all groups, so each curve's area is its group's share)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.kdeplot(data=plot_df,
                    x=feature,
                    hue=group_column,
                    hue_order=group_order,
                    palette=get_group_palette(group_order),
                    common_norm=True,
                    log_scale=log_scale,
                    clip=clip,
                    warn_singular=False,
                    ax=ax)
        ax.set_title(feature)
        if log_scale:
            ax.set_xlabel(f'{feature} (log)')
        save_fig(fig, kde_density_category_folder, feature, file_format)

        # violin plot, with each object as a dot
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.violinplot(data=plot_df,
                       x=group_column,
                       y=feature,
                       order=group_order,
                       hue=group_column,
                       hue_order=group_order,
                       palette=get_group_palette(group_order),
                       inner='quart',
                       cut=0,
                       log_scale=log_scale,
                       ax=ax)
        sns.stripplot(data=plot_df,
                      x=group_column,
                      y=feature,
                      order=group_order,
                      color='black',
                      size=2.5,
                      alpha=0.5,
                      jitter=0.15,
                      ax=ax)
        ax.set_title(feature)
        if log_scale:
            ax.set_ylabel(f'{feature} (log)')
            ax.yaxis.set_minor_formatter(PlainLogFormatter(labelOnlyBase=False, minor_thresholds=(2, 1)))
        save_fig(fig, violin_category_folder, feature, file_format)

    print()


#####################################################################
# correlation plots


def plot_correlations(cytnuc_df: pd.DataFrame,
                      cyto_features: list,
                      nuc_features: list,
                      correlation_folder: str,
                      file_format: str,
                      min_abs_correlation: float = 0.8
                      ) -> None:
    """
    Saves:
    - an annotated correlation matrix of the morphological features
      of cytoplasm and nucleus (readable, includes cyto x nuc relations);
    - a clustered correlation matrix of all features of each compartment
      (too many to read one by one, but shows blocks of redundant features);
    - a csv with every pair of features of each compartment with |r| >= min_abs_correlation,
      to help picking which redundant features to drop.
    Spearman correlation is used, as many features are skewed.
    """
    # morphological features of both compartments (one row per nucleus)
    morpho_features = [f'{prefix}_{feature}'
                       for prefix in ('cyto', 'nuc')
                       for feature in MORPHO_FEATURES
                       if f'{prefix}_{feature}' in cyto_features + nuc_features]
    morpho_corr = cytnuc_df[morpho_features].corr(method='spearman')

    # cyto x cyto block from one row per cytoplasm (multinucleated ones counted once),
    # cyto x nuc and nuc x nuc blocks stay one row per nucleus
    cyto_morpho_features = [feature for feature in morpho_features if feature.startswith('cyto_')]
    morpho_corr.loc[cyto_morpho_features, cyto_morpho_features] = (get_cyto_df(cytnuc_df)[cyto_morpho_features]
                                                                   .corr(method='spearman'))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(morpho_corr,
                vmin=-1,
                vmax=1,
                cmap='vlag',
                annot=True,
                fmt='.2f',
                annot_kws={'size': 7},
                square=True,
                ax=ax)
    ax.set_title('Morphological features correlation (Spearman)')
    save_fig(fig, correlation_folder, 'morpho_correlation', file_format)

    # all features, per compartment
    correlated_pairs_list = []
    for prefix, features in (('cyto', cyto_features), ('nuc', nuc_features)):
        plot_df = get_cyto_df(cytnuc_df) if prefix == 'cyto' else cytnuc_df
        corr = plot_df[features].corr(method='spearman')

        # clustering needs no missing values (constant within the subset -> nan)
        clustered_corr = corr.fillna(0)

        # sizing so labels stay legible when zooming in
        size = max(10, 0.12 * len(features))
        cluster_grid = sns.clustermap(clustered_corr,
                                      vmin=-1,
                                      vmax=1,
                                      cmap='vlag',
                                      figsize=(size, size),
                                      xticklabels=True,
                                      yticklabels=True,
                                      dendrogram_ratio=0.08)
        cluster_grid.ax_heatmap.tick_params(labelsize=4)
        cluster_grid.figure.suptitle(f'{COMPARTMENT_NAMES[prefix]} features correlation '
                                     f'(Spearman, clustered)', y=1.01)
        save_fig(cluster_grid.figure, correlation_folder, f'{prefix}_all_correlation_clustered', file_format)

        # getting each pair only once (upper triangle)
        upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.where(upper_mask).stack().reset_index()
        pairs.columns = ['feature_1', 'feature_2', 'spearman_r']
        pairs = pairs[pairs['spearman_r'].abs() >= min_abs_correlation]
        correlated_pairs_list.append(pairs)

    # saving correlated pairs, strongest first
    correlated_pairs = pd.concat(correlated_pairs_list, ignore_index=True)
    correlated_pairs = correlated_pairs.reindex(correlated_pairs['spearman_r'].abs()
                                                .sort_values(ascending=False).index)
    correlated_pairs.to_csv(join(correlation_folder, 'correlated_pairs.csv'), index=False)


#####################################################################
# 2d scatter plots


def plot_scatters(cytnuc_df: pd.DataFrame,
                  group_column: str,
                  scatter_folder: str,
                  file_format: str,
                  scatter_pairs: list = None
                  ) -> None:
    """
    Saves 2d scatter plots, colored by group, of the pairs in SCATTER_PAIRS
    (or of the given scatter_pairs, as (x, y, use one row per cytoplasm) tuples).
    """
    group_order = get_group_order(cytnuc_df, group_column)

    if scatter_pairs is None:
        scatter_pairs = SCATTER_PAIRS

    for x_feature, y_feature, cyto_only in scatter_pairs:
        plot_df = get_cyto_df(cytnuc_df) if cyto_only else cytnuc_df

        # log scale on axes where outliers squeeze the rest of the data
        x_log = use_log_scale(plot_df[x_feature], x_feature)
        y_log = use_log_scale(plot_df[y_feature], y_feature)
        file_name = f'{x_feature}_x_{y_feature}'

        draw_scatter(plot_df, x_feature, y_feature, group_column, group_order,
                     x_log, y_log, scatter_folder, file_name, file_format)

        # also saving a fully linear version, as log scale visually shrinks differences
        if x_log or y_log:
            draw_scatter(plot_df, x_feature, y_feature, group_column, group_order,
                         False, False, scatter_folder, f'{file_name}_linear', file_format)


def draw_scatter(plot_df: pd.DataFrame,
                 x_feature: str,
                 y_feature: str,
                 group_column: str,
                 group_order: list,
                 x_log: bool,
                 y_log: bool,
                 scatter_folder: str,
                 file_name: str,
                 file_format: str
                 ) -> None:
    """
    Draws and saves a single 2d scatter plot, colored by group,
    with log scale on the requested axes.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=plot_df,
                    x=x_feature,
                    y=y_feature,
                    hue=group_column,
                    hue_order=group_order,
                    palette=get_group_palette(group_order),
                    alpha=0.7,
                    ax=ax)
    ax.set_xlabel(get_axis_label(x_feature))
    ax.set_ylabel(get_axis_label(y_feature))

    if x_log:
        ax.set_xscale('log')
        ax.set_xlabel(f'{ax.get_xlabel()} (log)')
        # labeling minor ticks too when the axis spans only 1-2 decades
        ax.xaxis.set_minor_formatter(PlainLogFormatter(labelOnlyBase=False, minor_thresholds=(2, 1)))
    if y_log:
        ax.set_yscale('log')
        ax.set_ylabel(f'{ax.get_ylabel()} (log)')
        ax.yaxis.set_minor_formatter(PlainLogFormatter(labelOnlyBase=False, minor_thresholds=(2, 1)))

    save_fig(fig, scatter_folder, file_name, file_format)


def use_log_scale(values: pd.Series,
                  feature: str,
                  min_bulk_fraction: float = 0.4
                  ) -> bool:
    """
    Returns True if an axis should be in log scale:
    always for features in LOG_SCALE_FEATURES, otherwise when the axis is "too tight",
    i.e. the central 90% of the values (5th-95th percentile) spans less than
    min_bulk_fraction of the full range, so a few outliers squeeze the rest.
    Never for values <= 0, as they can't be shown in log scale.
    """
    values = values.dropna()

    if values.empty or values.min() <= 0:
        return False

    if feature.split('_', 1)[1] in LOG_SCALE_FEATURES:
        return True

    full_range = values.max() - values.min()
    bulk_range = values.quantile(0.95) - values.quantile(0.05)

    return full_range > 0 and bulk_range / full_range < min_bulk_fraction


def get_axis_label(feature: str) -> str:
    """
    Turns e.g. "cyto_area" into "Cytoplasmic area"
    """
    prefix, feature_name = feature.split('_', 1)
    feature_name = FEATURE_NAMES.get(feature_name, feature_name.replace('_', ' '))

    return f'{COMPARTMENT_NAMES[prefix]} {feature_name}'


#####################################################################
# multinucleated cytoplasms plot


def plot_multinucleated(cytnuc_df: pd.DataFrame,
                        group_column: str,
                        multinucleated_folder: str,
                        file_format: str
                        ) -> None:
    """
    For each group, saves a figure with area x eccentricity of the cytoplasms
    that have more than one nucleus (left) and of their nuclei (right).
    A cytoplasm and its nuclei share the same color.
    """
    # counting nuclei per cytoplasm
    nuclei_count = cytnuc_df.groupby(CYTO_KEYS)['nuc_index'].transform('size')
    multinucleated_df = cytnuc_df[nuclei_count > 1]

    if multinucleated_df.empty:
        print('no multinucleated cytoplasms found, skipping multinucleated plots')
        return

    # same axes limits in every group, so figures can be compared
    cyto_limits = get_limits(multinucleated_df, 'cyto_area', 'cyto_eccentricity')
    nuc_limits = get_limits(multinucleated_df, 'nuc_area', 'nuc_eccentricity')

    for group, group_df in multinucleated_df.groupby(group_column):
        # one color per cytoplasm
        cyto_ids = group_df[CYTO_KEYS].drop_duplicates().itertuples(index=False, name=None)
        cyto_ids = list(cyto_ids)
        colors = get_colors(len(cyto_ids))

        fig, (cyto_ax, nuc_ax) = plt.subplots(1, 2, figsize=(14, 6))

        for cyto_id, color in zip(cyto_ids, colors):
            image_name, cyto_index = cyto_id
            cyto_rows = group_df[(group_df['cyto_image_name'] == image_name) &
                                 (group_df['cyto_index'] == cyto_index)]
            label = f'{image_name} | cyto {int(cyto_index)} ({len(cyto_rows)} nuclei)'

            # the cytoplasm (once)
            cyto_ax.scatter(cyto_rows['cyto_area'].iloc[0],
                            cyto_rows['cyto_eccentricity'].iloc[0],
                            color=color,
                            s=60,
                            edgecolor='black',
                            linewidth=0.5,
                            label=label)

            # its nuclei
            nuc_ax.scatter(cyto_rows['nuc_area'],
                           cyto_rows['nuc_eccentricity'],
                           color=color,
                           s=60,
                           edgecolor='black',
                           linewidth=0.5)

        cyto_ax.set_title('Cytoplasm')
        cyto_ax.set_xlabel(get_axis_label('cyto_area'))
        cyto_ax.set_ylabel(get_axis_label('cyto_eccentricity'))
        cyto_ax.set_xlim(cyto_limits[0])
        cyto_ax.set_ylim(cyto_limits[1])

        nuc_ax.set_title('Nuclei')
        nuc_ax.set_xlabel(get_axis_label('nuc_area'))
        nuc_ax.set_ylabel(get_axis_label('nuc_eccentricity'))
        nuc_ax.set_xlim(nuc_limits[0])
        nuc_ax.set_ylim(nuc_limits[1])

        fig.suptitle(f'Multinucleated cytoplasms - {group_column}: {group} '
                     f'({len(cyto_ids)} cytoplasms)')

        # legend below both plots, identifying each cytoplasm
        fig.legend(loc='lower center',
                   bbox_to_anchor=(0.5, -0.02),
                   ncol=min(3, len(cyto_ids)),
                   fontsize=7,
                   frameon=False)
        legend_rows = int(np.ceil(len(cyto_ids) / 3))
        fig.subplots_adjust(bottom=0.12 + 0.03 * legend_rows)

        save_fig(fig, multinucleated_folder, f'multinucleated_{group.replace(" ", "_")}', file_format)


def get_limits(df: pd.DataFrame,
               x_feature: str,
               y_feature: str,
               margin: float = 0.05
               ) -> tuple:
    """
    Returns ((x_min, x_max), (y_min, y_max)) with a small margin.
    """
    limits = []
    for feature in (x_feature, y_feature):
        feature_min = df[feature].min()
        feature_max = df[feature].max()
        pad = (feature_max - feature_min) * margin or abs(feature_max) * margin or 1
        limits.append((feature_min - pad, feature_max + pad))

    return tuple(limits)


def get_colors(colors_num: int) -> list:
    """
    Returns a list of distinct colors (repeats only if there are more than 20).
    """
    if colors_num <= 10:
        return list(plt.get_cmap('tab10').colors[:colors_num])

    tab20_colors = list(plt.get_cmap('tab20').colors)

    return [tab20_colors[index % 20] for index in range(colors_num)]


#####################################################################
# main plotting function


def make_plots(input_df_path: str,
               output_folder: str,
               group_column: str,
               file_format: str
               ) -> None:
    """
    Reads the cytnuc df and saves all plots,
    each kind in its own (newly created) subfolder.
    """
    # reading df
    cytnuc_df = pd.read_pickle(input_df_path)
    cytnuc_df = add_group_column(cytnuc_df, group_column)

    # creating output folders
    folders = {name: join(output_folder, name)
               for name in ('kde', 'kde_density', 'violin', 'correlation', 'scatter', 'multinucleated')}
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # getting features (constant ones are skipped)
    cyto_features = get_features(get_cyto_df(cytnuc_df), 'cyto_')
    nuc_features = get_features(cytnuc_df, 'nuc_')
    print(f'{len(cyto_features)} cytoplasmic and {len(nuc_features)} nuclear features, '
          f'groups: {sorted(cytnuc_df[group_column].dropna().unique())}')

    # plotting
    plot_distributions(cytnuc_df=cytnuc_df,
                       features=cyto_features + nuc_features,
                       group_column=group_column,
                       kde_folder=folders['kde'],
                       kde_density_folder=folders['kde_density'],
                       violin_folder=folders['violin'],
                       file_format=file_format)

    print('plotting correlations...')
    plot_correlations(cytnuc_df=cytnuc_df,
                      cyto_features=cyto_features,
                      nuc_features=nuc_features,
                      correlation_folder=folders['correlation'],
                      file_format=file_format)

    print('plotting scatters...')
    plot_scatters(cytnuc_df=cytnuc_df,
                  group_column=group_column,
                  scatter_folder=folders['scatter'],
                  file_format=file_format)

    print('plotting multinucleated cytoplasms...')
    plot_multinucleated(cytnuc_df=cytnuc_df,
                        group_column=group_column,
                        multinucleated_folder=folders['multinucleated'],
                        file_format=file_format)

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
    description = 'plots features from the cytoplasm x nuclei df (generate_cytnuc_df output)'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input df path
    parser.add_argument('-i', '--input-dataframe',
                        dest='input_dataframe',
                        required=True,
                        help='defines path to input cytnuc pickle (contours_cytnuc.pickle)')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder (subfolders are created inside it)')

    # group column param
    parser.add_argument('-g', '--group-column',
                        dest='group_column',
                        required=False,
                        default='group',
                        help='column used to split plots by group. if missing from the df, '
                             'it is created from the cytoplasm image name (e.g. 01_f+c+_t50.tif -> f+c+)')

    # file format param
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

    # running plots
    make_plots(input_df_path=args_dict['input_dataframe'],
               output_folder=args_dict['output_folder'],
               group_column=args_dict['group_column'],
               file_format=args_dict['file_format'])


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
