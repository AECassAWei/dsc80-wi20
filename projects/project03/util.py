import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium

# ---------------------------------------------------------------------
# Permutation Test Helper function
# ---------------------------------------------------------------------


def permutation_test(data, col, group_col, test_statistic, N=1000):
    """
    Return the distribution of permuted statistics and the observed statistic
    resulting from a permutation test.

    :param: data: DataFrame of data observations and the labels for two groups.
    :param: col: Column name for the column containing the data.
    :param: group_col: Column name for the column contain the labels for the two groups.
    :param: test_statistic: The test statistic to apply to the groups (a function).
    :param: N: The number of times N to run the permutation test.
    """

    # get the observed test statistic
    obs = test_statistic(data, col, group_col)

    # run the permutations
    shuffled_stats = []
    for _ in range(N):
        
        shuffled = data[group_col].sample(frac=1, replace=False).reset_index(drop=True)
        with_shuffled = data[[col]].assign(shuffled=shuffled)
        shuffled_stat = test_statistic(with_shuffled, col, 'shuffled')
        shuffled_stats.append(shuffled_stat)

    shuffled_stats = np.array(shuffled_stats)

    return shuffled_stats, obs


def plot_distribution(stats, obs, num, title, col, p_val):
    """
    Plot the distribution of stats and observed value
    
    :param stats: statistics for distribution
    :param obs: observed value
    """
    plt.figure(num)
    pd.Series(stats).plot(kind='hist', density=True, alpha=0.8)
    plt.scatter(obs, 0, color='red', s=40)
    plt.title(title + col + ', p_val is ' + str(p_val))
    return


def choropleth(df, state, col, legend, color='BuPu'):
    """Plot the choropleth of col based on states"""
    url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
    state_geo = f'{url}/us-states.json'

    df = df.reset_index(drop=False)
    # bins = list(df[col].quantile([0, 0.25, 0.5, 0.75, 1]))

    m = folium.Map(location=[48, -102], zoom_start=3)

    folium.Choropleth(
        geo_data=state_geo,
        name='choropleth',
        data=df,
        columns=[state, col],
        key_on='feature.id',
        fill_color=color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend,
        # bins=bins,
        reset=True
    ).add_to(m)

    # folium.LayerControl().add_to(m)

    return m


def diff_in_means(data, col, group_col):
    """difference in means"""
    return data.groupby(group_col)[col].mean().diff().iloc[-1]


def tvd(data, col, group_col):
    """tvd of the distribution of values in col
    bewteen the two groups of group_col. col is
    assumed to be categorical."""

    tvd = (
        data
        .pivot_table(
            index=col, 
            columns=group_col, 
            aggfunc='size', 
            fill_value=0
        )
        .apply(lambda x: x / x.sum())
        .diff(axis=1).iloc[:, -1].abs().sum() / 2
        )

    return tvd


def ks(data, col, group_col):
    """tvd of the distribution of values in col
    bewteen the two groups of group_col. col is
    assumed to be categorical."""

    from scipy.stats import ks_2samp
    
    # should have only two values in column
    valA, valB = data[group_col].unique()
    ks, _ = ks_2samp(
        data.loc[data[group_col] == valA, col],
        data.loc[data[group_col] == valB, col]
    )

    return ks