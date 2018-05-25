import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def returns_data_frame(historical_data, return_type = "close_rate"):
    # TO-DO
    # Add asserts for case invalid historical data format

    coins = historical_data.keys()
    df = pd.DataFrame(index=historical_data[coins[0]].index, columns=[coins])

    for token in coins:
        if return_type == "close_rate":
            df[token] = historical_data[token]. close /historical_data[token].close.shift(1) - 1
        if return_type == "log":
            df[token] = np.log(historical_data[token].close).diff()
        if return_type == "open_close_rate":
            df[token] = historical_data[token]. close /historical_data[token].open

    df.fillna(0, inplace=True)
    return df[1:]

def pf_correlation(historical_data, return_type = "close_rate", size_tuple = (20,20), dir_to_save = None):
    returns = returns_data_frame(historical_data, return_type = "close_rate")
    corr = returns.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=size_tuple)

    sns_plot = sns.heatmap(corr, mask=mask, vmax=1, center=0, annot=True, fmt='.1f',
                           square=True, linewidths=.5, cbar_kws={"shrink": .5});

    fig = sns_plot.get_figure()
    if dir_to_save is not None:
        fig.savefig(dir_to_save + '/full_figure.png')

    return corr