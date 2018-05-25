import numpy as np
import metrics as mt
from market.portfolio_data import returns_data_frame
import pandas as pd

def ubah(data,save_results_dir = None):
    coins = data.keys()
    weight_arr = np.array([1./len(coins)]*len(coins))
    test_returns = returns_data_frame(data,return_type="log")
    max_dd, apv, sharpe_ratio = mt.portfolio_metrics(test_returns,weight_arr,"UBAH Daily Accumulated Returns",save_results_dir)
    mt.print_portfolio_perfomance(apv,sharpe_ratio,max_dd)


def bestStock(data, rebalance_period,save_results_dir = None):
    coins = data.keys()
    test_returns = returns_data_frame(data, return_type="log")
    best_coin_period = -np.inf
    for coin in coins:
        rol_sum = sum(test_returns[coin].iloc[:rebalance_period])
        if rol_sum > best_coin_period:
            best_coin_period = sum(test_returns[coin].iloc[:rebalance_period])
            best_stock = coin
    dd = {"max_dd": 0, "min_dd": 0}
    p_return_arr = []
    cum_p_return_arr = []
    for t in range(rebalance_period + 1, test_returns.shape[0]):
        if t % rebalance_period == 0:
            weights = (test_returns.columns == best_stock).astype(int)
            max_dd, i, j, apv, sr, cum_p_return, p_return = mt.portfolio_metrics(test_returns[t - rebalance_period:t],
                                                                              weights, plot=False, rebalanced=True)
            p_return_arr.append(p_return)
            cum_p_return_arr.append(cum_p_return)

            if p_return[i] < dd["min_dd"]:
                dd["min_dd"] = p_return[i]
                dd["min_dd_date"] = i
            if p_return[j] > dd["max_dd"]:
                dd["max_dd"] = p_return[j]
                dd["max_dd_date"] = j

            mt.print_best_stock_performance(test_returns, best_stock, rebalance_period, t, apv, sr, max_dd)

            # choose the best performed asset on period [t - rebalance_period, t]
            best_coin_period = -np.inf
            for coin in coins:
                if sum(test_returns[coin].iloc[t - rebalance_period:t]) > best_coin_period:
                    best_coin_period = sum(test_returns[coin].iloc[1:rebalance_period])
                    best_stock = coin
        if t == test_returns.shape[0] - 1:
            weights = (test_returns.columns == best_stock).astype(int)
            max_dd, i, j, apv, sr, cum_p_return, p_return = mt.portfolio_metrics(test_returns[t - rebalance_period:],
                                                                              weights, plot=False, rebalanced=True)
            p_return_arr.append(p_return)
            cum_p_return_arr.append(cum_p_return)

            if p_return[i] < dd["min_dd"]:
                dd["min_dd"] = p_return[i]
                dd["min_dd_date"] = i
            if p_return[j] > dd["max_dd"]:
                dd["max_dd"] = p_return[j]
                dd["max_dd_date"] = j
            mt.print_best_stock_performance(test_returns, best_stock, rebalance_period, t, apv, sr, max_dd)
    weighted_p_return = pd.concat(p_return_arr)
    final_cum_p_return = pd.concat(cum_p_return_arr)

    apv = final_cum_p_return[-1]
    sr = np.sqrt(365) * (np.mean(weighted_p_return) / np.std(weighted_p_return))
    max_dd_final = final_cum_p_return[dd["max_dd_date"]] - final_cum_p_return[dd["min_dd_date"]]
    print max_dd_final
    mt.print_portfolio_perfomance(apv, sr, max_dd)
    mt.plot_portfolio_backtest(final_cum_p_return, dd["min_dd_date"], dd["max_dd_date"], "BEST Daily Accumulated Returns",save_results_dir)