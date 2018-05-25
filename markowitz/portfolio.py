import pandas as pd
import numpy as np
from backtesting import metrics as mt
import matplotlib.pyplot as plt
import markowitz_metrics as mm

def random_portfolio(returns, n_portfolios):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    m = np.asmatrix(returns.mean())
    C = np.asmatrix(returns.cov())
    n_coins = returns.shape[1]

    results_array = np.zeros((3 + n_coins, n_portfolios))

    for p in range(n_portfolios):
        weights = rand_weights(n_coins)
        w = np.asmatrix(weights)

        p_mu, p_std = mt.portfolio_annualised_performance(w, m, C)

        results_array[0, p] = p_mu
        results_array[1, p] = p_std

        # store Sharpe Ratio (return / volatility) - risk free rate element
        # excluded for simplicity
        results_array[2, p] = results_array[0, p] / results_array[1, p]

        i = 0
        for iw in weights:
            results_array[3 + i, p] = weights[i]
            i += 1

    results_frame = pd.DataFrame(np.transpose(results_array),
                                 columns=['r', 'stdev', 'sharpe_ratio']
                                         + returns.columns.tolist())

    mt.plot_markowitz_portfolio(results_frame)
    return results_frame

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)


def ef_with_random(returns, num_portfolios, print_results=True,save_results_dir = None):
    df_random_portfolio = random_portfolio(returns, num_portfolios)
    m = returns.mean()
    C = returns.cov()
    n_coins = returns.shape[1]

    max_sharpe = mm.max_sharpe_ratio(m, C, n_coins)
    p_mu_max_sharpe, p_std_max_sharpe = mt.portfolio_annualised_performance(max_sharpe['x'], m, C)
    max_sharpe_allocation = pd.DataFrame(max_sharpe["x"], index=returns.columns, columns=['allocation'])
    # max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = mm.min_variance(m, C, n_coins)
    p_mu_min_vol, p_std_min_vol = mt.portfolio_annualised_performance(min_vol['x'], m, C)
    min_vol_allocation = pd.DataFrame(min_vol["x"], index=returns.columns, columns=['allocation'])
    # min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    if print_results:
        print "-" * 80
        print "Maximum Sharpe Ratio Portfolio Allocation\n"
        print "Annualised Return:", round(p_mu_max_sharpe, 2)
        print "Annualised Volatility:", round(p_std_max_sharpe, 2)
        print "\n"
        print max_sharpe_allocation.apply(lambda i: round(i * 100, 2))

        print "-" * 80
        print "Minimum Volatility Portfolio Allocation\n"
        print "Annualised Return:", round(p_mu_min_vol, 2)
        print "Annualised Volatility:", round(p_std_min_vol, 2)
        print "\n"
        print min_vol_allocation.apply(lambda i: round(i * 100, 2))

        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(15, 10))
        plt.scatter(df_random_portfolio.stdev,
                    df_random_portfolio.r,
                    c=df_random_portfolio.sharpe_ratio,
                    cmap='RdYlGn', marker='o', s=10, alpha=0.3)
        plt.colorbar()
        plt.scatter(p_std_max_sharpe, p_mu_max_sharpe, marker='*', color='b', s=500, label='Maximum Sharpe ratio')
        plt.scatter(p_std_min_vol, p_mu_min_vol, marker='*', color='r', s=500, label='Minimum Volatitlity')

        plt.xlabel('Volatility')
        plt.ylabel('Returns')
        plt.legend(labelspacing=0.8)

        target = np.linspace(p_mu_min_vol, 3, 50)
        efficient_portfolios = mm.efficient_frontier(m, C, target, n_coins)
        plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black',
                 label='efficient frontier')
        plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
        plt.xlabel('Annualised volatility')
        plt.ylabel('Annualised returns')
        plt.legend(labelspacing=0.8)

        if save_results_dir is not None:
            name = "Markowitz"
            plt.savefig(save_results_dir + '/{0}.png'.format(name))

        scaling_parametr = 0.2
        y_min = (1 - scaling_parametr) * df_random_portfolio.r.min()
        y_max = (1 + scaling_parametr) * df_random_portfolio.r.max()

    return max_sharpe_allocation.values[0], min_vol_allocation.values[0]