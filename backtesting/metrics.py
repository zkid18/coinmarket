import numpy as np
import matplotlib.pyplot as plt

def maximum_draw_down(ret):
    i = np.argmax(np.maximum.accumulate(ret) - ret) # end of the period
    j = np.argmax(ret[:i])
    max_dd = ret[j] - ret[i]
    return (max_dd, i,j)

def portfolio_metrics(test_returns, weights, name="Strategy_perfomance",plot=True, rebalanced = False,save_results_dir = None):
    m = np.asmatrix(test_returns.mean())
    C = np.asmatrix(test_returns.cov())
    w = np.asmatrix(weights)
    p_return = (test_returns*weights).sum(axis = 1)
    p_mu, p_std = portfolio_annualised_performance(w, m, C)
    sharpe_ratio = p_mu/p_std
    apv = np.sum(p_return)
    cum_p_return = np.cumsum(p_return)
    max_dd,i,j = maximum_draw_down(cum_p_return)
    if plot:
        plot_portfolio_backtest(cum_p_return,i,j,name,save_results_dir)
    if rebalanced:
        return max_dd, i, j, apv, sharpe_ratio, cum_p_return, p_return
    return max_dd, apv, sharpe_ratio

def print_portfolio_perfomance(apv,sharpe_ratio,max_dd):
    print "-"*80
    print "Return:", round(apv,2)
    print "Sharpe Ratio:", round(sharpe_ratio,2)
    print "Maximum Drawdown:", round(max_dd,2)
    print "\n"

def print_best_stock_performance(returns,best_stock,rebalance_period,t,apv,sr,max_dd):
    print "-"*80
    print "Best Stock:", best_stock
    print "Period:", t/rebalance_period, returns.index[t - rebalance_period], returns.index[t]
    print "Annualised Return:", round(apv,2)
    print "Annualised Sharpe Ratio:", round(sr,2)
    print "Maximum Drawdown on period:", round(max_dd,2)
    print "\n"

def plot_markowitz_portfolio(results_frame):
    #Assert if no columns stdev or r or sharpe_ratio
    sharpe_ratio_max_idx = results_frame.sharpe_ratio.idxmax()
    r_max, sr_max = results_frame.loc[sharpe_ratio_max_idx][["r","stdev"]]

    volatility_min_idx = results_frame.stdev.idxmin()
    r_min, stdev_min = results_frame.loc[volatility_min_idx][["r","stdev"]]

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(15, 10))
    plt.scatter(results_frame.stdev,
                    results_frame.r,
                    c=results_frame.sharpe_ratio,
                    cmap='RdYlGn', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sr_max,r_max, marker='*',color='b',s=500, label='Maximum Sharpe ratio')
    plt.scatter(stdev_min,r_min, marker='*',color='r',s=500, label='Minimum Volatitlity')

    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.legend(labelspacing=0.8)

    scaling_parametr = 0.1
    y_min = (1-scaling_parametr)*results_frame.r.min()
    y_max = (1+scaling_parametr)*results_frame.r.max()

    plt.ylim(y_min, y_max)

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    ''' Portfolio perfomance '''
    days = 365
    # days = 1
    returns = np.sum(weights * np.transpose(mean_returns)) * days
    std = np.sqrt(np.dot(np.dot(weights, cov_matrix), np.transpose(weights))) * np.sqrt(days)

    return returns, std

def plot_portfolio_backtest(returns, i, j,name,save_results_dir):
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(20,10))
    plt.title(name)
    plt.xlabel('Date')
    plt.ylabel('Accumulated Returns')
    plt.plot(returns)
    plt.axhline(y=0, color='g', linestyle='--')
    plt.plot([i, j], [returns[i], returns[j]], 'o', color='Red', markersize=10)
    if save_results_dir is not None:
        plt.savefig(save_results_dir+'/{0}.png'.format(name))