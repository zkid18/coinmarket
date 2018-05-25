from backtesting import metrics as mt
import numpy as np
import scipy.optimize as sco


def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    p_mu, p_std = mt.portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return - p_mu / p_std

def max_sharpe_ratio(mean_returns, cov_matrix, num_assets, risk_free_rate=0):
    args = (mean_returns, cov_matrix)
    x = np.ones(num_assets)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, x, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolio_volatility(weights, mean_returns, cov_matrix):
    p_mu, p_std = mt.portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return p_std

def min_variance(mean_returns, cov_matrix, num_assets):
    args = (mean_returns, cov_matrix)
    x = np.ones(num_assets)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, x, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target, num_assets):
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        p_mu, p_std = mt.portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        return p_mu

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range, num_assets):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret, num_assets))
    return efficients