# Reinforcement Learning for Portfolio Management
Portfolio management is the art and science of decision-making process about investment for individuals and institutions. This project presents a Reinforcement Learning framework for cryptocurrency portfolio management. Cryptocurrency is a digital decentralized asset. The most well-known example are Bitcoin and Ethereum. To accomplish the same level of performance as a human-trader, agents have to learn for themselves how to create successful biased-free strategies. This work covers different reinforcement learning approaches. The framework is designed using Deterministic Policy Gradient using  Recurrent Neural Networks (RNN). The robustness and feasibility of the system is verified with the market data from Poloniex exchange and compared to the the major portfolio management benchmarks.


####  Backtesting results

| Techniques        | APV           | Sharpe Ratio  | MDD |
| ------------ |:------:| ------:| ------:|
| UBAH     | -0.19 | -0.78 | 0.71 |
| BEST      | 0.59 |  1.0 | 1.32 |
| Maximum Sharpe Ratio | -0.31 | -1.32 | 0.72 |
| Minimum Varaince Strategy | -0.23 | -1.08 | 0.64 |
| Maximum Sharpe Ratio (rebalanced) | 0.59 | 1.98 | 1.45 |
| Minimum Varaince Strategy (rebalanced) | -0.07 | -1.31 | 0.41 |
| DPG-RNN | -0.16 | -0.76 | 0.58 |

Code implementation can be found in .ipynb file
#### Requirments
matplotlib 2.0.2
numpy 1.13.3
pandas 0.20.1
pip 9.0.1
seaborn 0.8.0
sklearn 0.18.1
xgboost 0.6
Python version: 2.7.13 |Anaconda custom (x86_64)| (default, Dec 20 2016, 23:05:08) 
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)]
Processor info : Intel(R) Core(TM) i5-5257U CPU @ 2.70GHz

More details can be found in .pdf file

