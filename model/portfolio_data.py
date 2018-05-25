import numpy as np
import pandas as pd
import time, datetime
import logging
import os


def get_max_shape(d):
    max_shape = 0
    for k in d:
        if len(d[k]) > max_shape:
            max_shape = len(d[k])
            coin = k
    return max_shape, coin


def returns_data_frame(historical_data, window, coins, return_type="close_rate"):
    # TO-DO
    # Add asserts for case invalid historical data format
    # Check if window < trade_period
    trade_period, coin = get_max_shape(historical_data)
    df = pd.DataFrame(index=historical_data[coin][-window:].index, columns=[historical_data.keys()])

    for token in historical_data.keys():
        if return_type == "close_rate":
            df[token] = historical_data[token][-window:].close / historical_data[token][-window:].close.shift(1) - 1
        if return_type == "log":
            df[token] = np.log(historical_data[token][-window:].close) - np.log(
                historical_data[token][-window:].close.shift(1))
        if return_type == "open_close_rate":
            df[token] = historical_data[token][-window:].close / historical_data[token][-window:].open

    df.fillna(0, inplace=True)
    return df[coins][1:]

def extendDim(arr):
	newshape = tuple(list(arr.shape) + [1])
	return np.reshape(arr, newshape)

def getInitialAllocation(D):
	prevA = np.array([0.0 for _ in range(D + 1)])
	prevA[-1] = 0.0
	return extendDim(prevA)

# get accumulated return based on growth rates (a list of floats like 1.02)
# e.g. [1.2, 1.0, 1.5] -> [1.2, 1.2, 1.8]
def getAccumulatedReturn(growthRates):
	cur = growthRates[0]
	out = [cur]
	for i in range(1, len(growthRates)):
		cur *= growthRates[i]
		out.append(cur)
	return out

def prod(arr):
	p = 1.0
	for ele in arr:
		p *= ele
	return p

def loss2mRatio(loss):
	return 1.0 / (1.0 - loss)

def setupLogger(outPath):
	#createPath
	os.mkdir(outPath)

	# get the output file name
	logFileName = outPath + '/model_log.txt'

	# create logger
	logger = logging.getLogger('drl_in_pm')

	# create formatter
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

	# get the file handler
	hdlr = logging.FileHandler(logFileName)
	hdlr.setFormatter(formatter)

	# get the stream handler for system stdout
	sh = logging.StreamHandler()
	sh.setFormatter(formatter)

	# add the handlers
	logger.addHandler(hdlr)
	logger.addHandler(sh)

	# set level to debug
	logger.setLevel(logging.DEBUG)
	return logger

def featureTensor(inputs, N, W, M, L):
    lRMtx = np.empty((W - N, M, N, L))
    for i in range(N,W):
        lRMtx[i-N] = np.swapaxes(inputs[i - N: i],0,1)
    return lRMtx

def logReturn(prices, W, D):
    logReturnPrices = np.diff(np.log(np.transpose(prices)))
    logReturnPrices = np.swapaxes(logReturnPrices, 0, 2)
    return logReturnPrices

def getInputs(stockData, N):
	'''
    stockData (WxMxL)
    W - window periods
    M - numer of assets
    L - features space
    L == 1 - close data.
    N - states of N previous days
    '''

	W, M, L = stockData.shape
	closePrice = stockData[:, :, 0].reshape(W, M, 1)
	returnMatrix = logReturn(closePrice, W, M)
	prevReturnMatrix = returnMatrix[N - 1:-1, :]
	nextReturnMatrix = returnMatrix[N:, :, :]
	returnTensor = featureTensor(stockData, N, W, M, L)
	return returnTensor, prevReturnMatrix, nextReturnMatrix

