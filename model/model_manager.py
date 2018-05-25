from __future__ import print_function
import tensorflow as tf
import numpy as np
import portfolio_data as pf

def calcProfit(action, returns, inBatch = False):
	if not inBatch:
		return calcProfitNoBatch(action, returns)
	else:
		return calcProfitBatch(action, returns)

def calcProfitNoBatch(action, returns):
	profit = tf.reduce_sum(tf.multiply(action[:-1], returns))
	return profit

def calcProfitBatch(action, returns):
	profit = tf.reduce_sum(tf.multiply(action[:,:-1], returns), axis = 1)
	return tf.reduce_sum(profit)


# multiply a batch of matrix a tensor in the shape of [D, N, L]
# with a transformation matrix [L, transformSize]
# output a tensor of shape [D, N, transformSize]
def batchMatMul(a, b, D):
	out = []
	print("batchMatMul", a.shape)
	for i in range(D):
		out.append(tf.matmul(a[i], b))
	return tf.stack(out)

def train1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess, B = None):
	return trainOrTest1Epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess)

def test1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess, B = None):
	return trainOrTest1Epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess, training = False)


def trainOrTest1Epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess, training=True):
	totalIters = returnTensor.shape[0]
	prevLoss = 0.0
	D = len(prevReturnMatrix[0])
	prevA = pf.getInitialAllocation(D)
	allActions = []
	allLosses = []

	for t in range(totalIters-1):
		mRatio = pf.loss2mRatio(prevLoss)
		inputs = {
			'X': returnTensor[t],
			'prevReturn': prevReturnMatrix[t],
			'nextReturn': nextReturnMatrix[t],
			'prevA': prevA,
			'mRatio': mRatio
		}
		if training:
			curA, curLoss = curModel.train(inputs, sess)
		else:
			curA, curLoss = curModel.get_action(inputs, sess)
		allActions.append(curA)
		allLosses.append(curLoss)

		prevLoss = curLoss
		prevA = curA

	totalLoss = sum(allLosses)
	growthRates = map(lambda x: 1 - x, allLosses)

	return allActions, growthRates

def getDecayingWeights(tensorList):
	l = len(tensorList)
	cur = 1.0
	out = [cur]
	total = cur

	# create list
	for i in range(1, len(tensorList)):
		cur = cur * np.exp(1.0)
		out.append(cur)
		total += cur

	# normalize
	out = [ele/total for ele in out]

	# convert to tensor
	out = [tf.constant(ele, dtype = tf.float32) for ele in out]
	return out

def calcTransCost(action, prevAction, prevLogR, transCostParams, mRatio, inBatch = False):
	if not inBatch:
		return calcTransCostNoBatch(action, prevAction, prevLogR, transCostParams, mRatio)
	else:
		return calcTransCostBatch(action, prevAction, prevLogR, transCostParams, mRatio)


def calcTransCostNoBatch(action, prevAction, prevLogR, transCostParams, mRatio):
	c = transCostParams['c']
	c0 = transCostParams['c0']
	priceRatio = tf.exp(prevLogR)
	changes = tf.abs(action[:-1] - mRatio * tf.multiply(priceRatio, prevAction[:-1]))
	transactionCost = tf.reduce_sum( tf.multiply(c, changes) )
	transactionCost += c0
	return transactionCost


def calcTransCostBatch(action, prevAction, prevLogR, transCostParams, mRatio):
	c = transCostParams['c']
	c0 = transCostParams['c0']
	priceRatio = tf.exp(prevLogR)
	changes = tf.abs(action[:,:-1] - mRatio * tf.multiply(priceRatio, prevAction[:,:-1]))
	transactionCost = tf.reduce_sum( tf.multiply(c, changes) , axis = 1)
	transactionCost += c0
	return tf.reduce_sum(transactionCost)


def combineList(tensorList, weights):
	t = tensorList[0] * weights[0]
	for i in range(1, len(tensorList)):
		t += tensorList[i] * weights[i]
	return t


# assume the input is of shape [D, 1]
# add one more 0 to it (output shape [D+1, 1])
def addBias(x):
	x = tf.concat([x, tf.constant(np.array([[0.0]]), dtype = tf.float32)], axis = 0)
	return x