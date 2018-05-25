from sampleModel import Model, raiseNotDefined
import tensorflow as tf
import model_manager as mm
import json

class Config:
	lr = 1e-4
	dropout = 1.0
	modelType = 'Basic'

class BaseModel(Model):

    # object constructor
    # D : the dimension of the portfolio,
    # N : the number of days looking back
    def __init__(self, D, N,transCostParams, L=1):
        self.D = D
        self.N = N
        self.L = L
        self.config = Config
        self.transCostParams = {
            key: tf.constant(transCostParams[key], dtype=tf.float32) for key in transCostParams
            }
        self.build()

    # define the placeholders (add it to self.placeholders)
    # X: whatever input it may be
    # prevA: previous action (allocation)
    # prevReturn: S_t / S_t-1
    # nextReturn: S_t+1 / S_t
    # mRatio: M_t-1 / M_t
    def add_placeholders(self):

        xShape = (self.D, self.N, self.L)
        prShape = (self.D, 1)
        nrShape = (self.D, 1)
        paShape = (self.D + 1, 1)

        allShapes = [xShape, prShape, nrShape, paShape]

        self.placeholders = {
            'X': tf.placeholder(dtype=tf.float32, shape=allShapes[0]),
            'prevReturn': tf.placeholder(dtype=tf.float32, shape=allShapes[1]),
            'nextReturn': tf.placeholder(dtype=tf.float32, shape=allShapes[2]),
            'prevA': tf.placeholder(dtype=tf.float32, shape=allShapes[3]),
            'mRatio': tf.placeholder(dtype=tf.float32, shape=()),
        }

    # create feed dict (return it)
    def create_feed_dict(self, inputs):
        feed_dict = {
            self.placeholders[key]: inputs[key] for key in inputs
            }
        return feed_dict

    # add an action (add to self)
    def add_action(self):
        # each model must implement this method
        raiseNotDefined()

    # create loss from action (return it)
    def add_loss(self, action):
        # calculate profit from action
        prevReturn = self.placeholders['prevReturn']
        nextReturn = self.placeholders['nextReturn']
        mRatio = self.placeholders['mRatio']
        prevA = self.placeholders['prevA']

        transCostParams = self.transCostParams

        # calculate return (hence loss)
        profit = mm.calcProfit(action, nextReturn)
        transCost = mm.calcTransCost(action, prevA, prevReturn, transCostParams, mRatio)
        R = profit - transCost
        loss = R * (-1.0)
        return loss

    # define how to train from loss (return it)
    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    # train the model with 1 iteration
    # return action and loss
    def train(self, inputs, sess):
        feed_dict = self.create_feed_dict(inputs)
        action = sess.run(self.action, feed_dict=feed_dict)
        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return action, loss

    # get the action of the next time step
    # return action and loss
    def get_action(self, inputs, sess):
        feed_dict = self.create_feed_dict(inputs)
        action, loss = sess.run([self.action, self.loss], feed_dict=feed_dict)
        return action, loss

    def get_model_info(self):
        model_info = {
            'lr': self.config.lr,
            'dropout': self.config.dropout,
            'model_type': self.config.modelType
        }
        print("model info")
        print(json.dumps(model_info))
        print()