import tensorflow as tf

class RNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, hiddenstate, scope=None):
        scope = scope or type(self).__name__

        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope):
            W_h = tf.get_variable('W_h',
                                  [self.state_size, self.state_size],
                                  initializer=initializer)
            W_x = tf.get_variable('W_x',
                                  [self.input_size, self.state_size],
                                  initializer=initializer)
            b1 = tf.get_variable('b',
                                 [self.state_size, ],
                                 initializer=initializer)

            new_state = tf.nn.sigmoid(tf.matmul(state, W_h) + tf.matmul(inputs, W_x) + b1)

        new_hiddenstate = hiddenstate
        return new_state, new_hiddenstate