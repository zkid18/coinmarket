def raiseNotDefined():
	raise NotImplementedError("Each Model must re-implement this method.")

class Model(object):
	def __init__(self):
		raiseNotDefined()

	# create feed dict (return it)
	def create_feed_dict(self, inputs):
		raiseNotDefined()

	# define the variables (add it to self.placeholders)
	def add_placeholders(self):
		raiseNotDefined()

	# add an action (add it to self)
	def add_action(self):
		raiseNotDefined()

	# create loss from action (return it)
	def add_loss(self, action):
		raiseNotDefined()

	# define how to train from loss (return it)
	def add_train_op(self, loss):
		raiseNotDefined()

	# train the model with 1 iteration
	# return action and loss
	def train(self, inputs, sess):
		raiseNotDefined()

	# get the action of the next time step
	# return action and loss
	def get_action(self, inputs, sess):
		raiseNotDefined()

	# get model meta data
	def get_model_info(self):
		raiseNotDefined()

	# build the computation graph (add them to self)
	# called after initializing an instance of the object
	def build(self):
		self.add_placeholders()
		self.add_action()
		self.loss = self.add_loss(self.action)
		self.train_op = self.add_train_op(self.loss)