from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class NeuralNet:
	"""
	Empty neural net; do not initialize
	"""
	def __init__(self, config):
		self.config = config

	def build(self):
		return None

	def print(self):
		print("Empty neural net, do not initialize")