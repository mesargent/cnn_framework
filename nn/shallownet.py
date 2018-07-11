from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

from nn.neural_net import NeuralNet

class ShallowNet(NeuralNet):
	def __init__(self, config):
		self.config = config

	def build(self):
		model = Sequential()
		model.add(Conv2D(32, (3,3), padding="same", input_shape=tuple(self.config["input_img_size"])))
		model.add(Activation("relu"))
		model.add(Flatten())
		model.add(Dense(self.config["classes"]))
		model.add(Activation("softmax"))
		return model

	def print(self):
		print("NN with architecture: INPUT => CONV (32 3x3 filters) => RELU => FC, padding=same")