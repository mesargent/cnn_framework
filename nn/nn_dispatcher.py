from keras.optimizers import SGD

from nn import *

NN_DISPATCHER = {
	"shallownet": shallownet.ShallowNet
}

NN_PARAMS = {
	"sgd": SGD
}