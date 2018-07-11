import unittest
import os

from utils.config_utils import *
from nn import *
from nn.nn_dispatcher import NN_DISPATCHER

class TestNN(unittest.TestCase):
	def setUp(self):
		self.cur_dir = os.path.dirname(__file__)
		self.config = get_config(self.cur_dir, "../config/config.json")

	def test_nn_not_none(self):
		neural_net = NN_DISPATCHER[self.config["nn"]](self.config)
		self.assertTrue(neural_net.build() is not None)


