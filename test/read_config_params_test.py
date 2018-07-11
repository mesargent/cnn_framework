import unittest

from utils.config_utils import *

class TestConfigParamImports(unittest.TestCase):

	def test_params_not_empty(self):
		params = get_config(os.path.dirname(__file__), "../config/config.json")
		self.assertTrue(params)

	def test_pos_image_file_path_read_correctly(self):
		params = get_config(os.path.dirname(__file__), "../config/config.json")
		self.assertEqual(params["pos_images_path"], "data/pos_images/")

	def test_neg_image_file_path_read_correctly(self):
		params = get_config(os.path.dirname(__file__), "../config/config.json")
		self.assertEqual(params["neg_images_path"], "data/neg_images/")