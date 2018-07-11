import unittest
import os

import numpy as np

from utils.config_utils import *
from utils.file_utils import *
from utils.data_utils import *


class TestDataLoad(unittest.TestCase):

	def test_get_relative_path_up_one(self):
		self.assertEqual(get_relative_path("/home/mark/Desktop/cnet/util", "../pos_image"), 
			"/home/mark/Desktop/cnet/pos_image")

	def test_get_relative_path_down_one(self):
		self.assertEqual(get_relative_path(os.path.dirname(__file__), "all_good"), 
			os.path.dirname(__file__) + "/all_good")

	def test_get_absolute_path(self):
			self.assertEqual(get_relative_path("/", "all_good"), "/all_good")

	#testing when root is reached test_load_images but relative path keeps going up
	def test_tried_to_go_up_from_root(self):
		self.assertEqual(get_relative_path("/", "../../all_good"), "/all_good")

	#testing to make sure list exists and isn't empty
	def test_file_load_not_empty(self):
		self.assertTrue(get_all_files(get_relative_path(os.path.dirname(__file__), 
			"../data/pos_images")))

	def test_file_load_has_correct_number(self):
		no_files = len(get_all_files(get_relative_path(os.path.dirname(__file__), 
			"../data/pos_images")))
		self.assertTrue(no_files, 212) 

	def test_load_images_with_labels_not_empty(self):
		labeled_images = load_images_with_labels("../data/pos_images", 1, os.path.dirname(__file__))
		self.assertTrue(labeled_images)

	def test_load_images_with_labels_has_right_dtype(self):
		labeled_images = load_images_with_labels("../data/pos_images", 1, os.path.dirname(__file__))
		self.assertTrue(isinstance(labeled_images[0], tuple))
		self.assertTrue(labeled_images[0][0] is not None)
		self.assertEqual(labeled_images[0][0].dtype, 'uint8')
		self.assertEqual(labeled_images[0][0].shape[2], 3)
		self.assertTrue(isinstance(labeled_images[0][1], int))

	def test_load_images_has_correct_number(self):
		labeled_images = load_images_with_labels("../data/pos_images", 1, os.path.dirname(__file__))
		self.assertEqual(len(labeled_images), 212)