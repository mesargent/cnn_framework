import unittest
import os
import cv2
import numpy as np

from utils.config_utils import *
from utils.file_utils import *
from utils.data_utils import *
from processors.resize_preprocessor import ResizePreprocessor
from processors.master_preprocessor import MasterPreprocessor
from processors.preprocessor import Preprocessor


class TestImagePreprocess(unittest.TestCase):
	def setUp(self):
		self.cur_dir = os.path.dirname(__file__)
		self.config = get_config(self.cur_dir, "../config/config.json")
		self.image_list = labeled_images = load_images_with_labels("../data/pos_images", 1, self.cur_dir)

	def test_image_resize(self):
		processor = ResizePreprocessor(self.config)
		path = get_relative_path(self.cur_dir, "../data/pos_images/cats_00137.jpg")
		img = cv2.imread(path)
		self.assertTrue(img is not None)
		img = processor.preprocess(img)
		self.assertEqual(img.shape, tuple(self.config["preprocessor_params"]["resize"]["size"]))

	def test_preprocessor_list_generate_not_empty_or_None(self):
		preprocessor_generator = MasterPreprocessor(self.config)
		self.assertTrue(preprocessor_generator.generate_preprocessor_list())

	def test_preprocessor_list_generate_preprocessors_right_type(self):
		preprocessor_generator = MasterPreprocessor(self.config)
		are_right_type = all([isinstance(p, Preprocessor) for p in preprocessor_generator.generate_preprocessor_list()])
		self.assertTrue(are_right_type)

	def test_preprocess_list_of_images_not_none(self):
		master_preprocessor = MasterPreprocessor(self.config)
		self.image_list = master_preprocessor.preprocess_image_list(self.image_list)
		size = tuple(self.config["preprocessor_params"]["resize"]["size"])
		self.assertTrue([example[0].shape == size for example in self.image_list] is not None)

	def test_preprocess_list_of_images(self):
		master_preprocessor = MasterPreprocessor(self.config)
		self.image_list = master_preprocessor.preprocess_image_list(self.image_list)
		#testing if all of the images are reshaped to width and height in config["size"]
		size = tuple(self.config["preprocessor_params"]["resize"]["size"])
		self.assertTrue(any([example[0].shape == size for example in self.image_list]))


