import cv2

from . preprocessor import Preprocessor

class ResizePreprocessor(Preprocessor):
	"""
	This class resizes an image to size provided in params
	"""
	def __init__(self, params=None):
		"""
		:param params: a dictionary of parameter values
		"""
		self.params = params

	def preprocess(self, img):
		return self.resize(img, tuple(self.params["preprocessor_params"]["resize"]["size"]))

	def resize(self, img, size):
		"""
		Resizes an image to dimensions tuple
		:param size: tuple width and height
		"""
		return cv2.resize(img, size[:2])