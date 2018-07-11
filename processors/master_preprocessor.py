from processors import *
from processors.preprocessor_dispatcher import PREPROCESSOR_DISPATCHER

class MasterPreprocessor:
	def __init__(self, config=None):
		self.config = config
		self.preprocessors = None

	def generate_preprocessor_list(self):
		preprocessors = []
		for p in self.config["preprocessors"]:
			try:
				preprocessors.append(PREPROCESSOR_DISPATCHER[p](self.config))
			except KeyError as e:
				print("Not all processors in config are also in PREPROCESSOR_DISPATCHER :" + str(e))
		return preprocessors

	def preprocess_image_list(self, images):
		if not self.preprocessors:
			self.preprocessors = self.generate_preprocessor_list()
		for p in self.preprocessors:
			return [(p.preprocess(example[0]), example[1]) for example in images]
