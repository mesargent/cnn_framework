import os

import cv2
import numpy as np

from . import file_utils as fu

def load_images_with_labels(path_to_images, label, curr_dir="/"):
	rel_path = fu.get_relative_path(curr_dir, path_to_images)
	files = fu.get_all_files(rel_path)
	files = [fu.get_relative_path(curr_dir, path_to_images) + "/" + file for file in files]
	return [(cv2.imread(file), label) for file in files]