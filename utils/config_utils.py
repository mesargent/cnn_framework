import json
import sys, os
from pprint import pprint

from . import file_utils as fu


def get_config(curr_dir, config_path):
	config_path = fu.get_relative_path(curr_dir, config_path)
	with open(config_path) as f:
		return json.load(f)