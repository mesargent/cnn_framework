from processors import *

# Map for all currently supported preprocessors
# To add a new, custom preprocessor, enter the config name as the key, 
# and the constructor function name (without quotes or parentheses) as the value
PREPROCESSOR_DISPATCHER = {
	"resize": resize_preprocessor.ResizePreprocessor,
	"do_nothing": do_nothing_preprocessor.DoNothingPreprocessor
}