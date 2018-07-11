import os

def get_relative_path(cur_dir, to_dir):
	to_dir_args = to_dir.split(os.sep)
	levels_up = to_dir_args.count('..')

	for i in range(levels_up):
		cur_dir = os.path.dirname(cur_dir)

	return os.path.join(cur_dir, *to_dir_args[levels_up:])

#note: files won't necessarily be in any order
def get_all_files(path):
    files = filter(lambda f: not f.startswith('.'), os.listdir(path))
    return list(files)


