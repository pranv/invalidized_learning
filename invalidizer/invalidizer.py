# OM
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def split(patch, size):
	sub_patches = []

	nb_patches_per_dimension = patch.shape[0] / size

	for i in range(nb_patches_per_dimension):
		for j in range(nb_patches_per_dimension):
			start_x = j * size
			start_y = i * size
			end_x = j * size +  size
			end_y = i * size + size

			sub_patch = patch[start_x:end_x, start_y:end_y, :]
			sub_patches.append(sub_patch)

	return sub_patches

def join(patches):
	nb_patches = len(patches)
	nb_patches_per_dimension = int(nb_patches ** 0.5)

	rows = []

	for i in range(0, nb_patches, nb_patches_per_dimension):
		row = np.concatenate(patches[i:i+nb_patches_per_dimension], axis=1)
		rows.append(row)

	joined_patch = np.concatenate(rows, axis=0)

	return joined_patch

def get_noise_grid(image_length, tile_length, width):
	'''
		For a given 'tile_length' and 'image_length', generate a gaussian noise 3D tensor that 
		can be added to the invalid image to smooth out the boundaries.
	'''
	return 

def invalidizer(image, tile_length, window_length, sigma=3.0):
	image_length, _, _ = image.shape
	
	nb_windows = (image_length / window_length) ** 2
	nb_tiles = (window_length / tile_length) ** 2

	windows = split(image, window_length)

	for i in range(nb_windows):
		tiles = split(windows[i], tile_length)
		tiles = np.random.permutation(tiles)
		windows[i] = join(tiles)

	invalid_image = join(windows)
	
	smooth_invalid_image = invalid_image + get_noise_grid(image_length, tile_length, 8)

	return smooth_invalid_image