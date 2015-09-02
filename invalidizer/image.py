import numpy as np
from copy import deepcopy
from scipy.ndimage.interpolation import rotate

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

def smoothen(image, image_length, tile_length, width, num_channels):
	'''
		Roughly interpolate the edges
	'''
	for w in range(width):
		for a in range(tile_length, image_length, tile_length):
			for i in range(a-w, a+w):
				mean = deepcopy(np.mean(image[i:i+2, :, :], axis=0))
				mean = mean.reshape(1, image_length, num_channels)
				mean = np.concatenate([mean, mean], axis=0)
				image[i:i+2, :, :] =  mean
				
				mean = deepcopy(np.mean(image[:,i:i+2, :], axis=1))
				mean = mean.reshape(image_length,1, num_channels)
				mean = np.concatenate([mean, mean], axis=1)
				image[:,i:i+2, :] =  mean
			
			for i in range(a+w, a-w):
				mean = deepcopy(np.mean(image[i:i+2, :, :], axis=0))
				mean = mean.reshape(1, image_length, num_channels)
				mean = np.concatenate([mean, mean], axis=0)
				image[i:i+2, :, :] =  mean
				
				mean = deepcopy(np.mean(image[:,i:i+2, :], axis=1))
				mean = mean.reshape(image_length,1, num_channels)
				mean = np.concatenate([mean, mean], axis=1)
				image[:,i:i+2, :] =  mean
				
	return image

def _rotate(image, angle):
	rotated = rotate(image,angle)
	return rotated

def _translate(image, trans_x, trans_y):
	image_length, _, num_channels = image.shape
	translate = image[trans_x:, trans_y:, :]
	t_y = np.concatenate([translate, np.zeros(image_length, trans_y, num_channels)], axis=0)
	t_x = np.concatenate([t_y, np.zeros(trans_x, image_length-trans_y, num_channels)], axis=1)
	return translate

def invalidizer(image, tile_length, window_length):
	image_length, _, num_channels = image.shape
	
	nb_windows = (image_length / window_length) ** 2
	nb_tiles = (window_length / tile_length) ** 2

	windows = split(image, window_length)

	for i in range(nb_windows):
		tiles = split(windows[i], tile_length)
		tiles = np.random.permutation(tiles)
		windows[i] = join(tiles)

	invalid_image = join(windows)
	smooth_invalid_image = smoothen(invalid_image, image_length, tile_length, tile_length/4, num_channels)

	translated = _translate(smooth_invalid_image, np.random.random_integers(0, image_length/4), np.random.random_integers(0, image_length/4))
	rotated = _rotate(translated, np.random.random_integers(0, 20))
	return rotated