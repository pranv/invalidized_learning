import numpy as np
from copy import deepcopy

from scipy.ndimage import gaussian_filter

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

def smoothen(image, image_length, tile_length, width):
	'''
		For a given 'tile_length' and 'image_length', generate a gaussian noise 3D tensor that 
		can be added to the invalid image to smooth out the boundaries.
	'''

	for w in range(width):
		for a in range(tile_length, image_length, tile_length):
			for i in range(a-w, a+w):
				mean = deepcopy(np.mean(image[i:i+2, :, :], axis=0))
				mean = mean.reshape(1, image_length, 3)
				mean = np.concatenate([mean, mean], axis=0)
				image[i:i+2, :, :] =  mean
				
				mean = deepcopy(np.mean(image[:,i:i+2, :], axis=1))
				mean = mean.reshape(image_length,1, 3)
				mean = np.concatenate([mean, mean], axis=1)
				image[:,i:i+2, :] =  mean
			
			for i in range(a+w, a-w):
				mean = deepcopy(np.mean(image[i:i+2, :, :], axis=0))
				mean = mean.reshape(1, image_length, 3)
				mean = np.concatenate([mean, mean], axis=0)
				image[i:i+2, :, :] =  mean
				
				mean = deepcopy(np.mean(image[:,i:i+2, :], axis=1))
				mean = mean.reshape(image_length,1, 3)
				mean = np.concatenate([mean, mean], axis=1)
				image[:,i:i+2, :] =  mean
				
	return image

def invalidizer(image, tile_length, window_length):
	image_length, _, _ = image.shape
	
	nb_windows = (image_length / window_length) ** 2
	nb_tiles = (window_length / tile_length) ** 2

	windows = split(image, window_length)

	for i in range(nb_windows):
		tiles = split(windows[i], tile_length)
		tiles = np.random.permutation(tiles)
		windows[i] = join(tiles)

	invalid_image = join(windows)
	
	smooth_invalid_image = smoothen(invalid_image, image_length, tile_length, tile_length/4)

	return smooth_invalid_image
def translate(image,image_size,shift_size_x,shift_size_y):
	translate=deepcopy(image)
	for i in range(image_size):
		for j in range(shift_size_x):
			for k in range(image.shape[2]):
				translate[i,j,k]=0
	for i in range(shift_size_y):
		for j in range(image_size):
			for k in range(image.shape[2]):
				translate[i,j,k]=0
	return translate	
		
def rotate(image,angle):
	rot=ndimage.interpolation.rotate(image,angle)
	return rot
