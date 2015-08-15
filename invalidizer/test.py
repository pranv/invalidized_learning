import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from invalidizer import invalidizer

'''
tile_size = 25
window_size = 100
image = np.array(Image.open('test_images/test.jpg'))
new_image = invalidizer(image, tile_size, window_size, sigma=1.0)

plt.ion()
plt.imshow(image)
raw_input()
plt.imshow(new_image)
raw_input()
'''

tile_size = 64
window_size = 128
image = np.array(Image.open('test_images/test2.jpg'))
new_image = invalidizer(image, tile_size, window_size, sigma=1.0)

plt.ion()
plt.imshow(image)
raw_input()
plt.imshow(new_image)
raw_input()