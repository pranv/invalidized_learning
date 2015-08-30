import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from invalidizer import invalidizer


tile_size = 50
window_size = 250
image = np.array(Image.open('test_images/test.jpg'))
image = invalidizer(image, 500, 500)
new_image = invalidizer(image, tile_size, window_size)

plt.ion()
plt.imshow(image)
raw_input()
plt.imshow(new_image)
raw_input()


tile_size = 64
window_size = 128
image = np.array(Image.open('test_images/test2.jpg'))
image = invalidizer(image, 256, 256)
new_image = invalidizer(image, tile_size, window_size)

plt.ion()
plt.imshow(image)
raw_input()
plt.imshow(new_image)
raw_input()