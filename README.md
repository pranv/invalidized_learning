# Invalidized Learning
All natural observations have inherent patterns in them and this makes the number of possible observations very less (manifold hypothesis). We aim to develop a way to generate good representations of data by training a classifier to diffrentiate between valid (naturally occuring) and invalid (synthetic) data. For each data point in our dataset, we break down the natural order in the it by splitting it up into small regions called **'tiles'** and shuffling these tiles randomly. The suffling is done locally in a neighbourhood which we call the **'window'**. By varying the size of the tiles from the lowest value to the largest and corresponsingly the window size, we get a set of invalid examples for each valid example. These data points are invalid due to inconsistencies at various levels of abstraction. We hope to show that when a Deep Neural Network is trained to diffrentiate between these valid data points and invalid data points, it generates a heirarchial representation of the data. Our intuition is that the features learnt will be relavent to only the natural occuring inputs as learning from randomness is prohibitively expensive.
## Example In The Case of Image Data:
Most permutations of RGB values won't result in a valid image. Further, stepping up a level of abstraction, most permutations of edges won't result in a valid image, and most combinations of contours won't lead to a valid image. 

Accordingly, we split the image into small square patches called tiles and then to obtain an invalid image by shuffling them. We shuffle at the tiles that only lie in a particular part of the image to preserve the components of a feature locally. This also reduces the chances of rapid changes of color. 

By varying the size of the tile, from one pixel to several pixels, we get a set of invalid images. We smoothen the edges between two random patches with interpolation. Random translation, rotation, noise and larger dropout at all stages are used to make sure that there are no other trends that differentiates valid and invalid images.

As per the above explanation, these invalid images are invalid at different levels of abstraction. When the tile size is small, the image is invalid due to lack of edges. When the tile size is big, the image is invalid due to lack of a naturally observable contour or shape. 

We then train a DNN to differentiate between valid and invalid images. We hope to see features observed in natural images like edges, contours and shapes are learnt at various layers of the DNN. Randomly shuffled inputs do not have any consistent pattern that the DNN can exploit to differentiate between the two. Hence it should be forced to have activations for features of natural images only.

Here is an example from ImageNet:

<img src="https://cloud.githubusercontent.com/assets/8753078/9814330/aca56cb6-58aa-11e5-837c-56602ab9c820.png" width="45%">
<img src="https://cloud.githubusercontent.com/assets/8753078/9814331/ad35e368-58aa-11e5-9c8a-5c43f9e9b789.png" width="45%">


## Related Work
Although independently conceived and developed, this idea is similar to the works 'Noise Contrastive Estimation' (Gutmann et al.) and 'Generative Adversarial Nets' (Goodfellow et al.). The difference however is in the methods employed to generate invalid data points and the distributions of the inputs provided to the classifiers.
