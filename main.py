import scipy.misc as sm
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
im = iio.imread('kvinna.jpg')[:,:,0]
iio.imsave('new_kvinna.jpg',im)
print(np.shape(im))