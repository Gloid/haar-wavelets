import scipy.misc as sm
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
im = iio.imread('kvinna.jpg')[:,:,0]
iio.imsave('new_kvinna.jpg',im)
print(np.shape(im))

def Wavelet(n):
    #Creates and returns a Wavelet matrix (n*n)
    #n (even integer) dimension of matrix
    W = np.zeros([n,n])
    for i in range(int(n/2)):
        W[i,i*2]=1/2
        W[i,i*2+1]=1/2
        W[i+int(n/2),i*2]=-1/2
        W[i+int(n/2),i*2+1]=1/2
    return W
print(Wavelet(10))