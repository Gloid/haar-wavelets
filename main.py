import scipy.misc as sm
import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
image = Image.open('kvinna.jpg').convert('L')


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

def compression1d(image):
    Wm = Wavelet(np.shape(image)[0])
    return np.abs(np.rint(np.matmul(Wm,image))).astype('int')


def compression2d(image):
    Wm = Wavelet(np.shape(image)[0])
    Wn = Wavelet(np.shape(image)[1])
    return np.rint(np.matmul(np.matmul(Wm,im),Wn.transpose())).astype('int')

compressedImage = Image.fromarray(compression1d(image))
compressedImage.save('filtrerad_kvinna.jpg')
