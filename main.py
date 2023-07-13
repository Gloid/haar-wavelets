import scipy.misc as sm
import imageio.v2 as iio
import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageOps
testIm = np.asarray(Image.open('kvinna.jpg').convert('L'))


def Wavelet(n):
    #Creates and returns a Wavelet matrix (n*n)
    #n (even integer) dimension of matrix
    W = np.zeros([n,n])
    for i in range(int(n/2)):
        W[i,i*2]=1/2
        W[i,i*2+1]=1/2
        W[i+int(n/2),i*2]=-1/2
        W[i+int(n/2),i*2+1]=1/2
    return np.sqrt(2)*W

def compression1d(image):
    #Returns ndarray of the 1-dimensional HWT of image
    #image (ndarray) 
    Wm = Wavelet(np.shape(image)[0])
    newImage = np.matmul(Wm,image)
    newImage  = np.abs(np.rint(newImage))
    newImage *= (255.0/newImage.max())
    return newImage


def compression2d(image):
    #Returns ndarray of the 2-dimensional HWT of image
    #image (ndarray) 
    Wm = Wavelet(np.shape(image)[0])
    Wn = Wavelet(np.shape(image)[1])
    newImage = np.matmul(np.matmul(Wm,image),Wn.transpose())
    newImage  = np.abs(np.rint(newImage))
    #Scaling values to fit [0,255]
    newImage *= (255.0/newImage.max())
    return newImage

plt.imshow(compression2d(testIm), cmap='gray',interpolation='nearest')
#plt.savefig('filtrerad_kvinna.jpg',bbox_inches='tight')
plt.show()

#Something wrong here, plt.imshow shows correct image when Image.show() does not
compressedImage = Image.fromarray(np.uint8((compression2d(testIm))), 'L')

compressedImage.show()
#compressedImage.save('komprimerad_kvinna.jpg')
