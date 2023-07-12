import scipy.misc as sm
import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
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
    Wm = Wavelet(np.shape(image)[0])
    return np.abs(np.rint(np.matmul(Wm,image))).astype('int')


def compression2d(image):
    Wm = Wavelet(np.shape(image)[0])
    Wn = Wavelet(np.shape(image)[1])
    return np.abs(np.rint(np.matmul(np.matmul(Wm,image),Wn.transpose()))).astype('int')

plt.imshow(compression2d(testIm), cmap='gray',interpolation='nearest')
plt.savefig('filtrerad_kvinna.jpg',bbox_inches='tight')
plt.show()

#Dont know why ImageOps.grayscale is necessary, get error if removed
#Saved image has poor quality compared to the plotted image
compressedImage = ImageOps.grayscale(Image.fromarray(compression2d(testIm)))

compressedImage.save('filtrerad_kvinna.jpg')
