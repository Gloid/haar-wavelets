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

def HWT1D(image):
    #Returns ndarray of the 1-dimensional HWT of image
    #image (ndarray) 
    if np.shape(image)[0] % 2 != 0:
        np.delete(image,(np.shape(image)[0]-1),axis=0)
    Wm = Wavelet(np.shape(image)[0])
    newImage = np.matmul(Wm,image)
    newImage  = np.abs(np.rint(newImage))
    newImage *= (255.0/newImage.max())
    return newImage


def HWT2D(image):
    #Returns ndarray of the 2-dimensional HWT of image
    #image (ndarray) 
    if np.shape(image)[0] % 2 != 0:
        np.delete(image,(np.shape(image)[0]-1),axis=0)
    if np.shape(image)[1] % 2 != 0:
        np.delete(image, (np.shape(image)[1]-1),axis=1)
    Wm = Wavelet(np.shape(image)[0])
    Wn = Wavelet(np.shape(image)[1])
    newImage = np.matmul(np.matmul(Wm,image),Wn.transpose())
    newImage  = np.abs(np.rint(newImage))
    #Scaling values to fit [0,255]
    newImage *= (255.0/newImage.max())
    return newImage

def inverseHaarTransformation(im1,im2,im3,im4):
    newImage = np.concatenate(np.concatenate(im1,im2,axis=0), np.concatenate(im3,im4,axis=0),axis=1)
    Wm = Wavelet(np.shape(newImage)[0])
    Wn = Wavelet(np.shape(newImage)[1])    
    newImage = np.matmul(np.matmul(Wm.transpose(),newImage),Wn)
    newImage  = np.abs(np.rint(newImage))
    newImage *= (255.0/newImage.max())

    return newImage

def inverseHaarTransformation(newImage):
    Wm = Wavelet(np.shape(newImage)[0])
    Wn = Wavelet(np.shape(newImage)[1])    
    newImage = np.matmul(np.matmul(Wm.transpose(),newImage),Wn)
    newImage  = np.abs(np.rint(newImage))
    newImage *= (255.0/newImage.max())

    return newImage

def Haarcompression(image, n):
    for i in range(n):
        image = HWT2D(image)
        image = image[0:int(np.shape(image)[0]/2),0:int(np.shape(image)[1]/2)]
    return image


compressedImage = Image.fromarray(np.uint8((Haarcompression(testIm,3))), 'L')

compressedImage.show()

#originalImage = Image.fromarray(np.uint8((inverseHaarTransformation(HWT2D(testIm)))), 'L')
#originalImage.show()
#compressedImage.save('komprimerad_kvinna.jpg')
