import scipy.misc as sm
import imageio.v2 as iio
import numpy as np
import time as tm

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

def averages(image, n):
    newImage = np.zeros([np.shape(image)[0], np.shape(image)[1]])
    rows = np.shape(image)[0]
    cols = np.shape(image)[1]
    for i in range(n):
        for r in range(int(rows/2)):
            for c in range(int(cols/2)):
                newImage[r,c]= image[2*r,2*c]/4 + image[2*r+1,2*c]/4 + image[2*r,2*c+1]/4 + image[2*r+1,2*c+1]/4
                newImage[r+int(rows/2),c] = -1*image[2*r,2*c]/4 + image[2*r+1,2*c]/4 - image[2*r,2*c+1]/4 + image[2*r+1,2*c+1]/4
                newImage[r,c+int(cols/2)] = image[2*r,2*c]/4 - image[2*r+1,2*c]/4 + image[2*r,2*c+1]/4 - image[2*r+1,2*c+1]/4
                newImage[r+int(rows/2),c+int(cols/2)]= image[2*r,2*c]/4 - image[2*r+1,2*c]/4 - image[2*r,2*c+1]/4 + image[2*r+1,2*c+1]/4
    return np.abs(newImage)


t01 = tm.time()
compressedImage1 = Image.fromarray(np.uint8((HWT2D(testIm))), 'L')
t11 = tm.time()
compressedImage1.show()
t02 = tm.time()
compressedImage2 = Image.fromarray(np.uint8(averages(testIm)), 'L')
t12 = tm.time()
compressedImage2.show()

print(f"Time for Haar transformation: {t11-t01}s")
print(f"Time for manual transformation: {t12-t02}s")

#originalImage = Image.fromarray(np.uint8((inverseHaarTransformation(HWT2D(testIm)))), 'L')
#originalImage.show()
#compressedImage.save('komprimerad_kvinna.jpg')
