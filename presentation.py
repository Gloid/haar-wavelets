import numpy as np
from PIL import Image
import matplotlib.pyplot as mplpy
testIm = np.asarray(Image.open('kvinna.jpg').convert('L'))


def HaarWaveletMatrix(n):
    """ Creates a Haar Wavelet matrix of size n x n:
    Parameters:
        n (int) represents the dimensions of the matrix
    on return:
        Returns a n x n Haar Wavelet matrix as an NumPy array
    """
    Matrix= np.zeros([n,n])
    
    for i in range(int(n/2)):
        Matrix[i,i*2]=1/2
        Matrix[i,i*2+1]=1/2
        Matrix[i+int(n/2),i*2]=-1/2
        Matrix[i+int(n/2),i*2+1]=1/2
    return np.sqrt(2)*Matrix

def HWT1D(image):
    '''
    Performs the 1-D Haar Wavelet transformation on an image
        Parameters:
            image (ndarray): The image of which the transform is applied to

        Returns:
            tuple (ndarray, ndarray):
                topIm (ndarray): the weighted averages of the columns
                botIm (ndarray): the weighted differences of the columns
    '''
    if np.shape(image)[0] % 2 != 0:
        np.delete(image,(np.shape(image)[0]-1),axis=0)
    Wm = HaarWaveletMatrix(np.shape(image)[0])
    newImage = np.matmul(Wm,image)
    newImage  = np.abs(np.rint(newImage))
    newImage *= (255.0/newImage.max())
    topIm = newImage[:int(np.shape(newImage)[0]/2), :]
    botIm = newImage[int(np.shape(newImage)[0]/2):, :]
    return topIm, botIm



def HWT2D(image):
    '''
    Performs the 2-D Haar Wavelet transformation on an image
        Parameters:
            image (ndarray): The image of which the transform is applied to

        Returns:
            tuple (ndarray, ndarray, ndarray, ndarray):
                topL (ndarray): the weighted averages
                topR (ndarray): the weighted differences of the rows
                botL (ndarray): the weighted differences of the columns
                botR (ndarray): the weighted differences of both the rows and columns
    ''' 
    if np.shape(image)[0] % 2 != 0:
        np.delete(image,(np.shape(image)[0]-1),axis=0)
    if np.shape(image)[1] % 2 != 0:
        np.delete(image, (np.shape(image)[1]-1),axis=1)
    Wm = HaarWaveletMatrix(np.shape(image)[0])
    Wn = HaarWaveletMatrix(np.shape(image)[1])
    newImage = np.matmul(np.matmul(Wm,image),Wn.transpose())
    newImage  = np.abs(np.rint(newImage))
    #Scaling values to fit [0,255]
    newImage *= (255.0/newImage.max())
    topL = newImage[0:int(np.shape(newImage)[0]/2), 0:int(np.shape(newImage)[1]/2)]
    topR = newImage[0:int(np.shape(newImage)[0]/2), int(np.shape(newImage)[1]/2):]
    botL = newImage[int(np.shape(newImage)[0]/2):, 0:int(np.shape(newImage)[1]/2)]
    botR = newImage[int(np.shape(newImage)[0]/2):, int(np.shape(newImage)[1]/2):]
    return topL, topR, botL, botR



def inverseHaarTransformation(arrs):
    """
    Performs the inverse HWT on a tuple of 4 subarrays.

    Args:
        arrs (Tuple): A tuple containing the 4 subarrays as returned from HWT2D.

    Returns:
        Array: An array representing the original image before the HWT was applied.
    """
    imarray = np.vstack((np.hstack((arrs[0], arrs[1])), np.hstack((arrs[2], arrs[3]))))
    m = imarray.shape[0]
    n = imarray.shape[1]
    
    marr = HaarWaveletMatrix(m).transpose()
    narr = HaarWaveletMatrix(n)
    
    transformed = np.matmul(marr, imarray)
    transformed = np.matmul(transformed, narr)
    
    return transformed

def Haarcompression(image, n = 1):
    """Compresses a image n+1 times:
    Parameters:
               image (ndarray) is the image to be compressed.
               n (int) number of iterations desired
    On return:
               Returns the n+1 compressed array and the four arrays from the n:the compressionround.
               """
    topL = image
    for i in range(n):
        topL, topR, botL, botR = HWT2D(topL)
        compressedimage = HWT2D(topL)[0]
    
    return compressedimage, topL, topR, botL, botR

def averages(image, n):
    """Compresses an image using block-averaging
    
    Parameters:
        image - the image that is compressed
        n - the number of compressions 

    On Return:
        Returns a compressed image using the averages in each block
    """
    rows = image.shape[0]
    cols = image.shape[1]
    new_rows, new_cols = rows // (2**n), cols // (2**n)
    newImage = np.zeros((new_rows, new_cols))

    for r in range(new_rows):
        for c in range(new_cols):
            block_sum = np.sum(image[r*2**n:(r+1)*2**n, c*2**n:(c+1)*2**n])
            newImage[r, c] = block_sum / (2**(2*n))

    return np.abs(newImage)

image = Image.open("Gruppbild.jpg").convert('L')
image = np.asarray(image)
compressedimage, topL, topR, botL, botR, = Haarcompression(image)
compressedimage = Image.fromarray(compressedimage)
mplpy.figure(figsize=(8,4))
mplpy.subplot(2,2,1)
mplpy.title('Top left, 1 iteration')
mplpy.imshow(topL, cmap='gray')
mplpy.subplot(2,2,2)
mplpy.title('Top right, 1 iteration')
mplpy.imshow(topR, cmap='gray')
mplpy.subplot(2,2,3)
mplpy.title('Bottom left, 1 iteration')
mplpy.imshow(botL, cmap='gray')
mplpy.subplot(2,2,4)
mplpy.title('Bottom right, 1 iteration')
mplpy.imshow(botR, cmap='gray')
mplpy.show()

mplpy.figure(figsize=(8,4))
mplpy.subplot(2,2,1)
mplpy.title('1 iteration')
mplpy.imshow(Haarcompression(image, 1)[0], cmap='gray')
mplpy.subplot(2,2,2)
mplpy.title('2 iterations')
mplpy.imshow(Haarcompression(image, 2)[0], cmap='gray')
mplpy.subplot(2,2,3)
mplpy.title('3 iterations')
mplpy.imshow(Haarcompression(image, 3)[0], cmap='gray')
mplpy.subplot(2,2,4)
mplpy.title('4 iterations')
mplpy.imshow(Haarcompression(image, 4)[0], cmap='gray')
mplpy.show()
