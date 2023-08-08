import numpy as np
from PIL import Image
testIm = np.asarray(Image.open('kvinna.jpg').convert('L'))

'''
Inledning/Teori - Jacob


Hur man skapar matrisen - Jacob
'''
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

''' 
1-dimensionell transformation + 2-dimensionell transform - Peter
'''
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


'''
Invers - Elias
'''

def inverseHaarTransformation(arrs):
    imarray = np.vstack((np.concatenate(arrs[0,:], axis=1), np.concatenate(arrs[1,:], axis=1)))
    m = imarray.shape[0]
    n = imarray.shape[1]
    if not m % 2 == 0:
        imarray = imarray[:-1]
        m -= 1
    if not n % 2 == 0:
        imarray = imarray[:,:-1]
        n -= 1
    
    marr = haar_matrix(m, True)
    narr = haar_matrix(n, True).transpose()
    
    transformed = np.matmul(marr, imarray)
    transformed = np.matmul(transformed, narr)
    
    return transformed

'''
Kompression/iteration
'''
'''
Manuella medelvärden - Maximilian
'''
'''
Exempel/Demonstration
'''
