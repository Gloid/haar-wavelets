
'''
Inledning/Teori - Jacob


Hur man skapar matrisen - Jacob
'''
def HaarWavelet(n):
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
