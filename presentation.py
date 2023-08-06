
'''
Inledning/Teori - Jacob


Hur man skapar matrisen - Jacob
'''
def HaarWavelet(n):
    
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
'''
Kompression/iteration
'''
'''
Manuella medelvärden - Maximilian
'''
'''
Exempel/Demonstration
'''
