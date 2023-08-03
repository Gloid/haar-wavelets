"""
Project suggested solution

Jacob Jannering
"""
#Importing packages

import numpy as np
from PIL import Image

#Import and process image

def ImageImport():
    image=Image.open("/Users/jacobjannering/Documents/Jacob/Lunds Universitet/Project/kvinna.jpg")
    image=image.convert("L")
    image=np.array(image)
    
    shape=np.shape(image)

    if shape[0] % 2 !=0:
        image=image[:-1,:]

    elif shape[1] % 2 !=0:
        image=image[:,:-1]
    
    return image
    
#Creating the HaarWavelet matrix
    
def HaarWavelet(n):
    
    Matrix= np.zeros([n,n])
    
    for i in range(int(n/2)):
        Matrix[i,i*2]=1/2
        Matrix[i,i*2+1]=1/2
        Matrix[i+int(n/2),i*2]=-1/2
        Matrix[i+int(n/2),i*2+1]=1/2
    return np.sqrt(2)*Matrix


#Conduct the transformation
def HWT(image):
    
    Wm=HaarWavelet(np.shape(image)[0])
    Wn=HaarWavelet(np.shape(image)[1])
    
    HWT= np.matmul(np.matmul(Wm,image),Wn.transpose())
    HWT=np.rint(HWT)
    HWT=np.abs(HWT)
    HWT1= HWT[:HWT.shape[0]//2, :HWT.shape[1]//2]
    HWT2= HWT[:HWT.shape[0]//2, HWT.shape[1]//2:]
    HWT3= HWT[HWT.shape[0]//2:, :HWT.shape[1]//2]
    HWT4= HWT[HWT.shape[0]//2:, HWT.shape[1]//2:]
    
    return HWT1,HWT2, HWT3,HWT4


def InverseHWT(HWT1,HWT2, HWT3, HWT4):
    TopHalf=np.concatenate((HWT1,HWT2),axis=1)
    BottomHalf= np.concatenate((HWT3,HWT4), axis=1)
    HWT=np.concatenate((TopHalf, BottomHalf), axis=0)
    Wm=HaarWavelet(np.shape(HWT)[0])
    Wn=HaarWavelet(np.shape(HWT)[1])
    
    ReconstructedImage= np.matmul(np.matmul(Wm.transpose(), HWT), Wn)
    
    return ReconstructedImage


def HaarCompression(image,n):
    
    for i in range(n):
        Wm=HaarWavelet(np.shape(image)[0])
        Wn=HaarWavelet(np.shape(image)[1])
        
        image= np.matmul(np.matmul(Wm,image),Wn.transpose())
        image=np.rint(image)
        image=np.abs(image)
        image= image[:image.shape[0]//2, :image.shape[1]//2]

    return image



#RUNNING THE CODE

#Import the picture
image=ImageImport()


#Conduct the compression
img1,img2,img3,img4= HWT(image)
compressedimage=HaarCompression(image,1)
image=Image.fromarray(compressedimage)
image.show()


#Inverse the compression
inversecompression=InverseHWT(img1,img2,img3,img4)
image=Image.fromarray(inversecompression)
image.show()





#HWT without matrices

def ManualTransform(image):
    shape=np.shape(image)
    
    rows=shape[0]
    columns=shape[1]
    
    HWT_rows=np.zeros([rows,columns])
    
    for i in range(rows):
        for j in range(0,columns,2):
            HWT_rows[i,j]=(image[i,j]+image[i,j+1])/2
            HWT_rows[i,j+1]=(image[i,j]-image[i,j+1])/2
    
    HWT_columns=np.zeros([rows,columns])
    
    for j in range(columns):
        for i in range(0,rows,2):
            HWT_columns[i,j]=(HWT_rows[i,j]+HWT_rows[i+1,j])/2
            HWT_columns[i+1,j]=(HWT_rows[i,j]-HWT_rows[i+1,j])/2
    
    return HWT_columns


#Running the code
image=ImageImport()
compressedimage=ManualTransform(image)
image=Image.fromarray(compressedimage)
image.show()

    
    
    
    
    