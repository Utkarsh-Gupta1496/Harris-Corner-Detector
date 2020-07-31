"""
Created on Sat Nov  2 19:32:41 2019

@author: Utkarsh
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

def gaussian_kernel(size, sigma=1):
    size = int(size)  //2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float64)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (Ix,Iy)

img=cv2.imread('.images/PCB.jpg',0)
img11=cv2.imread('.images/PCB.jpg')
img=np.asarray(img,np.float32)

Ix,Iy=sobel_filters(img)

j11=np.multiply(Ix,Ix)
j22=np.multiply(Iy,Iy)
j12=j21=np.multiply(Ix,Iy)
g=gaussian_kernel(5,1)
j11=ndimage.convolve(j11, g)
j22=ndimage.convolve(j22, g)
j12=ndimage.convolve(j12, g)
j21=ndimage.convolve(j21, g)
energy=j11+j22

d1=np.multiply(j11,j22)
d2=np.multiply(j12,j21)
det=d1-d2

k=0.11
r=det-k*(np.multiply(energy,energy))
img=cv2.imread('PCB.jpg',0)
imgRGB=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
imgRGB[r>11000]=np.array([255,0,0])

plt.figure()

plt.imshow(imgRGB)
plt.show()
