from cv2 import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

jetplane=cv2.imread("jetplane.tif")
jetplane=cv2.resize(jetplane,(800,600))
s=jetplane.shape
jetplaneGray=cv2.cvtColor(jetplane,cv2.COLOR_BGR2GRAY)
cv2.imshow('binary',jetplaneGray)


def Hist(image):
    H=np.zeros(shape=(256,1))
    s=image.shape
    for i in range(s[0]):
        for j in range(s[1]):
            k=image[i,j]
            H[k,0]=H[k,0]+1
    return H

histg=Hist(jetplaneGray)
plt.plot(histg)
def hist_eq(img,bins):
    x=histg.reshape(1,256)
    y=np.array([])
    y=np.append(y,x[0,0])

    for i in range(255):
        k=x[0,i+1]+y[i]
        y=np.append(y,k)
    y=np.round((y/(s[0]*s[1]))*(bins-1))

    for i in range(s[0]):
        for j in range(s[1]):
            k=img[i,j]
            img[i,j]=y[k]


hist_eq(jetplaneGray,64)
equal=Hist(jetplaneGray)
plt.plot(histg)
plt.figure()
plt.plot(equal)
plt.show()
cv2.imshow("equalized",jetplaneGray)
cv2.waitKey(0)

## reference https://youtu.be/cVg2WiAX8Lg


