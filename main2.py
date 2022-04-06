## box filter
#reference https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
import cv2
import numpy as np
from matplotlib import pyplot as plt

SP=cv2.imread("S&P.jpg")
GaussianNoise=cv2.imread("GaussianNoise.jpg")
ISO=cv2.imread("DSCN0482-001.jpg")
def box(image,x):
    kernel = np.ones((x,x),np.float32)/(x**2)
    dst = cv2.filter2D(image,-1,kernel)
    plt.subplot(121), plt.imshow(image), plt.title('noise')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    return dst
box(GaussianNoise,7)
plt.show()
box(GaussianNoise,3)
plt.show()
box(SP,7)
plt.show()
box(SP,3)
plt.show()
box(ISO,3)
plt.show()
box(ISO,7)
plt.show()
cv2.imwrite('boxGaussian7x7.jpg',
           box(GaussianNoise,x=7))
cv2.imwrite('boxGaussian3x3.jpg',
           box(GaussianNoise,x=3))
cv2.imwrite('boxSP7x7.jpg',
           box(SP,x=7))
cv2.imwrite('boxSP3x3.jpg',
           box(SP,x=3))
cv2.imwrite('boxISO3x3.jpg',
           box(ISO,x=3))
cv2.imwrite('boxISO7x7.jpg',
           box(ISO,x=7))