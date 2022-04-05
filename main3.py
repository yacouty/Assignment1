## box filter
#reference https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("added_noises.jpg")
img2=cv2.imread("DSCN0482-001.jpg",cv2.COLOR_BGR2GRAY)
def box(image):
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(image,-1,kernel)
    plt.subplot(121), plt.imshow(image), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    return dst
box(img)
plt.show()
box(img2)
plt.show()
cv2.imwrite('boxfilter_ISO.jpg',
           box(img2))

cv2.imwrite('boxfilter_addedNoise.jpg',
           box(img))