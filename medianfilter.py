#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG
### median filter
"""
Spyder Editor

cv2.medianBlur - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
skimage.filters.median - https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.median

See how median is much better at cleaning salt and pepper noise compared to Gaussian
"""
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import median


#Needs 8 bit, not float.
gaussian = cv2.imread("GaussianNoise.jpg", 0)
SP = cv2.imread("S&P.jpg", 0)
ISO=cv2.imread("DSCN0482-001.jpg",0)
def median(image,x):
    median_using_cv2 = cv2.medianBlur(image, x)
    return median_using_cv2

cv2.imshow("gaussian before filter", gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('median gaussian 7x7',median(gaussian,7))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('median gaussian 3x3',median(gaussian,3))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("s&p before filter", SP)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('median SP 3x3',median(SP,3))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('median SP 7x7',median(SP,7))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('median ISO 3x3',median(ISO,3))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('median ISO 7x7',median(ISO,7))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('mediaSP7x7.jpg',median(SP,7) )
cv2.imwrite('mediaSP3x3.jpg',median(SP,3) )
cv2.imwrite('mediaGaussian3x3.jpg',median(gaussian,3) )
cv2.imwrite('mediaGaussian7x7.jpg',median(gaussian,7) )
cv2.imwrite('medianISO3x3.jpg',median(ISO,3))
cv2.imwrite('medianISO7x7.jpg',median(ISO,7))

