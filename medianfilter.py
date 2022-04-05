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
img_gaussian_noise = cv2.imread("added_noises.jpg", 0)
original=cv2.imread("DSCN0479-001.jpg",0)
img = img_gaussian_noise
def median(image):
    median_using_cv2 = cv2.medianBlur(image, 3)
    return median_using_cv2



cv2.imshow("original",original)
cv2.imshow("before filter", img)
cv2.imshow('cv2median.jpg',median(img))
cv2.imwrite('cv2median.jpg',
           median(img))

cv2.waitKey(0)
cv2.destroyAllWindows()



