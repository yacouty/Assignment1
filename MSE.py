# import the necessary packages
#reference: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
#reference: https://pyimagesearch.com/2014/09/15/python-compare-two-images/
import math
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return print("PSNR=",20 * math.log10(PIXEL_MAX / math.sqrt(mse)))



def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()

    # load the images -- the original, the original + contrast,
    # and the original + photoshop
original = cv2.imread("DSCN0479-001.jpg")
mediaGaussian3x3 = cv2.imread("mediaGaussian3x3.jpg")
mediaGaussian7x7=cv2.imread("mediaGaussian7x7.jpg")
boxGaussian7x7=cv2.imread("boxGaussian7x7.jpg")
boxGaussian3x3=cv2.imread("boxGaussian3x3.jpg")
mediaSP3x3=cv2.imread("mediaSP3x3.jpg")
mediaSP7x7=cv2.imread("mediaSP7x7.jpg")
boxSP3x3=cv2.imread("boxSP3x3.jpg")
boxSP7x7=cv2.imread("boxSP7x7.jpg")

    # convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
mediaGaussian3x3 = cv2.cvtColor(mediaGaussian3x3, cv2.COLOR_BGR2GRAY)
mediaGaussian7x7=cv2.cvtColor(mediaGaussian7x7,cv2.COLOR_BGR2GRAY)
boxGaussian3x3=cv2.cvtColor(boxGaussian3x3,cv2.COLOR_BGR2GRAY)
boxGaussian7x7=cv2.cvtColor(boxGaussian7x7,cv2.COLOR_BGR2GRAY)
mediaSP3x3=cv2.cvtColor(mediaSP3x3,cv2.COLOR_BGR2GRAY)
mediaSP7x7=cv2.cvtColor(mediaSP7x7,cv2.COLOR_BGR2GRAY)
boxSP3x3=cv2.cvtColor(boxSP3x3,cv2.COLOR_BGR2GRAY)
boxSP7x7=cv2.cvtColor(boxSP7x7,cv2.COLOR_BGR2GRAY)
    # initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("mediaGaussian3x3", mediaGaussian3x3),("mediaGaussian7x7", mediaGaussian7x7),("boxGaussian7x7",boxGaussian7x7),("boxGaussian3x3",boxGaussian3x3),("mediaSP3x3",mediaSP3x3),("mediaSP7x7",mediaSP7x7),("boxSP3x3",boxSP3x3),("boxSP7x7",boxSP7x7)
    # loop over the images
for (i, (name, image)) in enumerate(images):
        # show the image
    ax = fig.add_subplot(3, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")
    # show the figure
plt.show()
    # compare the images
compare_images(original, mediaGaussian3x3, "Original vs. mediaGaussian3x3")
compare_images(original,mediaGaussian7x7 , "Original vs. mediaGaussian7x7")
compare_images(original,boxGaussian3x3 , "Original vs. boxGaussian3x3")
compare_images(original,boxGaussian7x7 , "Original vs. boxGaussian7x7")

compare_images(original, mediaSP3x3, "Original vs. mediaSP3x3")
compare_images(original,mediaSP7x7 , "Original vs. mediaSP7x7")
compare_images(original,boxSP3x3 , "Original vs. boxSP3x3")
compare_images(original,boxSP7x7 , "Original vs. boxSP7x7")

print("mediaGaussian3x3")
psnr(original,mediaGaussian3x3)
print("mediaGaussian7x7")
psnr(original,mediaGaussian7x7)
print("boxGaussian3x3")
psnr(original,boxGaussian3x3)
print("boxGaussian7x7")
psnr(original,boxGaussian7x7)
print("mediaSP3x3")
psnr(original,mediaSP3x3)
print("mediaSP7x7")
psnr(original,mediaSP7x7)
print("boxSP3x3")
psnr(original,boxSP3x3)
print("boxSP7x7")
psnr(original,boxSP7x7)
