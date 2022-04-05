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
filterd = cv2.imread("cv2median.jpg")
boxfilter=cv2.imread("boxfilter_addedNoise.jpg")
    # convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
filterd = cv2.cvtColor(filterd, cv2.COLOR_BGR2GRAY)
boxfilter=cv2.cvtColor(boxfilter,cv2.COLOR_BGR2GRAY)

    # initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("filtered", filterd),("boxfilter", boxfilter)
    # loop over the images
for (i, (name, image)) in enumerate(images):
        # show the image
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")
    # show the figure
plt.show()
    # compare the images
compare_images(original, filterd, "Original vs. median_filtered")
compare_images(original,boxfilter , "Original vs. box_filter")
psnr(original,filterd)
psnr(original,boxfilter)
