import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import os

def show_img_ls(img_ls):
    for ii in range(len(img_ls)):
        cv.imshow('image: %s' % ii, img_ls[ii])

def calc_histograms(img):
    channel = 0
    hist_size = 256
    hist_range = (0, 256)

    return cv.calcHist(img, [channel], None, [hist_size], hist_range , False)

def print_histograms(img):
    plt.hist(img.ravel(), 256, [0,256])
    plt.show()

def resize_img(img, reduction=0.5):
    width = img.shape[1] * reduction
    height = img.shape[0] * reduction
    cv.resize(img, (int(width), int(height)))


#TO DO: when you're rotating your images, it's cutting of some of the image
#hence, you need to figure out how to rotate the image and not cut out any
#of the card or the image
def rotate_image(img, angle=45):
    rows, cols, channels = img.shape
    rotated_mat = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0),
            angle,1)
    return cv.warpAffine(img,rotated_mat,(cols,rows))

def SIFT(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    #kp = sift.detect(img,None)
    kp, des = sift.detectAndCompute(img, None)
    img = cv.drawKeypoints(img, kp, img,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #kp, des = sift.compute(img, kp)

    return kp, des

def harris_corner_detection(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    #0.04 - 0.06
    detected_img  = cv.cornerHarris(gray_img, 2, 3, 0.04)
    detected_img = cv.dilate(detected_img, None)

    #if you make 0.01 smaller, it's going to pick up more corners in the
    #blurred image but, for the other images, it pretty much has no
    #effect on the numbr of corners it picked up
    img[detected_img > 0.01 * detected_img.max()] = [0,0,255]

