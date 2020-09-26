import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
from debug import *
import os

def map_colors(number):
    return {
            1: '#d3d3d3',
            2: '#2f4f4f',
            3: '#556b2f',
            4: '#7f0000',
            5: '#7f0000',
            6: '#008000',
            7: '#d2691e',
            8: '#00008b',
            9: '#daa520',
            10: '#8fbc8f',
            11: '#8b008b',
            12: '#b03060',
            13:	'#ff4500',
            14: '#00ced1',
            15: '#ffff00',
            16:	'#00ff00',
            17: '#8a2be2',
            18: '#00ff7f',
            19: '#e9967a',
            20: '#dc143c',
            21: '#00bfff',
            22: '#0000ff',
            23: '#adff2f',
            24: '#ff00ff',
            25: '#1e90ff',
            26: '#f0e68c',
            27: '#dda0dd',
            28: '#90ee90',
            29: '#ff1493',
            30: '#7b68ee'}[number]

def show_img_ls(img_ls):
    for ii in range(len(img_ls)):
        cv.imshow('image: %s' % ii, img_ls[ii])

def calc_histograms(img, channel=0, hist_size=256, hist_range=(0,256)):
    return cv.calcHist([img], [channel], None, [hist_size], hist_range)

def calc_hist_dist(primary_hist, in_ls, **kwargs):
    if len(kwargs) == 0:
        method=2
        distance = [cv.compareHist(primary_hist,ii,method) for ii in in_ls]
    elif 'method' not in kwin_lis:
        raise KeyError('key word is not supported')
    else:
        distance = [cv.compareHist(primary_hist,ii,kwargs['method']) for ii in in_lis]
    return distance

def show_histograms(image):
    """
    adapted from: https://medium.com/@rndayala/image-histograms-in-opencv-40ee5969a3b7
    """
    for i, col in enumerate(['b', 'g', 'r']):
        log('i: %s and col: %s' % (i, col))
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color = col)
        plt.xlim([0, 256])

    plt.show()

def resize_img(img, reduction=0.5):
    img_copy = img.copy()
    #chnage everything to the copy image
    width = img_copy.shape[1] * reduction
    height = img_copy.shape[0] * reduction
    return cv.resize(img_copy, (int(width), int(height)))


#TO DO: when you're rotating your images, it's cutting of some of the image
#hence, you need to figure out how to rotate the image and not cut out any
#of the card or the image
def rotate_image(img, angle=45):
    rows, cols, channels = img.shape
    rotated_mat = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0),
            angle,1)
    return cv.warpAffine(img,rotated_mat,(cols,rows))

def rotate_image_b(img, angle=45):
    '''
    METHOD NAME:
    IMPORTS:
    EXPORTS"

    PURPOSE:

    adopted from: #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    '''
    #grabbing the centre of the image, so we have a point to rotate the image
    (height, width) = img.shape[:2]
    (height_centre, width_centre) = (height // 2, width// 2)
    #negative angle is to get the clockwise rotation of the image
    rotated_mat = cv.getRotationMatrix2D((width_centre, height_centre), -angle,
            1.0)
    #getting the transformed sin, and cosine components of the rotated matrix
    cos = np.abs(rotated_mat[0,0])
    sin = np.abs(rotated_mat[0,1])

    #getting the new bounding dimensions of the image
    new_bounding_height = int((height * cos) + (width * sin))
    new_bounding_width = int((height * sin) + (width * cos))

    rotated_mat[0,2]  += (new_bounding_width / 2) - width_centre
    rotated_mat[1,2] += (new_bounding_height / 2) - height_centre

    #performing the actial rotation on the image, and returning the image
    return cv.warpAffine(img, rotated_mat, (new_bounding_width,
        new_bounding_height))

def SIFT(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    #kp = sift.detect(img,None)
    kp, des = sift.detectAndCompute(img, None)
    img = cv.drawKeypoints(img, kp, img,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #kp, des = sift.compute(img, kp)

    return kp, des

def harris(img, color=[0,0,255]):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    #0.04 - 0.06
    detected_img  = cv.cornerHarris(gray_img, 2, 3, 0.04)
    detected_img = cv.dilate(detected_img, None)

    #if you make 0.01 smaller, it's going to pick up more corners in the
    #blurred image but, for the other images, it pretty much has no
    #effect on the numbr of corners it picked up
    img[detected_img > 0.01 * detected_img.max()] = color
    return img
