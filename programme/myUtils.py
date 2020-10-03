import cv2 as cv
import csv
import numpy as np
import matplotlib.pyplot as plt
from debug import *
import math
import os


TOL = 0.00001

def clean():
    valid = False
    while not valid:
        proceed = input('Press [N]ext to go onto the next experiment: ')
        if proceed.upper().strip() == 'N':
            valid = True
            cv.destroyAllWindows()
            plt.close('all')

def map_colors(number):
    return {
            0: '#ee7b06',
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

def map_hex2RBG(hex_num):
    """
    CODE ADAPTED FROM: https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    """
    hex_num = hex_num.lstrip('#')
    return tuple(int(hex_num[ii:ii+2], 16) for ii in (0,2,4))

def show_img_ls(img_ls):
    for ii, mat in enumerate(img_ls):
        cv.imshow('image: %s' % ii, mat)

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

def resize_img_dim(img, nw_width, nw_len):
    img_copy = img.copy()
    #chnage everything to the copy image
    return cv.resize(img_copy, (int(nw_width), int(nw_len)))


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
    kp, des = sift.detectAndCompute(img, None)
    #img = cv.drawKeypoints(img, kp, img,
    #        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img = cv.drawKeypoints(img, kp, img)
    return des, kp, img
#SIFT(cv.imread('imgs/diamond2.png'))

def harris(img, thresh, color=[0,0,255]):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    #0.04 - 0.06
    detected_img  = cv.cornerHarris(gray_img, 2, 3, 0.04)
    #you want the raw key points, which hasn't been manipulated in any way
    kp_mat = detected_img
    detected_img = cv.dilate(detected_img, None)
    #filltering the corners we detected by our choosen threshold
    #img[detected_img > 0.01 * detected_img.max()] = color
    img[detected_img > thresh * detected_img.max()] = color
    return img, kp_mat


def count_pixels(img_ls):
    """
    IMPORT:
    EXPORT:
    PURPOSE: to count the how many non-zero pixels exist in an image
    """
    return [np.count_nonzero(ii > 0) for ii in img_ls]

def get_diff_pixels(base_comp, comp_ls):
    return [abs(float(base_comp) - float(ii)) for ii in comp_ls]

def generate_labels(num_images):
    ret = ['experiment image: %s' % ii for ii in range(num_images)]
    ret.insert(0, 'orginal')
    return ret

def save_stats(fileName, area_label_ls, num_labels):
    headers = ['number of labels', 'label', 'Area of label (units^2)']

    with open(fileName, 'w') as inStrm:
        csv_writer = csv.writer(inStrm)
#        csv_writer.writerow([headers[0]])
#        csv_writer.writerow(str(num_labels))
        csv_writer.writerow([headers[0]] + [num_labels])
        csv_writer.writerow(headers[1:])

        for ii, area in enumerate(area_label_ls):
            csv_writer.writerow([ii] + [area])

def save_comparisons(labels, raw_pixels, diff_frm_og, fileName):
    headers = ['Image Name', 'Number of key points', 'difference between orginal keypoints and experiment']
    all_data = zip(labels, raw_pixels, diff_frm_og)
    with open(fileName, 'w') as inStrm:
        csv_writer = csv.writer(inStrm)
        csv_writer.writerow(headers)

        for ii in all_data:
            csv_writer.writerow(ii)

def open_file(fileName):
    os.system('xdg-open %s' % fileName)

def show_diff_hists(base_hist, op_base_hist, op_hists, xLim, **kwargs):
    #showing all the rotated histograms

    ret = plt.figure(kwargs['name'])
    plt.plot(base_hist, color=map_colors(1), label='original image')
    plt.plot(op_base_hist, color=map_colors(2), label='harris orignal image')

    for ii, hist in enumerate(op_hists):
        #need to offset color by 2 as the first two colors were used by the first
        #two images
        color = ii + 2
        plt.plot(hist, color=map_colors(color), label='op: %s' % ii, linestyle='--')

    plt.xlim([0,xLim])
    plt.legend(loc='upper center')
    plt.ylabel('Frequency')
    plt.xlabel('intensity value')
    plt.title(kwargs['name'])
    #plt.show()

    return ret

def show_diff_dist(distance, **kwargs):
    #getting the distances of the rotated image relative to the orginal image
    ret = plt.figure(kwargs['title'])
    labels = ['img: %s' % ii for ii in range(len(distance))]
    labels = tuple(labels)
    y_pos = np.arange(len(labels))
    #distances = [0, 20, 30]

    plt.bar(y_pos, distance, align='center', alpha=0.25)
    plt.xticks(y_pos, labels)
    plt.ylabel('Distance from orginal Harris image')
    plt.xlabel('Distances (units)')
    plt.title(kwargs['title'])

    #plt.show()
    return ret

def crop_img(img, pt1, pt2):
    x_l = int(pt1[0])
    y_l = int(pt1[1])
    x_r = int(pt2[0])
    y_r = int(pt2[1])
    return img[y_l:y_r, x_l:x_r]


def select_key_point(im, **kwargs):
    pass

def pad_image(im, row_pad, col_pad):
    npad = ((row_pad, col_pad), (row_pad, col_pad), (0,0))
    return np.pad(im, pad_width=npad, mode='constant', constant_values=0)


def hog_preprocessing(im):
    """
    IMPORT:
    EXPORT:
    PURPOSE: To rescale the image, so it will be easier, to do the later steps in
    the HOG descriptor steps like the 8x8 box, and the 16x16 box thing later in
    the tutorial
    """
    len_im = im.shape[0]
    width_im = im.shape[1]

    ratio = width_im / len_im
    if abs(ratio - 0.5)  > TOL:
        len_im = width_im * 2
        im = resize_img_dim(im, width_im, len_im)
        nw_ratio  = im.shape[1] / im.shape[0]
        #sanity check to make sure that the image, has scaled to the right
        assert abs(nw_ratio - 0.5) < TOL, 'image is not 1:2 ratio'
    return im

def hog_descriptor(im, **kwargs):
    """
    IMPORT:im (image matrice) a dictonary of all the  hog parameters
    (it doesn't matter which order
    you import them)
    EXPORT: HOG descriptor object

    ASSERTS: returns a hog descriptor object with your given imported parameters
    """

    win_size = im.shape[:2]
    print(win_size)
    #using the convention to set the block size which is typically going to be
    # 2 x cell size
    return cv.HOGDescriptor(win_size,
            kwargs['block_size'], kwargs['block_stride'],
            kwargs['cell_size'], kwargs['num_bins'],
            kwargs['deriv_aperature'], kwargs['win_sigma'],
            kwargs['hist_norm_type'], kwargs['mag_thresh'],
            kwargs['gamma'], kwargs['num_lvls'],
            kwargs['signed_grad'])

#def HOG(im):
#    """
#    IMPORT:
#    EXPORT:
#
#    PURPOSE:
#
#    ADAPTED FROM: Satya Mallick. 2016. Histogram  of oriented Gradients. Learn OpenCv.
#    https://www.learnopencv.com/histogram-of-oriented-gradients/
#
#
#    ADAPTED FROM: Satya Mallick. 2017. Handwritten Digits Classification: An
#    OpenCV (C++/Python) Tutorial. Learn OpenCv
#    .https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
#    """
#    #corresponds to step one of the hog process: pre-processing
#    im = cv.imread(im)
#    im_copy = im.copy()
#
#
#    #corresponds to step two of the hog process: calculating the gradients
#    im = np.float32(im_copy) / 255.0
#    grad_x = cv.Sobel( im, cv.CV_32F, 1, 0, ksize=1)
#    grad_y = cv.Sobel(im, cv.CV_32F, 0, 1, ksize=1)
#    mag, angle = cv.cartToPolar(grad_x, grad_y, angleInDegrees=True)
#
#    #corresping to step three: calculating HOG of even cells
#
#    #corresponding to step four: caclulating the HOG of bigger cells
#
#    #corresponding to step five: caclulating the HOG feature vector
#
#    #visualizing HOG

#helper functions for hog
