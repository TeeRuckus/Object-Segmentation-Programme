import cv2 as cv
import numpy as np
from myUtils import *
import csv
import argparse
import random as rng
import matplotlib.pyplot as plt
rng.seed(12345)

para_one = 'tawana'
para_two = 'kwararamba'
list_one = [ii+1 for ii in range(10)]
list_two = [ii + 2 for ii in range(10)]
list_three = [ii * 2 for ii in range(10)]

def plot_something():
    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')
    plt.show()


plot_something()

def some_func(para_one, para_two, *args):
    print("%s" % para_one )
    print("%s" % para_two)
    print("%s and type: %s" % (args[0], type(args[0])))
    print("%s" % args[1])
    print("%s" % args[2])

def return_two(thing_one, thing_two):
    return thing_one, thing_two


def write_file():
    headers = ['some 1', 'some 2', 'some 3']
    row_one = [1, 'yes', 'no']
    row_two = [2, 'no', 'no']
    data_one = ['age']
    date_two = ['20']

    all_data  = [row_one, row_two]
    print(all_data)

    with open('test_file.csv', 'wt') as f:
        csv_writer = csv.writer(f)

        csv_writer.writerow(data_one + date_two)
        #writing the headers down
        csv_writer.writerow(headers)
        csv_writer.writerow(row_one)
        csv_writer.writerow(row_two)

def write_file_two():
    headers = ['some 1', 'some 2', 'some 3']
    labels = ['image one', 'image two', 'image three']
    differences = [0, 0, 0]
    raw_distances = [200,5000, 600]

    with open('test_file.csv', 'wt') as f:
        all_data = zip(labels, raw_distances, differences)
        csv_writer = csv.writer(f)

        csv_writer.writerow(headers)
        for ii in all_data:
            #writing the headers down
            csv_writer.writerow(ii)

def playing_tuples():
    im = cv.imread('imgs/diamond2.png')
    print(im.shape[:2])

def image_properties():
    white_img = np.array([[255, 255, 255],
                          [255, 255, 255],
                          [255, 255, 255],
                          [255, 255, 255]])

    padding = np.zeros(white_img.shape)
    padding[:white_img.shape[0], :white_img.shape[1]] = white_img

    print('white image \n {}'.format(white_img))
    print('padding image \n {}'.format(padding))

    padded = np.pad(white_img, [2, 1], mode='constant')
    print('padded \n{}'.format(padded))

def finding_contours_play(im):
    im = cv.imread(im)
    im_copy = im.copy()


    im_gray = cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)
    cv.imshow('gray-scale image', im_gray)

    cv.waitKey()


def play_flatten():
    some_array = np.arange(100)
    nw_array = some_array.reshape((10,10))
    print(nw_array)
    print(nw_array.flatten())

def play_v_stack():
    arr_one = [1,2,3,4]
    arr_two = [5,6,7,8]
    arr_three = [9,10,11,12]

    res = np.vstack((arr_one, arr_two, arr_three))
    print(res)

#play_v_stack()
#def thresh_callback(val):
#    threshold = val
#
#    #detecting the edges in the image using canny
#
#    canny_output = cv.Canny(src_gray, threshold, threshold*2)
#    #finding the countours of the image
#    _, contours, heierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#    #drawing the contours onto the image
#    drawing  = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
#
#    for ii in range(len(contours)):
#        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
#        cv.drawContours(drawing, contours, ii, color, 2, cv.LINE_8, heierarchy,0)
#
#    cv.imshow('Contours found', drawing)
#
#src = cv.imread('imgs/diamond2.png')
#
##converting the image to a grayscale image, and blurring it
#src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#src_gray = cv.blur(src_gray, (3,3))
#
#
##creating my window
#source_window = 'Source'
#cv.namedWindow(source_window)
#cv.imshow(source_window, src)
#
#max_thresh = 255
#thresh = 100 #setting the initila threshold of the image
#
#cv.createTrackbar('canny thresh:', source_window, thresh, max_thresh, thresh_callback)
#thresh_callback(thresh)
#
#cv.waitKey()
