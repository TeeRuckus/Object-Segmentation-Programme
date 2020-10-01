import cv2 as cv
import numpy as np
from myUtils import *
import csv
import argparse

para_one = 'tawana'
para_two = 'kwararamba'
list_one = [ii+1 for ii in range(10)]
list_two = [ii + 2 for ii in range(10)]
list_three = [ii * 2 for ii in range(10)]

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

if __name__ == '__main__':
    #write_file()
    #image_properties()
    #playing_tuples()
#    ap = argparse.ArgumentParser()
#    ap.add_argument("-i", "--images", required=True, help='path to imageas directory')
#    args = vars(ap.parse_args())
    #I wonder, what the hell this does aye

    #write_file_two()
#some_func(para_one, para_two, list_one, list_two, list_three)
    #I am curious if you can double format a string

    name = 'Tawana'
    print('hello {}, my name is also {}'.format(name, name))
    print('hello %s my name is also %s ' % (name, name))
