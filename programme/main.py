from myUtils import *
import numpy as np
from matplotlib import pyplot as plt
from debug import *


def activity_one(imgList):
    """
    IMPORTS:
    EXPORTS:
    PURPOSE:

    TO DO:
        -I need to make the process of comparing the rotated images into
        its own function. So I can use the exact same function to do the
        comparisions for the scaled images
    """
    diamond_img = cv.imread(imgList[0])
    dugong_img = cv.imread(imgList[1])
    #SET UP
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    diamond_img_copy = diamond_img.copy()
    dugong_img_copy = dugong_img.copy()

    #creating a list of rotated images with angles of 15 degrees between each image
    rotated_diamonds = [rotate_image_b(diamond_img_copy, angle) for angle in range(15,360,15)]
    rotated_dugong = [rotate_image_b(dugong_img_copy, angle) for angle in range (15, 360, 15)]

    #create a list of scaled images with each image with a factos difference of 0.0416
    #between each image
    scaled_diamonds = [resize_img(diamond_img_copy, factor/24) for factor in range(12, 36, 1)]
    scaled_dugong = [resize_img(dugong_img_copy, factor/24) for factor in range(12, 36, 1)]

    #performing the harris corner detection on the original image, so we have
    #a base point for comparisions latter onwards
    og_diamond_harris = harris(diamond_img.copy(), [0,255,0])

    #creating a list of images which contains the rotate iamges with the harris
    #corner detection performed on each image
    harris_diamonds_rotated = [harris(ii) for ii in rotated_diamonds]
    #the thing which is happening above is the same thing which is happening here
    harris_diamonds_scaled = [harris(ii) for ii in scaled_diamonds]

    #taking advantage of the image, the only thing green on the image is the
    #detected points
    channel = 1
    bin_size = 16
    size = len(harris_diamonds_rotated)
    og_diamond_hist_harris = calc_histograms(og_diamond_harris, channel, bin_size)
    og_diamond_hist = calc_histograms(diamond_img.copy(), channel, bin_size)

    hists_diamonds_rotated = [calc_histograms(ii,channel, bin_size) for ii in harris_diamonds_rotated]

    #showing all the rotated histograms
    plt.plot(og_diamond_hist, color=map_colors(1), label='original image')
    plt.plot(og_diamond_hist_harris, color=map_colors(2), label='harris orignal image')

    for ii, hist in enumerate(hists_diamonds_rotated):
        #need to offset color by 2 as the first two colors were used by the first
        #two images
        color = ii + 2
        plt.plot(hist, color=map_colors(color), label='harris roated: %s' % ii, linestyle='--')

    plt.xlim([0,bin_size])
    plt.legend(loc='upper center')
    plt.ylabel('Frequency')
    plt.xlabel('intensity value')
    plt.title('Rotated Diamonds harris comparison')
    plt.show()

    distance = calc_hist_dist(og_diamond_hist, hists_diamonds_rotated)
    #getting the distances of the rotated image relative to the orginal image
    #labels = ('distnance| 1 : 2', 'distance | 1: 3', 'distance | 1: 4')
    labels = ['img: %s' % ii for ii in range(len(distance))]
    labels = tuple(labels)
    y_pos = np.arange(len(labels))
    #distances = [0, 20, 30]
    plt.bar(y_pos, distance, align='center', alpha=0.15)
    plt.xticks(y_pos, labels)
    plt.ylabel('Distance from orginal Harris image')
    plt.xlabel('Distances (units)')
    plt.title('Comparing histogram distances from the orginal histogram')

    plt.show()

if __name__ == '__main__':
    imgList = ['imgs/diamond2.png', 'imgs/Dugong.jpg']
    activity_one(imgList)
    cv.waitKey(0)
