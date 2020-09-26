from myUtils import *
import numpy as np
from matplotlib import pyplot as plt
from debug import *


def activity_one_harris_diamond_rotated(imgList):
    """
    IMPORTS:
    EXPORTS:
    PURPOSE:

    TO DO:
        - you need to be able to extract the keypoints from the harris function
        so you can compre the number of keypoints found
        -create your algorithm which counts how many corners you found in your
        image
        - have a nother algorithm which will calculate the difference of points
        found from the orignal image  to the first  image
        - then write all those data to a file, so you can include it inside your
        report later on


    NOTES:
            - you might be able to get better results by comparing the matrix
            of keypoints --> maybe this is something for you to think about
    """
    diamond_img = cv.imread(imgList[0])
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    diamond_img_copy = diamond_img.copy()

    #creating a list of rotated images with angles of 15 degrees between each image
    rotated_diamonds = [rotate_image_b(diamond_img_copy, angle) for angle in range(15,360,15)]

    #performing the harris corner detection on the original image, so we have
    #a base point for comparisions latter onwards
    green = [0,255,0]
    og_diamond_harris = harris(diamond_img.copy(), green)[0]

    #creating a list of images which contains the rotate iamges with the harris
    #corner detection performed on each image
    harris_diamonds_rotated = [harris(ii, green)[0] for ii in rotated_diamonds]

    channel = 1
    bin_size = 16
    #the histogram of the very orginal image is needed, to confirm that the
    #harris corner detection which introduce variance in the produced histograms
    og_diamond_hist = calc_histograms(diamond_img.copy(), channel, bin_size)
    og_diamond_hist_harris = calc_histograms(og_diamond_harris, channel, bin_size)
    hists_diamonds_rotated = [calc_histograms(ii,channel, bin_size) for ii in harris_diamonds_rotated]

    #setting up the appropriate lists to do the comparisons for the keypoints
    #found in each matrix
    og_diamond_harris_kp = harris(diamond_img.copy(), green)[1]
    harris_diamonds_rotated_kp = [harris(ii, green)[1] for ii in rotated_diamonds]

    num_kp_og_rotated = count_pixels([og_diamond_harris_kp])
    num_kp_rotated = count_pixels(harris_diamonds_rotated_kp)

    #---------------------------------------------------------------------------
    #Experiments for rotated diamond images
    #---------------------------------------------------------------------------
    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins
    fileName = 'results/Task_1/rotated_experiements/Harris/playing_card/comparison.csv'
    diff_frm_og = get_diff_pixels(num_kp_og_rotated[0],num_kp_rotated)
    labels = generate_labels(len(num_kp_rotated))
    save_comparisons(labels, num_kp_rotated,diff_frm_og, fileName)
    show_diff_dist(diff_frm_og, title='Difference between key points')
    #open_file(fileName)

    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #change in the count of green pixels found in the image
    show_diff_hists(og_diamond_hist, og_diamond_hist_harris, hists_diamonds_rotated, bin_size)


    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_diamond_hist, hists_diamonds_rotated)
    show_diff_dist(distance, title='Diffetences between histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    harris_diamonds_rotated.insert(0, og_diamond_harris)
    show_img_ls(harris_diamonds_rotated)

def activity_one_harris_diamond_scaled(im):
    """
    IMPORTS:
    EXPORTS:
    PURPOSE:
    """
    diamond_img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    diamond_img_copy = diamond_img.copy()
    green = [0,255,0]
    #performing the harris corner detection on the original image, so we have
    #a base point for comparisions latter onwards
    og_diamond_harris = harris(diamond_img.copy(), green)[0]

    #create a list of scaled images with each image with a factos difference of 0.0416
    #between each image
    scaled_diamonds = [resize_img(diamond_img_copy, factor/24) for factor in range(12, 36, 1)]

    channel = 1
    bin_size = 16
    #the histogram of the very orginal image is needed, to confirm that the
    #harris corner detection which introduce variance in the produced histograms
    og_diamond_hist = calc_histograms(diamond_img.copy(), channel, bin_size)
    og_diamond_hist_harris = calc_histograms(og_diamond_harris, channel, bin_size)
    hists_diamonds_scaled = [calc_histograms(ii,channel, bin_size) for ii in scaled_diamonds]

    #setting up the appropriate lists to do the comparisons for the keypoints
    #found in each matrix
    og_diamond_harris_kp = harris(diamond_img.copy(), green)[1]
    harris_diamonds_scaled_kp = [harris(ii, green)[1] for ii in scaled_diamonds]

    num_kp_og_scaled = count_pixels([og_diamond_harris_kp])
    num_kp_scaled = count_pixels(harris_diamonds_scaled_kp)

    #---------------------------------------------------------------------------
    #Experiments for scaled diamond images
    #---------------------------------------------------------------------------

    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins
    fileName = 'results/Task_1/scaled_experiements/Harris/playing_card/comparison.csv'
    diff_frm_og = get_diff_pixels(num_kp_og_scaled[0],num_kp_scaled)
    labels = generate_labels(len(num_kp_scaled))
    save_comparisons(labels, num_kp_scaled,diff_frm_og, fileName)
    show_diff_dist(diff_frm_og, title='Difference between key points')
    #open_file(fileName)

    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #change in the count of green pixels found in the image
    show_diff_hists(og_diamond_hist, og_diamond_hist_harris, hists_diamonds_scaled, bin_size)


    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_diamond_hist, hists_diamonds_scaled)
    show_diff_dist(distance, title='Diffetences between histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    scaled_diamonds.insert(0, og_diamond_harris)
    show_img_ls(scaled_diamonds)


def show_diff_hists(base_hist, op_base_hist, op_hists, xLim):
    #showing all the rotated histograms
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
    plt.title('Rotated Diamonds harris comparison')
    plt.show()

def show_diff_dist(distance, **kwargs):
    #getting the distances of the rotated image relative to the orginal image
    labels = ['img: %s' % ii for ii in range(len(distance))]
    labels = tuple(labels)
    y_pos = np.arange(len(labels))
    #distances = [0, 20, 30]
    plt.bar(y_pos, distance, align='center', alpha=0.25)
    plt.xticks(y_pos, labels)
    plt.ylabel('Distance from orginal Harris image')
    plt.xlabel('Distances (units)')
    plt.title(kwargs['title'])

    plt.show()

if __name__ == '__main__':
    imgList = ['imgs/diamond2.png', 'imgs/Dugong.jpg']
    #activity_one_harris_diamond_rotated(imgList)
    activity_one_harris_diamond_scaled(imgList[0])
    cv.waitKey(0)
