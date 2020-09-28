from myUtils import *
import numpy as np
from matplotlib import pyplot as plt
from debug import *
"""
TO DO:
    - refactor the diamonds rotated, and the scaled ones, so you can pass
    in whatever image, and you can do the exact same experiement with the
    dugong, and you can just change a couple of parameters
    - do the harris corner detection with the dugong aswell
    - for your sift experiments, you can get the number og key-points they're
    by just getting the length of the list the key-points returned too, and
    comparing if each transform got the same number of keypoints
    -
"""

def activity_one_harris_rotated(im):
    """
    IMPORTS:
    EXPORTS:
    PURPOSE:
    """
    img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    img_copy = img.copy()

    #creating a list of rotated images with angles of 15 degrees between each image
    rotated_s = [rotate_image_b(img_copy, angle) for angle in range(15,360,15)]

    #performing the harris corner detection on the original image, so we have
    #a base point for comparisions latter onwards
    green = [0,255,0]
    og_harris = harris(img.copy(), green)[0]

    #creating a list of images which contains the rotate iamges with the harris
    #corner detection performed on each image
    harris_s_rotated = [harris(ii, green)[0] for ii in rotated_s]

    channel = 1
    bin_size = 16
    #the histogram of the very orginal image is needed, to confirm that the
    #harris corner detection which introduce variance in the produced histograms
    og_hist = calc_histograms(img.copy(), channel, bin_size)
    og_hist_harris = calc_histograms(og_harris, channel, bin_size)
    hists_s_rotated = [calc_histograms(ii,channel, bin_size) for ii in harris_s_rotated]

    #setting up the appropriate lists to do the comparisons for the keypoints
    #found in each matrix
    og_harris_kp = harris(img.copy(), green)[1]
    harris_s_rotated_kp = [harris(ii, green)[1] for ii in rotated_s]

    num_kp_og_rotated = count_pixels([og_harris_kp])
    num_kp_rotated = count_pixels(harris_s_rotated_kp)

    #---------------------------------------------------------------------------
    #Experiments for rotated  images
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
    show_diff_hists(og_hist, og_hist_harris, hists_s_rotated, bin_size)


    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist, hists_s_rotated)
    show_diff_dist(distance, title='Diffetences between histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    harris_s_rotated.insert(0, og_harris)
    show_img_ls(harris_s_rotated)

def activity_one_harris_scaled(im):
    """
    IMPORTS:
    EXPORTS:
    PURPOSE:
    """
    img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    img_copy = img.copy()
    green = [0,255,0]
    #performing the harris corner detection on the original image, so we have
    #a base point for comparisions latter onwards
    og_harris = harris(img.copy(), green)[0]

    #create a list of scaled images with each image with a factos difference of 0.0416
    #between each image
    scaled_s = [resize_img(img_copy, factor/24) for factor in range(12, 36, 1)]

    channel = 1
    bin_size = 16
    #the histogram of the very orginal image is needed, to confirm that the
    #harris corner detection which introduce variance in the produced histograms
    og_hist = calc_histograms(img.copy(), channel, bin_size)
    og_hist_harris = calc_histograms(og_harris, channel, bin_size)
    hists_s_scaled = [calc_histograms(ii,channel, bin_size) for ii in scaled_s]

    #setting up the appropriate lists to do the comparisons for the keypoints
    #found in each matrix
    og_harris_kp = harris(img.copy(), green)[1]
    harris_s_scaled_kp = [harris(ii, green)[1] for ii in scaled_s]

    num_kp_og_scaled = count_pixels([og_harris_kp])
    num_kp_scaled = count_pixels(harris_s_scaled_kp)

    #---------------------------------------------------------------------------
    #Experiments for scaled  images
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
    show_diff_hists(og_hist, og_hist_harris, hists_s_scaled, bin_size)


    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist, hists_s_scaled)
    show_diff_dist(distance, title='Diffetences between histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    scaled_s.insert(0, og_harris)
    show_img_ls(scaled_s)

def activity_one_SIFT_diamond_rotated(im):
    pass


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
    activity_one_harris_rotated(imgList[0])
    activity_one_harris_scaled(imgList[0])
    cv.waitKey(0)
