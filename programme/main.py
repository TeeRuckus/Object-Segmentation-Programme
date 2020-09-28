from myUtils import *
import numpy as np
from matplotlib import pyplot as plt
from debug import *
"""
TO DO:
    - refactor the diamonds rotated, and the scaled ones, so you can pass
    in whatever image, and you can do the exact same experiement with the
    dugong, and you can just change a couple of parameters
    -you need to add an extra parameter, so you can either choose where to save
    the file too, either the playing card or the dugong
    - do the harris corner detection with the dugong aswell
    - for your sift experiments, you can get the number og key-points they're
    by just getting the length of the list the key-points returned too, and
    comparing if each transform got the same number of keypoints
"""

def activity_one_harris_rotated(im, channel, **kwargs):
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
    og_harris = harris(img.copy(),kwargs['thresh'], kwargs['color'])[0]

    #creating a list of images which contains the rotate iamges with the harris
    #corner detection performed on each image
    harris_s_rotated = [harris(ii, kwargs['thresh'], kwargs['color'])[0] for ii in rotated_s]

    bin_size = 16
    #the histogram of the very orginal image is needed, to confirm that the
    #harris corner detection which introduce variance in the produced histograms
    og_hist = calc_histograms(img.copy(), channel, bin_size)
    og_hist_harris = calc_histograms(og_harris, channel, bin_size)
    hists_s_rotated = [calc_histograms(ii,channel, bin_size) for ii in harris_s_rotated]

    #setting up the appropriate lists to do the comparisons for the keypoints
    #found in each matrix
    og_harris_kp = harris(img.copy(),kwargs['thresh'], kwargs['color'])[1]
    harris_s_rotated_kp = [harris(ii, kwargs['thresh'], kwargs['color'])[1] for ii in rotated_s]

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
    #save_comparisons(labels, num_kp_rotated,diff_frm_og, fileName)
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

def activity_one_harris_scaled(im, channel, **kwargs):
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
    og_harris = harris(img.copy(),kwargs['thresh'], kwargs['color'])[0]

    #create a list of scaled images with each image with a factos difference of 0.0416
    #between each image
    scaled_s = [resize_img(img_copy, factor/24) for factor in range(12, 36, 1)]

    bin_size = 16
    #the histogram of the very orginal image is needed, to confirm that the
    #harris corner detection which introduce variance in the produced histograms
    og_hist = calc_histograms(img.copy(), channel, bin_size)
    og_hist_harris = calc_histograms(og_harris, channel, bin_size)
    hists_s_scaled = [calc_histograms(ii,channel, bin_size) for ii in scaled_s]

    #setting up the appropriate lists to do the comparisons for the keypoints
    #found in each matrix
    og_harris_kp = harris(img.copy(),kwargs['thresh'],kwargs['color'])[1]
    harris_s_scaled_kp = [harris(ii,kwargs['thresh'], kwargs['color'])[1] for ii in scaled_s]

    num_kp_og_scaled = count_pixels([og_harris_kp])
    num_kp_scaled = count_pixels(harris_s_scaled_kp)

    #---------------------------------------------------------------------------
    #Experiments for scaled  images
    #---------------------------------------------------------------------------

    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins
    #fileName = 'results/Task_1/scaled_experiements/Harris/playing_card/comparison.csv'
    diff_frm_og = get_diff_pixels(num_kp_og_scaled[0],num_kp_scaled)
    labels = generate_labels(len(num_kp_scaled))
    #save_comparisons(labels, num_kp_scaled,diff_frm_og, fileName)
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

def activity_one_SIFT_rotated(im):
    """
    IMPORT:
    EXPORT:

    Purpose:
    """
    img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    img_copy = img.copy()

    #so we can calculate the intensities of a black and white image
    channel = 0
    bin_size = 16
    #creating a list of rotated images with angles of 15 degrees between each image
    rotated = [rotate_image_b(img_copy, angle) for angle in range(15,360,15)]

    og_SIFT = SIFT(img_copy)
    rotated_SIFT_Des = [SIFT(ii)[0] for ii in rotated]
    rotated_SIFT_KP = [SIFT(ii)[1] for ii in rotated]
    rotated_SIFT_imgs = [SIFT(ii)[2] for ii in  rotated]

    og_hist_des = calc_histograms(og_SIFT[0], channel, bin_size)
    rotated_hist_des = [calc_histograms(ii, channel, bin_size) for ii in rotated_SIFT_Des]
    #---------------------------------------------------------------------------
    #Experiments for rotated  images
    #---------------------------------------------------------------------------
    #save_file = 'results/Task_1/rotated_experiements/SIFT/%s/%s' % (image, fileName)
    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins

    #finding the number of keypoints found, since keypoints are classes, were're
    #just going to check the length of keypoints returned by each list
    kp_len_og = len(og_SIFT[1])
    kp_lens = [len(ii) for ii in rotated_SIFT_KP]
    diff_frm_og = get_diff_pixels(kp_len_og, kp_lens)
    labels = generate_labels(len(kp_lens))
    #you need to re-factor this so it works for here
    #save_comparisons(labels, num_kp_rotated,diff_frm_og, fileName)
    show_diff_dist(diff_frm_og, title='the difference of keypoints found relative to first image')
    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #to see how the intensities of the descriptors are changing throughout the experiment
    #placing a zero in the first parameter, as we don't have a base image, as
    #we're going to be caclulating the histograms of the descriptors
    show_diff_hists(0, og_hist_des, rotated_hist_des, bin_size)

    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist_des, rotated_hist_des)
    show_diff_dist(distance, title='Differences between the histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    rotated_SIFT_imgs.insert(0, og_SIFT[2])
    show_img_ls(rotated_SIFT_imgs)

def activity_one_SIFT_scaled(im):
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
    og_SIFT = SIFT(img_copy)

    channel = 0
    bin_size = 16

    scaled = [resize_img(img_copy, factor/24) for factor in range(12, 36, 1)]

    scaled_SIFT_Des = [SIFT(ii)[0] for ii in scaled]
    scaled_kp = [SIFT(ii)[1] for ii in scaled]
    scaled_SIFT_imgs = [SIFT(ii)[2] for ii in scaled]

    og_hist_des = calc_histograms(og_SIFT[0], channel, bin_size)
    scaled_des_hists = [calc_histograms(ii, channel, bin_size) for ii in scaled_SIFT_Des]

    #---------------------------------------------------------------------------
    #Experiments for scaled  images
    #---------------------------------------------------------------------------

    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins

    kp_len_og = len(og_SIFT[1])
    kp_lens = [len(ii) for ii in scaled_kp]
    diff_frm_og = get_diff_pixels(kp_len_og, kp_lens)
    labels = generate_labels(len(kp_lens))
    show_diff_dist(diff_frm_og, title='the difference of keypoints found relative to first image')

    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #to see how the intensities of the descriptors are changing throughout the experiment
    #placing a zero in the first parameter, as we don't have a base image, as
    #we're going to be caclulating the histograms of the descriptors
    show_diff_hists(0, og_hist_des, scaled_des_hists, bin_size)

    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist_des, scaled_des_hists)
    show_diff_dist(distance, title='Differences between the histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    scaled_SIFT_imgs.insert(0, og_SIFT[2])
    show_img_ls(scaled_SIFT_imgs)


if __name__ == '__main__':
    imgList = ['imgs/diamond2.png', 'imgs/Dugong.jpg']
    #---------------------------------------------------------------------------
    #TASK ONE: Diamond playing card
    #---------------------------------------------------------------------------
    #running all the experiements for the diamond card
    #activity_one_harris_rotated(imgList[0])
    #activity_one_harris_scaled(imgList[0])
    #activity_one_SIFT_rotated(imgList[0])
    #activity_one_SIFT_scaled(imgList[0])

    #---------------------------------------------------------------------------
    #TASK ONE: Dugong image
    #---------------------------------------------------------------------------
    activity_one_harris_rotated(imgList[1], 2, color=[0,0,255], thresh=0.06)
    activity_one_harris_scaled(imgList[1], 2, color=[0,0,255], thresh=0.06)
    activity_one_SIFT_rotated(imgList[1])
    activity_one_SIFT_scaled(imgList[1])
    cv.waitKey(0)
