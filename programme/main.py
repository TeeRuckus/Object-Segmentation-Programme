from myUtils import *
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from debug import *
import random as rng
import argparse
#any matplot lib figures which are opened, will be opened in full size

#intializing the randon number generator, and initialising the beginning number
rng.seed(1000)

#forcing arrays to print out in full, and not to be truncated
np.set_printoptions(threshold=np.inf)
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
    -For task 3, when you print out each of the components to the terminal, also number which object is which number
"""

def activity_one_harris_rotated(im, channel, **kwargs):
    """
    IMPORTS:
        - im (String): the name of the imag
        - channel (integer): the channel were you wan to calculate your histograms on
        - kwargs (directionary mapping): importing settings for the harris corner detector
        regardless of the order of the impor. These settings inlcude the following
            - threshold
            - color of the harris corner detector corners in the resultant image

    EXPORTS: None
    PURPOSE: to perform the experiemnts relative to task one of the machine
    percpetion assignment one. Hence, to see if the harris corner detection is
    actually rotation invariant, and to see if the SIFT is scale and rotation
    invariant
    """
    print('-'*20 + 'PERFORMING ROTATIONAL HARRIS EXPERIEMENTS: %s' % kwargs['name'] + '-'*20)
    img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    img_copy = img.copy()

    #the path where all the results of this experiment is kept
    path = 'results/Task_1/rotated_experiements/Harris/%s/' % kwargs['name']

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
    fileName = path + 'comparison.csv'
    diff_frm_og = get_diff_pixels(num_kp_og_rotated[0],num_kp_rotated)
    labels = generate_labels(len(num_kp_rotated))
    save_comparisons(labels, num_kp_rotated,diff_frm_og, fileName)
    exp_one = show_diff_dist(diff_frm_og, title='Experiment One: Difference between key points: %s' % kwargs['name'])

    open_file(fileName)
    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #change in the count of green pixels found in the image
    exp_two = show_diff_hists(og_hist, og_hist_harris, hists_s_rotated, bin_size,
            name='Experiment two: comparing histograms: %s' % kwargs['name'])

    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist, hists_s_rotated)
    exp_three = show_diff_dist(distance, title='Experiment three: Differences between histograms %s'
            % kwargs['name'])

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    harris_s_rotated.insert(0, og_harris)
    show_img_ls(harris_s_rotated, path)

    #showing all the produced plots at once
    exp_one.show()
    exp_two.show()
    exp_three.show()
    cv.waitKey()
    clean()

def activity_one_harris_scaled(im, channel, **kwargs):
    """
    IMPORTS:
    EXPORTS:
    PURPOSE:
    """
    print('-'*20 + 'PERFORMING SCALING HARRIS EXPERIEMENTS: %s' % kwargs['name'] + '-'*20)
    path = 'results/Task_1/scaled_experiements/Harris/%s/' % kwargs['name']
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
    fileName = path + 'comparison.csv'
    diff_frm_og = get_diff_pixels(num_kp_og_scaled[0],num_kp_scaled)
    labels = generate_labels(len(num_kp_scaled))
    save_comparisons(labels,  num_kp_scaled, diff_frm_og, fileName)
    exp_one = show_diff_dist(diff_frm_og, title='Difference between key points')
    open_file(fileName)

    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #change in the count of green pixels found in the image
    exp_two = show_diff_hists(og_hist, og_hist_harris, hists_s_scaled, bin_size,
            name='Experiment two: comparing histograms: %s' % kwargs['name'])


    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist, hists_s_scaled)
    exp_three = show_diff_dist(distance, title='Diffetences between histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    scaled_s.insert(0, og_harris)
    show_img_ls(scaled_s, path)

    exp_one.show()
    exp_two.show()
    exp_three.show()
    cv.waitKey()
    clean()

def activity_one_SIFT_rotated(im, **kwargs):
    """
    IMPORT:
    EXPORT:

    Purpose:
    """
    print('-'*20 + 'PERFORMING ROTATED SIFT EXPERIEMENTS: %s' % kwargs['name'] + '-'*20)
    path = 'results/Task_1/rotated_experiements/SIFT/%s/' % kwargs['name']
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
    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins

    #EXPERIMENT ONE: testing the number of key features extracted
    #finding the number of keypoints found, since keypoints are classes, were're
    #just going to check the length of keypoints returned by each list
    kp_len_og = len(og_SIFT[1])
    kp_lens = [len(ii) for ii in rotated_SIFT_KP]
    diff_frm_og = get_diff_pixels(kp_len_og, kp_lens)
    labels = generate_labels(len(kp_lens))
    #you need to re-factor this so it works for here
    #save_comparisons(labels, num_kp_rotated,diff_frm_og, fileName)
    exp_one = show_diff_dist(diff_frm_og, title='the difference of keypoints found relative to first image')
    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #to see how the intensities of the descriptors are changing throughout the experiment
    #placing a zero in the first parameter, as we don't have a base image, as
    #we're going to be caclulating the histograms of the descriptors
    exp_two = show_diff_hists(0, og_hist_des, rotated_hist_des, bin_size,
            name='Showing the difference between the obtained descriptors')

    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist_des, rotated_hist_des)
    exp_three = show_diff_dist(distance, title='Differences between the histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    rotated_SIFT_imgs.insert(0, og_SIFT[2])
    show_img_ls(rotated_SIFT_imgs, path)

    exp_one.show()
    exp_two.show()
    exp_three.show()

    cv.waitKey()
    clean()

def activity_one_SIFT_scaled(im, **kwargs):
    """
    IMPORTS:
    EXPORTS:
    PURPOSE:
    """
    print('-'*20 + 'PERFORMING SCALING SIFT EXPERIEMENTS: %s' % kwargs['name'] + '-'*20)
    img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    img_copy = img.copy()
    og_SIFT = SIFT(img_copy)

    path = 'results/Task_1/scaled_experiements/SIFT/%s/' % kwargs['name']
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
    exp_one = show_diff_dist(diff_frm_og, title='the difference of keypoints found relative to first image')

    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #to see how the intensities of the descriptors are changing throughout the experiment
    #placing a zero in the first parameter, as we don't have a base image, as
    #we're going to be caclulating the histograms of the descriptors
    exp_two = show_diff_hists(0, og_hist_des, scaled_des_hists, bin_size,
            name='Showing the difference on the obtained descriptors')

    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist_des, scaled_des_hists)
    exp_three = show_diff_dist(distance, title='Differences between the histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    scaled_SIFT_imgs.insert(0, og_SIFT[2])
    show_img_ls(scaled_SIFT_imgs, path)

    exp_one.show()
    exp_two.show()
    exp_three.show()
    cv.waitKey()
    clean()

def activity_two_SIFT_rotated(im, pt1, pt2, **kwargs):
    im = cv.imread(im)
    im_copy = im.copy()

    feature  = crop_img(im_copy, pt1, pt2)
    cv.imshow('original image', im_copy)
    rotated_features = [rotate_image(feature, ii) for ii in range(15,360,15)]
    og_sift  = SIFT(im_copy)[2]
    sift_des = [SIFT(ii)[2] for ii in rotated_features]

    print(og_sift.shape)
    check_sizes(sift_des)

    sift_des.insert(0, og_sift)
    #comp = [cv.norm(og_sift - ii) for ii in sift_des]

    show_img_ls(sift_des,"results/Task_2/%s/rotation/" %kwargs['name'])
    cv.waitKey()
    clean()

def activity_two_SIFT_scaled(im, pt1, pt2, **kwargs):
    im = cv.imread(im)
    im_copy = im.copy()

    scaled_features = [resize_img(im_copy, factor/24 ) for factor in range(12,36,1)]
    cropped_features = [crop_img(ii, pt1, pt2) for ii in scaled_features]

    feature = crop_img(im, pt1, pt2)

    og_sift = SIFT(im_copy)[2]
    sift_des = [SIFT(ii)[2] for ii in cropped_features]

    sift_des.insert(0, og_sift)
    show_img_ls(sift_des, "results/Task_2/%s/scaling/" %kwargs['name'])
    #comp = [cv.norm(og_sift - ii) for ii in sift_des]
    #exp_one = show_diff_dist(comp, title='the difference between sift scaled descrptors')

    cv.waitKey()
    clean()

def activity_two_hog_scaled(im, pt1, pt2):
    im = cv.imread(im)
    im_copy = im.copy()

    scaled_features = [resize_img(im_copy, factor/24) for factor in range(12,36,1)]
    cropped_features = [crop_img(ii, pt1, pt2) for ii in scaled_features]

    feature = crop_img(im_copy, pt1, pt2)
    cv.imshow('feature', feature)

    hog = cv.HOGDescriptor()
    des_og = hog.compute(feature)

    h_ls = [hog.compute(ii) for ii in cropped_features[7:]]
    comp = [cv.norm(des_og - ii) for ii in h_ls]

    exp_one = show_diff_dist(comp, title='Showing the difference between orignal descriptors and scaled descriptors')

    exp_one.show()
    cv.waitKey()
    clean()

def activity_two_hog_rotated(im, pt1, pt2):
    im = cv.imread(im)
    im_copy = im.copy()

    cv.imshow('original', im_copy)

    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------

    #I have choosen the two as the intresting keypoint hence, extracting the two
    #pre-processing: the image must have a ratio of 1:2 for the hog to work
    #properly

    #need to pad the image, so we can extract the two by itself without the
    #diamond, and to maintain a ratio of 1:2
    feature = crop_img(im_copy, pt1, pt2)
    print(feature.shape)
    cv.imshow('feature', feature)

    rotated_features = [rotate_image(feature, ii) for ii in range(15,360,15)]

    hog = cv.HOGDescriptor()
    des_og = hog.compute(feature.copy())
    h_ls = [hog.compute(ii) for ii in rotated_features]
    comp = [cv.norm(des_og - ii) for ii in h_ls]
    exp_one = show_diff_dist(comp, title='showing the difference between orignal hog descriptor and rotated hog descriptor')

    exp_one.show()
    cv.waitKey()
    clean()


def display_kp_ls(in_ls):
    for ii, num_kp in  enumerate(in_ls):
        print('image: {}, {} key points found'.format(ii,num_kp))


def activity_three(im, invert_threshold=False, **kwargs):
    """
    Adapted from: #https://iq.opengenus.org/connected-component-labeling/#:~:text=Connected%20Component%20Labeling%20can%20be,connectedComponents()%20function%20in%20OpenCV.&text=The%20function%20is%20defined%20so,path%20to%20the%20original%20image.
    """
    name = kwargs['im_name']
    path ='results/Task_3/%s/' % name.lower()
    imgs = []
    #task i
    im = cv.imread(im)
    im_copy = im.copy()
    gray_img = cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)

    #PRE-PROCESSING
    cv.imshow('original gray image', gray_img)
    #blurring the image to remove any potential noise
    blur = cv.GaussianBlur(gray_img, (5,5),0)
    cv.imshow('blurred image', blur)
    thresh = cv.threshold(blur, 0, 255,  cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    if invert_threshold:
        thresh = thresh.max() - thresh

    #cv.imshow('threshold', thresh)
    imgs.append(thresh)

    #applyting the connected component labelling algorithm
    connectivity=8
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity,cv.CV_32S)

    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    #set bg label to black
    labeled_img[label_hue==0] = 0

    #drawing the contours onto the image
    contours  = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(im_copy, contours[1], -1, (0,255,0), 3)

    #cv.imshow('after component labelling: %s' % kwargs['im_name'], labeled_img)
    imgs.append(labeled_img)

    #task ii)
    #fileName ='results/Task_3/%s/results_for_%s.csv' % (name.lower(), name)
    fileName = path + 'resutls_for_%s.csv' % name
    area_of_all_labels = [stats[ii][cv.CC_STAT_AREA] for ii in range(num_labels)]
    save_stats(fileName, area_of_all_labels, labels.max())
    open_file(fileName)

    show_img_ls(imgs, path)
    cv.waitKey()
    clean()

    return labels, area_of_all_labels, centroids

def activity_four_contours(im, thresh, **kwargs):
    """
    CODE ADAPTED FROM: https://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html
    """
    imgs = []
    path = 'results/Task_4/%s/contours/' % (kwargs['name'].lower().strip())
    im = cv.imread(im)
    im_copy = im.copy()
    im_gray =  cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)

    canny_trans = cv.Canny(im_gray, thresh, thresh * 2)

    contours, hierarchy = cv.findContours(canny_trans, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1:]

    #drawing the found contours onto the image

    #setting up a black canvas, the size of the image. To draw the picture onto
    drawing_canvas = np.zeros((canny_trans.shape[0], canny_trans.shape[1], 3), dtype=np.uint8)


    for ii in range(len(contours)):
        #randint is exlusive hence, it's actually doing numbers from 0 - 255
        color  = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing_canvas, contours, ii, color, 1, cv.LINE_AA, hierarchy,0)

    imgs.append(drawing_canvas)
    show_img_ls(imgs, path)
    cv.waitKey()
    clean()
    #cv.watershed(im_res, markers)


#apparently the HSV color scheme is better for image detection
def activity_four_kMeans(raw_im, im, **kwargs):
    imgs = []
    path = 'results/Task_4/%s/%s/' % (kwargs['name'].lower().strip(), kwargs['color_space'].upper().strip())

    im_flat = im.reshape((-1,3))
    im_flat = np.float32(im_flat)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, kwargs['num_iter'] , kwargs['eplison'])

    K = kwargs['K']
    attempts = kwargs['num_iter']

    #flags  = cv.KMEANS_RANDOM_CENTERS
    flags = cv.KMEANS_PP_CENTERS
    labels, (centers) = cv.kmeans(im_flat, K, None, criteria, attempts, flags)[1:]

    centers = np.uint8(centers)

    labels = labels.flatten()

    #putting together the segmented image
    seg_im = centers[labels.flatten()]

    #converting it back to the original image shape
    seg_im = seg_im.reshape(im.shape)


    #cv.imshow('the segmented image', seg_im)
    imgs.append(seg_im)

    #showing each of the segments individually

    #for ii in range(labels.max()):
    for ii in  range(3):
        #making a copy, so the mask is only applied once onto the image
        im_mask = raw_im.copy()
        im_mask = im_mask.reshape((-1,3))
        #setting any segment in the image which corresponds to that cluster to blue
        im_mask[labels == ii] = [255,0,0]
        #converting the flattened pixle matrices into the original image
        im_mask = im_mask.reshape(im.shape)

        #cv.imshow('cluster %s: %s' % (ii, kwargs['color_space']), im_mask)
        imgs.append(im_mask)

    #a quick way for me to save the images
    show_img_ls(imgs, path)
    cv.waitKey()
    clean()

#I AM GOING TO COME BACK TO THIS, IT'S CAUSING ME A HEADACHE
def activity_four_watershed(im, invert_threshold=False):
    """
    CODE ADAPTED FROM: https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    """
    im = cv.imread(im)
    im_copy = im.copy()
    im_gray = cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)

    #Part one - Preprocessing the removal of noise from the image
    blur = cv.GaussianBlur(im_gray, (5,5), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    if invert_threshold:
        thresh = thresh.max() - thresh

    cv.imshow('threshold image %s' % invert_threshold, thresh)

    #noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(thresh , cv.MORPH_OPEN, kernel, iterations = 2)

    #sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    #finding sure foreground area
    dist = cv.distanceTransform(opening, cv.DIST_L2,5)
    sure_fg = cv.threshold(dist, 0.7 * dist.max(), 255,0)[1]

    #finding the unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    #marker labelling
    markers  = cv.connectedComponents(sure_fg)[1]

    #adding 1 to labels to ensure that background is not 0 but 1
    markers += 1

    #marking the unkown region with zero
    markers[unknown==255] = 0

    markers = cv.watershed(im_copy, markers)
    im_copy[markers == - 1] = [255,0,0]

    cv.imshow('resultant', im_copy)
    cv.waitKey()
    clean()

if __name__ == '__main__':
    imList = ['imgs/diamond2.png', 'imgs/Dugong.jpg']

    parser = argparse.ArgumentParser(description='''A tool which will run
            experiments on some pre-defined images relative to the machine
            perception assignment one''')

    parser.add_argument('task_num', metavar='Task number:', type=int,
    help='please select which task to run experiments for')

    parser.add_argument('im', metavar='Image', type=str,
            help='Please enter [Di]amond [Du]gong to run experiments on either\
            the diamond image or the dugong image')


    args = parser.parse_args()

    task_num = args.task_num
    image = args.im.upper().strip()
    #---------------------------------------------------------------------------
    #TASK ONE: Diamond playing card
    #---------------------------------------------------------------------------

    #running all the experiements for the diamond card
    if task_num == 1 and image == 'DI':
        activity_one_harris_rotated(imList[0], 1, color=[0,255,0], thresh=0.04, name='diamond')
        activity_one_harris_scaled(imList[0], 1, color=[0,255,0], thresh=0.04, name='diamond')
        activity_one_SIFT_rotated(imList[0], name='diamond')
        activity_one_SIFT_scaled(imList[0], name='diamond')

    #---------------------------------------------------------------------------
    #TASK ONE: Dugong image
    #---------------------------------------------------------------------------
    if task_num == 1 and image == 'DU':
        activity_one_harris_rotated(imList[1], 2, color=[0,0,255], thresh=0.06, name='dugong')
        activity_one_harris_scaled(imList[1], 2, color=[0,0,255], thresh=0.06, name='dugong')
        activity_one_SIFT_rotated(imList[1], name='dugong')
        activity_one_SIFT_scaled(imList[1], name='dugong')

    #---------------------------------------------------------------------------
    #TASK TWO: Diamond
    #---------------------------------------------------------------------------
    if task_num == 2 and image == 'DI':
        activity_two_hog_rotated(imList[0], (8,1), (69,128))
        activity_two_hog_scaled(imList[0], (8,1), (72, 129))
        activity_two_SIFT_rotated(imList[0], (8,1), (69, 128), name='diamond')
        activity_two_SIFT_scaled(imList[0], (8,1), (69,128), name='diamond')


    if task_num == 2 and image == 'DU':
        activity_two_hog_rotated(imList[1], (393,237), (457, 365))
        activity_two_hog_scaled(imList[1], (393,237), (457,365))
        activity_two_SIFT_rotated(imList[1], (392, 237), (457, 365), name='dugong')
        activity_two_SIFT_scaled(imList[1], (392,237), (457,365), name='dugong')


    #---------------------------------------------------------------------------
    #TASK Three:
    #---------------------------------------------------------------------------
    if task_num == 3 and image == 'DI':
        labels, areas, centroids = activity_three(imList[0],True, im_name='Diamond')

    if task_num == 3 and image == 'DU':
        activity_three(imList[1], im_name='dugong')

    #---------------------------------------------------------------------------
    #TASK Four: Diamond
    #---------------------------------------------------------------------------
    if task_num == 4 and image == 'DI':
        #activity_four(imList[0], 200)
        im_og = cv.imread(imList[0])
        im = cv.cvtColor(im_og.copy(), cv.COLOR_BGR2HSV)
        activity_four_kMeans(im_og.copy(), im, color_space='HSV',name='diamond',eplison=0.01, K=14, num_iter=200)
        im = cv.cvtColor(im_og.copy(), cv.COLOR_BGR2Luv)
        activity_four_kMeans(im_og.copy(), im, eplison=0.01, K=14, num_iter=200, name='diamond', color_space = 'LUV')
        im = cv.cvtColor(im_og.copy(), cv.COLOR_BGR2Lab)
        activity_four_kMeans(im_og.copy(), im, eplison=0.01, K=14, color_space='LAB',name='diamond', num_iter=200)
        activity_four_kMeans(im_og.copy(), im_og.copy(), eplison=0.01,name='diamond', K=14, color_space='BGR', num_iter=200)

        #CONTOURS FOR ACTIVITY FOUR
        activity_four_contours(imList[0], 0.1, name='diamond')

        #APPLYING THE WATERSHED
        activity_four_watershed(imList[0], True)

    #---------------------------------------------------------------------------
    #TASK Four: Dugong
    #---------------------------------------------------------------------------
    if task_num == 4 and image == 'DU':
        #activity_four(imList[1], 100)
        im_og = cv.imread(imList[1])
        im = cv.cvtColor(im_og.copy(), cv.COLOR_BGR2HSV)
        activity_four_kMeans(im_og.copy(), im, color_space='HSV',eplison=0.01, K=14,name='dugong', num_iter=100)
        im = cv.cvtColor(im_og.copy(), cv.COLOR_BGR2Luv)
        activity_four_kMeans(im_og.copy(), im, eplison=0.01, K=14, num_iter=100, color_space = 'LUV', name='dugong')
        im = cv.cvtColor(im_og.copy(), cv.COLOR_BGR2Lab)
        activity_four_kMeans(im_og.copy(), im, eplison=0.01, K=14, color_space='LAB', num_iter=100, name='dugong')
        activity_four_kMeans(im_og.copy(), im_og.copy(), eplison=0.01, K=14, color_space='BGR', num_iter=100, name='dugong')

        #CONTOURS FOR ACTIVITY FOUR
        activity_four_contours(imList[1], 80, name='dugong')

        activity_four_watershed(imList[1], True)


