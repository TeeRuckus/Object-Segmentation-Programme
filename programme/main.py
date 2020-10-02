from myUtils import *
import numpy as np
from matplotlib import pyplot as plt
from debug import *
import random as rng

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

    #EXPERIMENT ONE: testing the number of key features extracted
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

def activity_two_hog(im, pt1, pt2):
    im = cv.imread(im)
    im_copy = im.copy()

    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------

    #I have choosen the two as the intresting keypoint hence, extracting the two
    cv.imshow('diamond', im_copy)
    #pre-processing: the image must have a ratio of 1:2 for the hog to work
    #properly

    #need to pad the image, so we can extract the two by itself without the
    #diamond, and to maintain a ratio of 1:2
    feature = crop_img(im_copy, pt1, pt2)
    feature = pad_image(feature, 4, 1)
    cv.imshow('feature', feature)
    print(feature.shape)

#    rotated_twos = [rotate_image_b(two.copy(), angle) for angle in range(15,360,15)]
#
    #focessing the images, too maintain that image ratio of 1:2
    #rotated_twos = [hog_preprocessing(ii) for ii in rotated_twos]

    #using the recommended values for hog

    og_hog = hog_descriptor(feature,
            cell_size=(5,5),
            block_size=(10,10),
            block_stride=(5,5),
            num_bins=9,
            deriv_aperature=1,
            win_sigma=-1,
            hist_norm_type=0,
            mag_thresh=0.2,
            gamma=1,
            num_lvls=64,
            signed_grad=True )

    og_des = og_hog.compute(feature)

def activity_two_rotated(im):
    im = cv.imread(im)
    im_copy = im.copy()

    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------

    #I have choosen the two as the intresting keypoint hence, extracting the two
    cv.imshow('diamond', im_copy)
    #pre-processing: the image must have a ratio of 1:2 for the hog to work
    #properly

    #need to pad the image, so we can extract the two by itself without the
    #diamond, and to maintain a ratio of 1:2
    two = crop_img(im_copy, (8, 1), (28, 46))
    two = pad_image(two, 4, 1)

    rotated_twos = [rotate_image_b(two.copy(), angle) for angle in range(15,360,15)]
    rotated_twos.insert(0, two)
    #to keep the experiements fair, we have to test the operations on the same
    #scalling

    #MAKE SURE THAT YOU'RE USING THE SAME IMAGES AS HOG
    rotated_twos = [hog_preprocessing(ii) for ii in rotated_twos]
    rotated_twos_SIFT_kp = [SIFT(ii)[1] for ii in rotated_twos]

    #EXPERIMENT ONE: finding the number of features extracted in the image
    lens_imgs = [len(ii) for ii in rotated_twos_SIFT_kp]
    show_diff_dist(lens_imgs, title='number of keypoints found for each image')

    #EXPERIMENT TWO:

def display_kp_ls(in_ls):
    for ii, num_kp in  enumerate(in_ls):
        print('image: {}, {} key points found'.format(ii,num_kp))


def activity_three(im, invert_threshold=False, **kwargs):
    """
    Adapted from: #https://iq.opengenus.org/connected-component-labeling/#:~:text=Connected%20Component%20Labeling%20can%20be,connectedComponents()%20function%20in%20OpenCV.&text=The%20function%20is%20defined%20so,path%20to%20the%20original%20image.
    """

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

    cv.imshow('threshold', thresh)

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

    cv.imshow('after component labelling: %s' % kwargs['im_name'], labeled_img)

    #task ii)
    name = kwargs['im_name']
    fileName ='results/Task_3/%s/results_for_%s.csv' % (name.lower(), name)
    area_of_all_labels = [stats[ii][cv.CC_STAT_AREA] for ii in range(num_labels)]

    save_stats(fileName, area_of_all_labels, labels.max())
    open_file(fileName)

    return labels, area_of_all_labels, centroids

def activity_four(im, thresh):
    """
    CODE ADAPTED FROM: https://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html
    """
    im = cv.imread(im)
    im_copy = im.copy()
    im_gray =  cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)

    #METHOD ONE: K-means

    #METHOD TWO: WATERSHED
    canny_trans = cv.Canny(im_gray, thresh, thresh * 2)

    contours, hierarchy = cv.findContours(canny_trans, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1:]

    #drawing the found contours onto the image

    #setting up a black canvas, the size of the image. To draw the picture onto
    drawing_canvas = np.zeros((canny_trans.shape[0], canny_trans.shape[1], 3), dtype=np.uint8)


    for ii in range(len(contours)):
        #randint is exlusive hence, it's actually doing numbers from 0 - 255
        color  = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing_canvas, contours, ii, color, 1, cv.LINE_AA, hierarchy,0)

    cv.imshow('Contours found: %s' % thresh, drawing_canvas)

    #cv.watershed(im_res, markers)


def activity_four_kmeans_area(im, total_area, in_labels, in_centroids, **kwargs):
    """
    IMPORT:
    EXPORT:

    PURPOSE:
    """
    print(total_area)
    im = cv.imread(im)
    im_copy = im.copy()
    data = total_area

    #converting the np.float32
    data = np.float32(data)

    #defining the criteria in which we're going to apply k-means tooo
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    K = 3
    flags = cv.KMEANS_PP_CENTERS
    label, center, = cv.kmeans(data, K, None, criteria, attempts, flags)[1:]


    center  = np.uint8(center)

    print(label)
    cv.imshow('woooo', im_copy)
    cv.imshow('huh', center)

#this shit doesn't work too well
def activity_four_kMeans_edges(im):
    im = cv.imread(im)
    im_copy = im.copy()
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    sobel_y = np.array([[-1, -2, -1],
                        [0,0,0],
                        [1,2,1]])

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    filtered_img_x = cv.filter2D(im_gray, -1, sobel_x)
    filtered_img_y = cv.filter2D(im_gray, -1, sobel_y)

    #getting rid of negative values out of the matrice, so it doesn't affect
    #the image colour later on
    filtered_img_x_abs = cv.convertScaleAbs(filtered_img_x)
    filtered_img_y_abs = cv.convertScaleAbs(filtered_img_y)


    combined_trans = cv.addWeighted(filtered_img_x_abs, 0.5, \
            filtered_img_y_abs, 0.5, 0)
    #im_flat = combined_trans.reshape((-1,3))
    im_flat = combined_trans
    #the k-means algorithm will accept the data type of float32 not uint8
    im_flat = np.float32(im_flat)
    #import the number of attempts, adn the eplison value for the image
    attempts = 200
    eplison = 0.1
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,attempts, 0.1)

    K = 15
    flags = cv.KMEANS_PP_CENTERS
    labels, centers = cv.kmeans(im_flat, K, None, criteria, attempts, flags)[1:]

    centers = np.uint8(centers)
    labels = labels.flatten()

    #puttin  together the segmented image
    seg_im = centers[labels.flatten()]

    #converting it back to the orginal image shape
    seg_im = seg_im.reshape(combined_trans.shape)

    cv.imshow('the segmented image', seg_im)

    #showing each of the segments individually
    for ii in range(labels.max()):
        #making a copy, so the mask is only applied once onto the image
        im_mask = combined_trans.copy()
        #im_mask = im_mask.reshape((-1,3))
        #setting any segment in the image which corresponds to that cluster to blue
        print(ii)
        im_mask[labels == ii] = 125
        #converting the flattened pixle matrices into the original image
        im_mask = im_mask.reshape(combined_trans.shape)

        cv.imshow('cluster: %s' % ii, im_mask)

    cv.imshow('sobel transform', combined_trans)


def activity_four_kMeans_corners(im):
    im = cv.imread(im)
    im_copy = im.copy()

    im_gray = cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)
    im_gray = np.float32(im_gray)

    #using the default values of 0.04 - 0.06
    detected_im = cv.cornerHarris(im_gray, 2, 3, 0.04)
    #this is a matrices of the corners found in the image, and each corner found
    #in the image is emphasised more
    detected_im = cv.dilate(detected_im, None)
    print(detected_im.shape)
    cv.imshow('detecte image', detected_im)
    detected_im = np.float32(detected_im)

    #stopping the kmeans clustering either if the algorithm reaches a specified
    #number of iterations or when the labels change less than a given epilson
    stop_runs = 200
    eplison = 0.01
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, stop_runs, eplison)

    K = 10
    attempts = 10
    #flags  = cv.KMEANS_RANDOM_CENTERS
    flags = cv.KMEANS_PP_CENTERS
    labels, (centers) = cv.kmeans(detected_im, K, None, criteria, attempts, flags)[1:]

    centers = np.uint8(centers)

    labels = labels.flatten()

    #putting together the segmented image
    seg_im = centers[labels.flatten()]

    #converting it back to the original image shape
    seg_im = seg_im.reshape(detected_im.shape)

    cv.imshow('the segmented image', seg_im)

    im[seg_im > 0.01 * seg_im.max()] = [255,0,0]
    cv.imshow('the image', im)

    for ii in range(labels.max()):
        #making a copy, so the mask is only applied once onto the image
        im_mask = detected_im.copy()
        #setting any segment in the image which corresponds to that cluster to blue
        im_mask[labels == ii] = [125]
        #converting the flattened pixle matrices into the original image
        im_mask = im_mask.reshape(detected_im.shape)

        cv.imshow('cluster: %s' % ii, im_mask)

def activity_four_kMeans_RGB(im, **kwargs):
    """
    IMPORT:
    EXPORT:

    PURPOSE:
    """
    im = cv.imread(im)
    im_copy = im.copy()


    #the k-means functions takes in as an input a 2-D matrice but our image is
    #a 3-D image hence, we need to reshape the image into 2-D

    #I found if I did the blurring on the green channel only, for the
    #dugong image. I produced better segementation results
    im_copy = cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)
    im_copy = cv.GaussianBlur(im_copy, (5,5), 0)

    im_copy = cv.cvtColor(im_copy, cv .COLOR_GRAY2BGR)
    cv.imshow('blurred image', im_copy)
    im_flat = im_copy.reshape((-1,3))

    #the k-means algorithm will accept the data type of float32 not uint8
    im_flat = np.float32(im_flat)
    #stopping the kmeans clustering either if the algorithm reaches a specified
    #number of iterations or when the labels change less than a given epilson
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, kwargs['num_iter'] , kwargs['eplison'])

    K = kwargs['K']
    #CHANGE THIS AND SEE WHAT HAPPENS
    #attempts = kwargs['num_iter']
    attempts = 10

    #flags  = cv.KMEANS_RANDOM_CENTERS
    flags = cv.KMEANS_PP_CENTERS
    labels, (centers) = cv.kmeans(im_flat, K, None, criteria, attempts, flags)[1:]

    centers = np.uint8(centers)

    labels = labels.flatten()

    #putting together the segmented image
    seg_im = centers[labels.flatten()]

    #converting it back to the original image shape
    seg_im = seg_im.reshape(im.shape)

    cv.imshow('the segmented image', seg_im)

    #showing each of the segments individually

    for ii in range(labels.max()):
        #making a copy, so the mask is only applied once onto the image
        im_mask = im.copy()
        im_mask = im_mask.reshape((-1,3))
        #setting any segment in the image which corresponds to that cluster to blue
        im_mask[labels == ii] = [255,0,0]
        #converting the flattened pixle matrices into the original image
        im_mask = im_mask.reshape(im.shape)

        cv.imshow('cluster: %s' % ii, im_mask)

#apparently the HSV color scheme is better for image detection
def activity_four_kMeans(raw_im, im, **kwargs):
    im_flat = im.reshape((-1,3))
    im_flat = np.float32(im_flat)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, kwargs['num_iter'] , kwargs['eplison'])

    K = kwargs['K']
    attempts = 10

    #flags  = cv.KMEANS_RANDOM_CENTERS
    flags = cv.KMEANS_PP_CENTERS
    labels, (centers) = cv.kmeans(im_flat, K, None, criteria, attempts, flags)[1:]

    centers = np.uint8(centers)

    labels = labels.flatten()

    #putting together the segmented image
    seg_im = centers[labels.flatten()]

    #converting it back to the original image shape
    seg_im = seg_im.reshape(im.shape)

    cv.imshow('the segmented image', seg_im)

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

        cv.imshow('cluster %s: %s' % (ii, kwargs['color_space']), im_mask)


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

#    #removing noise from the image by performing morphological operations
#    #kernel = np.ones((3,3), np.uint8)
#    #performaing erosion to remove the fore-ground and background in the image
#    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
#    im_eroded = cv.erode(thresh, kernel, iterations=1)
#    cv.imshow('eroded image', im_eroded)
#
#
#    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
#    #cv.imshow('opening image', opening)
#
#    #what we can determine is definetly the background
#    #you want to remove small objects i.e. noise in the image
#    sure_bg = cv.dilate(opening, kernel, iterations=1)
#    cv.imshow("sure background", sure_bg)
#
#    #finding the sure foreground area in the image
#    #calculates the distance from a coloured pixel to a black pixel
#    dist = cv.distanceTransform(opening, cv.DIST_L2, 5)
#    sure_fg = cv.threshold(dist, 0.7 * dist.max(), 255, 0)[1]
#    cv.imshow('sure foreground image', sure_fg)
#
#    #finding the unknown region: this region will be later classfied with the
#    sure_fg = np.uint8(sure_fg)
#    unkown_region = cv.subtract(sure_bg, sure_fg)
#
    #labelling the markers for the watershed algorithm
    markers = cv.connectedComponents(sure_fg)[1]
    cv.imshow('markers', markers)

    #adding 1 to all the labels so the background can be numbered as 1
    markers += 1

    #performing the watershed algorithm on the provided images
    #markers = cv.watershed(thresh, markers)

    cv.imshow('resultant image', thresh)

if __name__ == '__main__':
    imList = ['imgs/diamond2.png', 'imgs/Dugong.jpg']
    #---------------------------------------------------------------------------
    #TASK ONE: Diamond playing card
    #---------------------------------------------------------------------------
    #running all the experiements for the diamond card
    #activity_one_harris_rotated(imList[0])
    #activity_one_harris_scaled(imList[0])
    #activity_one_SIFT_rotated(imList[0])
    #activity_one_SIFT_scaled(imList[0])

    #---------------------------------------------------------------------------
    #TASK ONE: Dugong image
    #---------------------------------------------------------------------------
#    activity_one_harris_rotated(imList[1], 2, color=[0,0,255], thresh=0.06)
#    activity_one_harris_scaled(imList[1], 2, color=[0,0,255], thresh=0.06)
#    activity_one_SIFT_rotated(imList[1])
#    activity_one_SIFT_scaled(imList[1])

    #---------------------------------------------------------------------------
    #TASK TWO: Diamond
    #---------------------------------------------------------------------------
    activity_two_hog(imList[0], (8,1), (28,46))
    #activity_two_rotated(imList[0])

    #---------------------------------------------------------------------------
    #TASK Three:
    #---------------------------------------------------------------------------
    #labels, areas, centroids = activity_three(imList[0],True, im_name='Diamond')
    #activity_three(imList[1], im_name='dugong')

    #---------------------------------------------------------------------------
    #TASK Four:
    #---------------------------------------------------------------------------
    #activity_four(imList[0], 200)
    #activity_four(imList[1], 100)

    #experiments for the water shed algorithm
    #activity_four_watershed(imList[0], True)
    #activity_four_watershed(imList[1])

    #experiments for the K-means algorithm

    #setting the clusters to 3 as I am counting the right side up, and the
    #upside down two's in the image, as two different clusters
    #activity_four_kMeans_RGB(imList[0], num_iter=100, eplison=0.01, K=7)
    #activity_four_kMeans_RGB(imList[1], eplison=0.01, K=14, num_iter=200)

#    im_og = cv.imread(imList[0])
#    im = cv.cvtColor(im_og.copy(), cv.COLOR_BGR2HSV)
#    activity_four_kMeans(im_og.copy(), im, color_space='HSV',eplison=0.01, K=14, num_iter=200)
#
#    im = cv.cvtColor(im_og.copy(), cv.COLOR_BGR2Luv)
#    activity_four_kMeans(im_og.copy(), im, eplison=0.01, K=14, num_iter=200, color_space = 'LLUV')
#
#    im = cv.cvtColor(im_og.copy(), cv.COLOR_BGR2Lab)
#    activity_four_kMeans(im_og.copy(), im, eplison=0.01, K=14, color_space='LAB', num_iter=200)
#
#    activity_four_kMeans(im_og.copy(), im_og.copy(), eplison=0.01, K=14, color_space='BGR', num_iter=200)
#
    #activity_four_kMeans_edges(imList[0])
    #activity_four_kMeans_corners(imList[0])

    #activity_four_kmeans_area(imList[0], areas, labels, centroids)
    cv.waitKey(0)
