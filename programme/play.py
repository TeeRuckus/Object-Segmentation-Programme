import cv2 as cv
import numpy as np
from myUtils import *

imageOne = np.zeros((10,10), dtype='int32')
imageTwo = [[255 for cols in range(10)] for rows in range(10)]
imageTwo = np.array(imageTwo, dtype='int32')
print(imageOne)
print(imageTwo)
#imageOne_hist = calc_histograms(imageOne)

cv.imshow('black image', imageOne)
cv.imshow('white image', imageTwo)
cv.waitKey()
cv.destroyAllWindows()
