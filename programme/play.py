import cv2 as cv
import numpy as np
from myUtils import *

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


some_func(para_one, para_two, list_one, list_two, list_three)
