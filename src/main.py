### main script ###

# python libraries (none, all imports should be in subfiles)

# script imports
from ImageProcessor import ImageProcessor
from preprocessing import *

testImagePath = "ece549-final-project/data/1.png"

# initialize object
img = ImageProcessor(testImagePath)
corners = detect_corners(img(0))

show_corners(img(0), corners)