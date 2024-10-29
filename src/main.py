### main script ###

# python libraries (i changed my mind)
from PIL import Image

# script imports
from ImageProcessor import ImageProcessor
from preprocessing import *

testImagePath = "ece549-final-project/data/1.png"

# initialize object
img = ImageProcessor(testImagePath)

# find corners of paper
corners = detect_corners(img(0))

# transform corners of paper to image size
w, h = img.shape
new_corners = np.array( [[0,0],
                         [h,0],
                         [h,w],
                         [0,w]] )

transformed = warp_corners(img(1), order_corners(corners), new_corners, (h,w))

# apply filters
filtered = filter_image(transformed)

Image.fromarray(filtered).show()

# show_corners(img(0), order_corners(corners))