### main script ###

# python library imports
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# script imports
from ImageProcessor import ImageProcessor
from imageToText import *

base_path = "./data/worksheets"

for i in range(51, 52):
    clean_p1 = f"{base_path}/{str(i)}/clean/p1.JPG"
    clean_p2 = f"{base_path}/{str(i)}/clean/p2.JPG"

    if os.path.exists(clean_p1) and os.path.exists(clean_p2):

        # initialize image object
        clean_img1 = ImageProcessor(clean_p1, 0)
        clean_img2 = ImageProcessor(clean_p2)
        
        digitized1 = clean_img1("DIGITIZED").astype(np.uint8)
        digitized2 = clean_img2("DIGITIZED").astype(np.uint8)

        Image.fromarray(digitized1).show()

        # # word recognition
        # word_recog(digitized)

    else:
        print(f"Files not found in folder clean")