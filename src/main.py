### main script ###

# python library imports
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# script imports
from ImageProcessor import ImageProcessor
from imageToText import *

base_path = "./data/samples"

for i in range(1, 4):
    sample = f"{base_path}/sample_text_{i}.png"

    if os.path.exists(sample):

        # initialize image object
        sample_img = ImageProcessor(sample)
        digitized = sample_img("SIMPLE_DIGITIZED").astype(np.uint8)

        # # show digitized images
        # Image.fromarray(digitized).show()

        # word recognition
        word_recog(digitized)

    else:
        print(f"Files not found in folder {i}")