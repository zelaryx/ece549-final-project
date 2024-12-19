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

base_path = "./data/samples"

for i in range(1, 4):
    clean_p1 = f"{base_path}/{str(i)}/clean/p1.JPG"
    clean_p2 = f"{base_path}/{str(i)}/clean/p2.JPG"
    sample = f"{base_path}/sample_text_{i}.png"

    if os.path.exists(sample):

        # initialize image object
        clean_img1 = ImageProcessor(clean_p1)
        clean_img2 = ImageProcessor(clean_p2)
        
        sample_img = ImageProcessor(sample)
        digitized = sample_img("SIMPLE_DIGITIZED").astype(np.uint8)

        # dirty_img1 = ImageProcessor(dirty_p1)
        # dirty_img2 = ImageProcessor(dirty_p2)

        # # show digitized images
        Image.fromarray(clean_img1("DIGITIZED").astype(np.uint8)).show()
        Image.fromarray(clean_img2("DIGITIZED").astype(np.uint8)).show()
        # Image.fromarray(dirty_img1("DIGITIZED").astype(np.uint8)).show()
        # Image.fromarray(dirty_img2("DIGITIZED").astype(np.uint8)).show()


        # # Ease of viewing if plotting in a ipynb
        # plt.imshow(clean_img1("DIGITIZED"), cmap='gray')
        # plt.show()
        # plt.imshow(clean_img2("DIGITIZED"), cmap='gray')
        # plt.show()
        # plt.imshow(dirty_img1("DIGITIZED"), cmap='gray')
        # plt.show()
        # plt.imshow(dirty_img2("DIGITIZED"), cmap='gray')
        # plt.show()
        # # show digitized images
        # Image.fromarray(digitized).show()

        # word recognition
        word_recog(digitized)

    else:
        print(f"Files not found in folder {i}")