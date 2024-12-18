### main script ###

# python library imports
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# script imports
from ImageProcessor import ImageProcessor

base_path = "./data/worksheets"

for i in range(20):
    clean_p1 = f"{base_path}/{str(i)}/clean/p1.JPG"
    clean_p2 = f"{base_path}/{str(i)}/clean/p2.JPG"
    # dirty_p1 = f"{base_path}/{str(i)}/dirty/p1.JPG"
    # dirty_p2 = f"{base_path}/{str(i)}/dirty/p2.JPG"
    if os.path.exists(clean_p1) and os.path.exists(clean_p2):

        # initialize image object
        clean_img1 = ImageProcessor(clean_p1)
        clean_img2 = ImageProcessor(clean_p2)

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
    else:
        print(f"Files not found in folder {i}")