### main script ###

# python library imports
from PIL import Image
import os

# script imports
from ImageProcessor import ImageProcessor

base_path = "./data/worksheets"

## TODO: for some reason images 5 and 6 are broken
for i in range(20):
    clean_p1 = f"{base_path}/{str(i)}/clean/p1.JPG"
    clean_p2 = f"{base_path}/{str(i)}/clean/p2.JPG"

    if os.path.exists(clean_p1) and os.path.exists(clean_p2):

        # initialize image object
        img1 = ImageProcessor(clean_p1)
        img2 = ImageProcessor(clean_p2)

        # show digitized images
        Image.fromarray(img1("DIGITIZED")).show()
        Image.fromarray(img2("DIGITIZED")).show()

    else:
        print(f"Files not found in folder {i}")