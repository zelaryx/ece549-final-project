### image class to store different aspects of the image ###

# python library imports
from PIL import Image, ImageOps
import numpy as np

# script imports
from preprocessing import *

class ImageProcessor:
    """
    Not sure if this is even needed. It's basically a class that takes a path to an image \
    and stores the Numpy array to both its colored version and greyscale version. It'll be \
    helpful if we do this on a lot of images i guess

    Parameters
    ----------
    path : str
        A string of the path the image.

    auto_resize 
        (default = 1)
        Resizes the image to a standardized size, while maintaining proportions
        Maximizes either the height or width at 1600 px


    Attributes
    ----------
    arr : numpy.ndarray
        The color image as a Numpy array.
    arr_greyscale : numpy.ndarray
        The grayscale version of the image as a Numpy array.
    shape : tuple
        The (w, h) of the image.
    arr_digitized : numpy.ndarray
        The binarized and warped version of the image as a Numpy array.
    """
    def __init__(self, path: str, auto_norm = 1):

        # Extracts exif info (whether image has been rotated 90deg)
        original = ImageOps.exif_transpose(Image.open(path))

        # By default, always normalize for kernel sizes
        if auto_norm:

            # Normalize image to 1600px as max dimension
            max_dim = 1600
            w, h = original.size

            # Algebra/Geometry for new dimensions
            if h < w:
                resize_width = max_dim
                resize_height = int((max_dim / w) * h)

            else:
                resize_height = max_dim
                resize_width = int((max_dim / h) * w)

            # Resize accordingly
            original = original.resize((resize_width, resize_height))

        greyscale = original.convert("L")
        
        # store numpy arrays
        self.arr = np.array(original)
        self.arr_greyscale = np.array(greyscale)
        self.shape = self.arr_greyscale.shape
        self.arr_digitized = digitize(self.arr)
        self.arr_simple_digitized = simple_digitize(self.arr)

    def show(self):
        Image.open(self.image_path).show()

    def __call__(self, mode="ORIGINAL"):
        """
        Returns the image as a numpy array in either color or greyscale.

        Parameters
        ----------
        mode : string
            "ORIGINAL"  -> original color
            "GREYSCALE" -> greyscale
            "DIGITIZED" -> digitized
            "SIMPLE_DIGITIZED" -> digitized but without the warping

        Returns
        -------
        numpy.ndarray
            The image as a NumPy array in the selected mode.

        Raises
        ------
        ValueError
            If `mode` is not 0 or 1.

        """
        if mode == "ORIGINAL":
            return self.arr
        elif mode == "GREYSCALE":
            return self.arr_greyscale
        elif mode == "DIGITIZED":
            return self.arr_digitized
        elif mode == "SIMPLE_DIGITIZED":
            return self.arr_simple_digitized
        else:
            raise ValueError("Invalid mode.")