### image class to store different aspects of the image ###

# python library imports
from PIL import Image
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


    Attributes
    ----------
    arr : numpy.ndarray
        The color image as a Numpy array.
    arr_greyscale : numpy.ndarray
        The grayscale version of the image as a Numpy array.
    size : tuple
        The (w, h) of the image.
    """
    def __init__(self, path: str):

        image_path = path
        
        original = Image.open(image_path)
        greyscale = original.convert("L")
        
        # store numpy arrays
        self.arr = np.array(original)
        self.arr_greyscale = np.array(greyscale)
        self.shape = self.arr_greyscale.shape
        self.arr_digitized = digitize(self.arr)

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
        else:
            raise ValueError("Invalid mode.")