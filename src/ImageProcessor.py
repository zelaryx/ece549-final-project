### image class to store different aspects of the image ###
from PIL import Image
import numpy as np

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
        The color image as a NumPy array.
    arr_greyscale : numpy.ndarray
        The grayscale version of the image as a NumPy array.
    """
    def __init__(self, path: str):

        image_path = path
        
        original = Image.open(image_path)
        greyscale = original.convert("L")
        
        # store numpy arrays
        self.arr = np.array(original)
        self.arr_greyscale = np.array(greyscale)

    def show(self):
        Image.open(self.image_path).show()

    def __call__(self, mode=0):
        """
        Returns the image as a numpy array in either color or greyscale.

        Parameters
        ----------
        mode : 0 or 1
            0 -> original color
            1 -> greyscale

        Returns
        -------
        numpy.ndarray
            The image as a NumPy array in the selected mode.

        Raises
        ------
        ValueError
            If `mode` is not 0 or 1.

        """
        if mode == 0:
            return self.arr
        elif mode == 1:
            return self.arr_greyscale
        else:
            raise ValueError("Invalid mode. Use 0 for color or 1 for greyscale.")