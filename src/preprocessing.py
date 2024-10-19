### preprocessing functions ###

import cv2
import numpy as np

def detect_corners(img_arr: np.ndarray):
    """
    Detects corners in the given image and returns their coordinates.

    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.

    Returns
    -------
    numpy.ndarray
        A 4x2 array of corner coordinates in the format:
        
        [[x1, y1],
         [x2, y2],
         [x3, y3],
         [x4, y4]]

    """

    # convert to greyscale if not already
    grey = img_arr if len(img_arr.shape) == 2 else cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

    # gaussian blur to remove noise
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)

    # canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # contour
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sort by area (we're assuming the largest contour will be the paper)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # # show contour with largest area
    # contour_image = np.ones_like(img_arr) * 255
    # cv2.drawContours(contour_image, contours, 0, (0,255,0), 3)
    # cv2.imshow('Contours', contour_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # search from largest to smallest for a contour that looks like it has 4 corners
    for contour in contours:
        # approximate contour into polygon
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # first approximation to have 4 points should be the paper
        if len(approx) == 4:
            corners = approx
            break

    # get rid of the extra dimension in the middle idk why it needs that extra dimension ngl
    corners = np.squeeze(corners)

    # return
    return corners

def show_corners(img_arr: np.ndarray, corners):
    """
    Given an image and points in a 2D numpy array, show the points

    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.

    corners
        A 2D NumPy array with the corners

    Returns
    -------
    None.

    """
    
    # convert image to color if image is in greyscale
    img_color = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR) if len(img_arr.shape) == 2 else img_arr

    for point in corners:
        cv2.circle(img_color, tuple(point), 10, (0, 255, 0), -1)

    # # Show the image with detected corners
    cv2.imshow("Corners Detected (press ANY KEY to exit)", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()