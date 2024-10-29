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

def order_corners(corners):
    """
    Orders a set of four points to the following order:\
    top-left, top-right, bottom-right, bottom-left.

    Parameters
    ----------
    corners : numpy.ndarray
        A 4x2 array of corners.

    Returns
    -------
    numpy.ndarray
        A 4x2 array of points ordered as top-left, top-right, bottom-right, bottom-left.
    """

    corners = np.array(corners)
    
    sort_y = corners[np.argsort(corners[:, 1])]
    top = sort_y[:2]
    bottom = sort_y[2:]

    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left])

def warp_corners(img, src, dst, output_size):
    """
    Warps an image from the source points to the destination points.

    Parameters
    ----------
    img : numpy.ndarray
        The input image including the to be warped portion
    src : numpy.ndarray
        A 4x2 array of the corners to be warped
    dst : numpy.ndarray
        A 4x2 array of the target corners
    output_size : tuple
        The output size of the warped image (width, height)

    Returns
    -------
    numpy.ndarray
        The warped image.
    """

    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)

    h = cv2.getPerspectiveTransform(src, dst)

    warped_img = cv2.warpPerspective(img, h, output_size)

    return warped_img

def filter_image(img):
    """
    Takes an image and applies binary filter + whatnot to remove artifacts

    Parameters
    ----------
    img : numpy.ndarray
        The input image

    Returns
    -------
    numpy.ndarray
        The filtered image
    """

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary