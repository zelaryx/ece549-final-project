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

    ### NUOYAN THESE KERNEL SIZES AND THRESHOLD NUMBERS NEED
    ### TO BE FIXED BECAUSE THEYRE HARDCODED TO MATCH THE IMAGE SIZE
    ## TODO: fix hardcoded numbers below

    # gaussian blur to remove noise
    blurred = cv2.GaussianBlur(grey, (9, 9), 0)

    # dilate to help avoid text from being detected as edges
    kernel = np.ones((15,15), np.uint8)
    dilated = cv2.dilate(blurred, kernel, iterations=1)

    # canny edge detector
    edges = cv2.Canny(dilated, 50, 150)

    # dilate again to thicken lines in canny edge detector
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    ### TILL HERE

    # contour
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sort by area (we're assuming the largest contour will be the paper)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    ## show contour with largest area
    # contour_image = img_arr # np.ones_like(img_arr) * 255
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


def remove_shadows(img_arr: np.ndarray):
    """
    Takes color image of just the paper and removes shadows on the paper
    https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv

    Parameters
    ----------
    img_arr : numpy.ndarray
        The input colored image

    Returns
    -------
    numpy.ndarray
        The shadowless image
    """
    rgb_planes = cv2.split(img_arr)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
        
    result_norm = cv2.merge(result_norm_planes)

    return result_norm


def filter_image(img_arr: np.ndarray):
    """
    Takes an image and applies binary filter + whatnot to remove artifacts.

    "whatnot" is currently nothing LOL

    Parameters
    ----------
    img_arr : numpy.ndarray
        The input image

    Returns
    -------
    numpy.ndarray
        The filtered image
    """
    if len(img_arr.shape) == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

    # enhanced = cv2.convertScaleAbs(img_arr, alpha=1.5, beta=30) # this gets rid of pencil marks....
    inverted = cv2.bitwise_not(img_arr)
    kernel = np.ones((3,3), np.uint8)
    inverted_dilated = cv2.dilate(inverted, kernel, iterations=1)

    uninverted = cv2.bitwise_not(inverted_dilated)

    # _, binary = cv2.threshold(uninverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(uninverted, 220, 255, cv2.THRESH_BINARY)
    return binary


def digitize(img_arr: np.ndarray):
    """
    Takes and performs the following steps:
        1. removes shadows
        2. finds corners of image
        3. warp image to get rid of background
        4. removes shadows
        5. binarize image

    Parameters
    ----------
    img_arr : numpy.ndarray
        The COLORED input image array

    Returns
    -------
    numpy.ndarray
        The digitized black and white image
    """

    # find corners of paper
    corners = detect_corners(img_arr)
    # show_corners(img_arr, order_corners(corners))

    # transform corners of paper to image size (preserves color)
    w, h = img_arr.shape[:2]
    new_corners = np.array( [[0,0],
                             [h,0],
                             [h,w],
                             [0,w]] )

    transformed = warp_corners(img_arr, order_corners(corners), new_corners, (h,w))

    # remove shadows (preserves color)
    shadowless = remove_shadows(transformed)

    # apply filters (binarization)
    filtered = filter_image(shadowless)

    return filtered