### preprocessing functions ###
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import medfilt2d


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

    # dilate to help avoid text from being detected as edges
    kernel = np.ones((7,7), np.uint8)
    dilated = cv2.dilate(blurred, kernel, iterations=1)

    # canny edge detector
    edges = cv2.Canny(dilated, 50, 150)

    # dilate again to thicken lines in canny edge detector
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

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
            return np.squeeze(approx)

    # If we reach here, then we could not find 4 corners, try using morphology
    # This is essentially a 2nd attempt to find 4 corners, using new method
    blurred = cv2.GaussianBlur(grey, (3, 3), 0)

    # otsu threshold on blurred image
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # attempt using morphology
    kernel = np.ones((7,7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort by area (we're assuming the largest contour will be the paper)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # search from largest to smallest for a contour that looks like it has 4 corners
    for contour in contours:
        # approximate contour into polygon
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # first approximation to have 4 points should be the paper
        if len(approx) == 4:

            # remove extra dim and return
            return np.squeeze(approx)

    raise Exception("Could not find corners")

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


def binarize(img_arr: np.ndarray):
    """
    Takes an image and applies binary filter to remove artifacts.

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

    bound = np.mean(img_arr) - 16

    result = np.where(img_arr < bound, 0, 255)
    
    return result.astype(np.uint8)

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
        dilated_img = cv2.dilate(plane, np.ones((3,3), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
        
    result_norm = cv2.merge(result_norm_planes)

    return result_norm

def enhance_contrast(img_arr: np.ndarray, factor=2):
    """
    Increases contrast for ease of detection
    
    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.
    
    factor : int
        Number determining amount of constract
        x < 1: decrease contrast
        x > 1: increase contrast

    Returns
    -------
    numpy.ndarray
        Resulting contrast enhanced image

    """
    image = img_arr.astype(np.float32)

    new_image = np.clip(factor * (image - 128) + 128, 0, 255)

    return new_image.astype(np.uint8)

def denoise(img_arr: np.ndarray, factor = 5):
    """
    Applies high-quality non-local means denoising
    
    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.

    kernel_size : int
        Number determining kernel_size for controlling amount of denoising

    Returns
    -------
    numpy.ndarray
        Resulting blurred image

    """
    return cv2.fastNlMeansDenoising(img_arr, h = factor)

def blur_denoise(img_arr: np.ndarray, factor = 5):
    """
    Applies basic-quality medianBlur denoising
    
    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.

    kernel_size : int
        Number determining kernel_size for controlling amount of denoising

    Returns
    -------
    numpy.ndarray
        Resulting blurred image

    """
    # Apply a median filter to remove salt-and-pepper noise
    return cv2.medianBlur(img_arr, factor)

def auto_expose(img_arr: np.ndarray):

    """
    Linear adjustment of pixel brightness, such that the resulting average pixel
    value is 200. Result is clipped to (0, 255) range
    
    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.

    Returns
    -------
    numpy.ndarray
        Resulting brightness adjusted image

    """

    ideal_avg = 200
    cur_avg = np.mean(img_arr)

    gap = ideal_avg - cur_avg
    result = img_arr + gap

    return np.clip(result, 0, 255).astype(np.uint8)

def enhance_shadow(img_arr: np.ndarray, factor=2):

    """
    Alter the shadows by applying gamma correction.

    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.
    
    factor : int 
        factor < 1 increases brightness of shadows 
        factor > 1 reduces brightness of shadows 

    Returns
    -------
    numpy.ndarray
        Result image
    
    :return: Image with enhanced shadows as a NumPy array.
    """
    # Apply gamma correction
    img_normalized = img_arr / 255.0
    img_corrected = np.power(img_normalized, factor)
    
    # Return in [0, 255] range as integer
    return np.clip(img_corrected * 255, 0, 255).astype(np.uint8)

def dilate(img_arr: np.ndarray, kernel_size=3, iterations=1):

    """
    Widens lines/text to connect potentially unconnected parts of letters

    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.
    
    kernel_size : int 
        Integer value determining degree of dilation
    
    iterations: int

    Returns
    -------
    numpy.ndarray
        Result image
    
    :return: Image with dilated text as a NumPy array.
    """

    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    return cv2.dilate(img_arr, kernel, iterations=iterations)

def erode(img_arr: np.ndarray, kernel_size=3, iterations=1):

    """
    Narrows text/lines agter dilation to be more legible

    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.
    
    kernel_size : int 
        Integer value determining degree of erosion
    
    iterations: int

    Returns
    -------
    numpy.ndarray
        Result image
    
    :return: Image with eroded text as a NumPy array.
    """

    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    return cv2.erode(img_arr, kernel, iterations=iterations)  

def remove_extremes(img_arr: np.ndarray):

    """
    Eliminates non-paper artifacts from image in preprocessing:
    All values outside a specified range of (low, high) are set to 0 or 255 respectively

    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.

    Returns
    -------
    numpy.ndarray
        Result image
    
    :return: Image with semi-bright / semi-dark (non-paper, non-background) removed
    """

    # Convert to grayscale if needed
    if len(img_arr.shape) == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

    # Delete all pixels below brightness 100
    filtered_arr = img_arr[img_arr > 100]

    # We should be left with mostly the "paper" and some noise
    counts = np.bincount(filtered_arr)

    # Find the most occurring brightness in the paper
    likely_paper_value = np.argmax(counts)

    # We only want to keep values with +-64 from the paper brightness
    low = max(likely_paper_value - 64, 35)
    high = min(likely_paper_value + 64, 220)


    # Attempt to remove all non-paper artifacts
    result = img_arr.copy()
    result[img_arr < low] = 0
    result[img_arr > high] = 255
    return result

def full_cleaning(img_arr: np.ndarray, display_all_results=0):

    """
    Runs the full suite of image operations in a specific order to enhance
    corner detection before image is extracted. Auto adjusts depending on
    characteristics of input image, fixing "dirty" images

    Parameters
    ----------
    img_arr : numpy.ndarray
        A NumPy array representing the image.

    display_all_results : int (0, 1)
        If 1, display each result image after each step 

    Returns
    -------
    numpy.ndarray
        Result image
    
    :return: Modified Image specifically to be used for corner detection
    """

    # For images that may be too bright, attempt darkening
    if np.mean(img_arr) > 128:

        # Step 1: Darken image shadows
        shadows_darkened = enhance_shadow(img_arr)

        # Step 2: Denoise the image
        denoised = denoise(shadows_darkened)

        # Step 3: Darken image shadows AGAIN
        shadows_darkened2 = enhance_shadow(denoised)

        # Step 4: Denoise the image AGAIN
        denoised2 = denoise(shadows_darkened2)

    # Else, for darker images, attempt to brighten
    else:
        # Step 1: Brighten image highlights
        shadows_darkened = enhance_shadow(img_arr, factor=0.5)

        # Step 2: Denoise the image
        denoised = denoise(shadows_darkened)

        # Step 3: Brighten image highlights AGAIN
        shadows_darkened2 = enhance_shadow(denoised, factor=0.8)

        # Step 4: Denoise the image AGAIN
        denoised2 = denoise(shadows_darkened2)

    # Step 5: Erode
    eroded = erode(denoised2, kernel_size=3)

    # Step 6: Dilate2
    dilated = dilate(eroded, kernel_size=3)

    # Step 7: Auto Adjust Exposure
    auto_exposed = auto_expose(dilated)

     # Step 8: Remove Extremities
    result = remove_extremes(auto_exposed)

    if display_all_results:

        print("Step 1")
        plt.imshow(shadows_darkened)
        plt.show()

        print("Step 2")
        plt.imshow(denoised)
        plt.show()

        print("Step 3")
        plt.imshow(shadows_darkened2)
        plt.show()

        print("Step 4")
        plt.imshow(denoised2)
        plt.show()

        print("Step 5")
        plt.imshow(eroded)
        plt.show()

        print("Step 6")
        plt.imshow(dilated)
        plt.show()

        print("Step 7")
        plt.imshow(auto_exposed)
        plt.show()

        print("Step 8")
        plt.imshow(result, cmap='gray')
        plt.show()

    return Image.fromarray(result)


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

    # find corners of paper, on fully_cleaned np_array of image
    corners = detect_corners(np.array(full_cleaning(img_arr)))
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
    filtered = binarize(shadowless)

    return filtered

def simple_digitize(img_arr: np.ndarray):
    """
    Takes and performs the following steps:
        1. removes shadows
        2. binarize image

    Parameters
    ----------
    img_arr : numpy.ndarray
        The COLORED input image array

    Returns
    -------
    numpy.ndarray
        The digitized black and white image
    """

    # remove shadows (preserves color)
    shadowless = remove_shadows(img_arr)

    # apply filters (binarization)
    filtered = binarize(shadowless)

    return filtered