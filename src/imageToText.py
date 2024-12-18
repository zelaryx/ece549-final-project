import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pytesseract
from PIL import Image

from textCorrection import spellcheck

def cc_analysis(img_gray, img):

    # dynamically select c-value based on the image contrast
    contrast = np.std(img_gray)
    val= max(1, int(contrast / 10)) 
    # print(val)
    
     # Higher contrast -> Higher `C`
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, val)


    # Display the thresholded image using matplotlib
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')
    # plt.show()


    # apply connected component analysis to the thresholded image
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    mask = np.zeros(img_gray.shape, dtype="uint8")

    output_dir = '/Users/alyssahuang/Downloads/cc_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # loop over the number of unique connected component labels, skipping
    # over the first label (as label zero is the background)
    for i in range(1, numLabels):
        # extract the connected component statistics for the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # these dimensions are dependent on how the student wrote -> angled in a particular way??
        keepWidth = w > 0 and w < 100
        keepHeight = h > 0 and h < 100
        keepArea = area > 100 and area < 200

        if keepWidth and keepHeight and keepArea:
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

            # Save the connected component to a file
            
            component_img = img[y:y+h, x:x+w]
            component_path = os.path.join(output_dir, f'component_{i}.png')
            cv2.imwrite(component_path, component_img)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the original input image and the mask for the license plate
    # characters
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    # plt.show()

    plt.imshow(img)
    # plt.show()

def load_images(img_path):

    img = cv2.imread(img_path)
    blurred_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = pytesseract.image_to_boxes(Image.open(img_path))
    text = ''.join([line.split()[0] for line in boxes.splitlines()])

    # Apply Gaussian blur to reduce noise
    # blurred_image = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # dilate image to make strokes easier to detect -> use erosion cuz inverted
    kernel = np.ones((3,3), np.uint8)
    blurred_image = cv2.erode(blurred_image, kernel, iterations=1)

    plt.imshow(blurred_image, cmap='gray')
    # plt.show()

    cc_analysis(blurred_image, img)
    return text

def main():
    text = load_images('data/samples/sample_text_3.png')
    print(spellcheck(text))

if __name__ == '__main__':
    main()