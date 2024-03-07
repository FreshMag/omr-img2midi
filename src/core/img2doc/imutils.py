# Import the necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt


def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Return the translated image
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def warped2sharpened(image, blur_kernel_size=(1, 1), sharpen_weight=1.5, sharpen_blur_weight=-0.5,
                     threshold_block_size=21, threshold_constant=15):
    # Convert the warped image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpen the image using un-sharp masking
    sharpen = cv2.GaussianBlur(gray, blur_kernel_size, 0)
    sharpen = cv2.addWeighted(gray, sharpen_weight, sharpen, sharpen_blur_weight, 0)

    # Apply adaptive thresholding to get black and white effect
    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   threshold_block_size, threshold_constant)

    return thresh


def show_contours(image, contours, corners):
    cv2.drawContours(image, [contours], -1, (20, 20, 255), 2)
    plt.scatter(*zip(*corners))
    plt.imshow(image)
    plt.show()
