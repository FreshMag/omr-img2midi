import numpy as np
import cv2
from matplotlib import pyplot as plt


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    """
    Performs automatic brightness and contrast optimization with optional histogram clipping
    :param image: to be optimized
    :param clip_hist_percent: clipping percent for histogram clipping
    :return: optimized image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes an image to specified width and height
    :param image: to be resized
    :param width: new width
    :param height: new height
    :param inter: type of interpolation
    :return: the resized image
    """
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def warped2sharpened(image, thresh_block_size, thresh_c):
    """
    To apply on the warped image. Sharpens the image and applies an adaptive threshold
    :param image: to process
    :param thresh_block_size: block size used in the thresh
    :param thresh_c: constant used in the thresh
    :return: processed image
    """
    img = automatic_brightness_and_contrast(image)[0]
    # Convert the warped image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sharpen image
    sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

    # apply adaptive threshold to get black and white effect
    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   thresh_block_size, thresh_c)

    return thresh


def show_contours(image, contours, corners):
    """
    Utility function to show contors
    :param image: to be drawn on
    :param contours: to be drawn
    :param corners: corners to highlight
    :return: None
    """
    cv2.drawContours(image, [contours], -1, (20, 20, 255), 2)
    plt.scatter(*zip(*corners))
    plt.imshow(image)
    plt.show()
