import cv2
import numpy as np

from .quadrilaterals import is_valid_quadrilateral, get_max_area_or_whole
from .corner_utils import get_corners


def get_contour(rescaled_image, min_quad_area_ratio, max_quad_angle_range):
    """
    Returns a numpy array of shape (4, 2) containing the vertices of the four corners
    of the document in the image. It considers the corners returned from get_corners()
    and uses heuristics to choose the four corners that most likely represent
    the corners of the document. If no corners were found, or the four corners represent
    a quadrilateral that is too small or convex, it returns the original four corners.
    """

    image_h, image_w, _ = rescaled_image.shape

    edged = get_edged(rescaled_image)
    approx_contours = []

    #
    # test_corners = get_corners(edged)
    #
    # if len(test_corners) >= 4:
    #     approx_quad = most_probable_quadrilateral(test_corners)
    #
    #     if is_valid_quadrilateral(approx_quad, image_w, image_h, min_quad_area_ratio, max_quad_angle_range):
    #         approx_contours.append(approx_quad)
    #
    #     # for debugging: uncomment the code below to draw the corners and countour found
    #     # by get_corners() and overlay it on the image
    #
    #     # imutils.show_contours(rescaled_image, approx_quad, test_corners)

    # also attempt to find contours directly from the edged image, which occasionally
    # produces better results
    (contours, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in contours:
        # approximate the contour
        approx_quad = cv2.approxPolyDP(c, 80, True)
        if is_valid_quadrilateral(approx_quad, image_w, image_h, min_quad_area_ratio, max_quad_angle_range):
            approx_contours.append(approx_quad)
            break

    # If we did not find any valid contours, just use the whole image

    final_contour = get_max_area_or_whole(approx_contours, image_w, image_h)
    return final_contour


def get_edged(image, canny_thresh=84, morph_size=9):
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # dilate helps to remove potential holes between edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # find edges and mark them in the output map using the Canny algorithm
    edged = cv2.Canny(dilated, 0, canny_thresh)
    return edged
