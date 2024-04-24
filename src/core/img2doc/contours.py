import math

import cv2
import numpy as np
from core.img2doc.quadrilaterals import is_valid_quadrilateral, get_max_area_or_whole


def draw_points(image, points):
    img = image.copy()
    for point in points:
        f_point = (min(point[1], img.shape[1] - 1), min(point[0], img.shape[0] - 1))
        img = cv2.circle(img, f_point, radius=0, color=(0, 0, 255), thickness=-1)
    return img


def harris_quad(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    mask = dst > 0.01 * dst.max()

    h, w = gray.shape
    true_indices = np.argwhere(mask)
    if len(true_indices) == 0:
        return [[h-1, w-1], [h-1, 0], [0, 0], [0, w-1]]

    top_left = true_indices[np.argmin([math.dist(p, (0, 0)) for p in true_indices])]
    top_right = true_indices[np.argmin([math.dist(p, (0, w-1)) for p in true_indices])]
    bottom_right = true_indices[np.argmin([math.dist(p, (h-1, w-1)) for p in true_indices])]
    bottom_left = true_indices[np.argmin([math.dist(p, (h-1, 0)) for p in true_indices])]

    corners = np.array([top_left, top_right, bottom_right, bottom_left])
    return corners


def harris(rescaled_image, min_quad_area_ratio, max_quad_angle_range):
    image_h, image_w, _ = rescaled_image.shape
    rect = harris_quad(rescaled_image)
    if is_valid_quadrilateral(rect, image_w, image_h, min_quad_area_ratio, max_quad_angle_range):
        return rect, True
    return None, False


def approx_contours(rescaled_image, min_quad_area_ratio, max_quad_angle_range):
    """
    Returns a numpy array of shape (4, 2) containing the vertices of the four corners
    of the document in the image. If no corners were found, or the four corners represent
    a quadrilateral that is too small or convex, it returns the original four corners.
    """
    image_h, image_w, _ = rescaled_image.shape

    found = False
    it = 0
    max_it = 1
    thresh = range(84 - max_it, 84)
    while not found and it < max_it:
        edged = get_edged(rescaled_image, thresh[it])
        approx_contours = []

        # find contours directly from the edged image
        (contours, hierarchy) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            for eps in np.linspace(0.001, 0.05, 10):
                approx_quad = cv2.approxPolyDP(c, eps * peri, True)
                if is_valid_quadrilateral(approx_quad, image_w, image_h, min_quad_area_ratio, max_quad_angle_range):
                    found = True
                    approx_contours.append(approx_quad)
                    break
            if found:
                break
        final_contour, found = get_max_area_or_whole(approx_contours, image_w, image_h)
        it += 1

    return final_contour, found


def get_edged(image, canny_thresh=84, morph_size=9):
    # convert the image to grayscale and blur it slightly
    gray = image
    if len(gray.shape) != 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # dilate helps to remove potential holes between edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # find edges and mark them in the output map using the Canny algorithm
    edged = cv2.Canny(dilated, 0, canny_thresh)
    return edged
