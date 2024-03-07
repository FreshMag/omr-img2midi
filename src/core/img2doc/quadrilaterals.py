import math
import numpy as np
import cv2
import itertools
from . import transform


def is_valid_quadrilateral(quad, image_w, image_h, min_quad_area_ratio, max_quad_angle_range):
    """Returns True if the contour satisfies all requirements set at instantitation"""

    return (len(quad) == 4 and cv2.contourArea(quad) > image_w * image_h * min_quad_area_ratio
            and angle_range(quad) < max_quad_angle_range)


def angle_range(quad):
    """
    Returns the range between max and min interior angles of quadrilateral.
    The input quadrilateral must be a numpy array with vertices ordered clockwise
    starting with the top left vertex.
    """
    tl, tr, br, bl = quad
    ura = get_angle(tl[0], tr[0], br[0])
    ula = get_angle(bl[0], tl[0], tr[0])
    lra = get_angle(tr[0], br[0], bl[0])
    lla = get_angle(br[0], bl[0], tl[0])

    angles = [ura, ula, lra, lla]
    return np.ptp(angles)


def angle_between_vectors_degrees(u, v):
    """Returns the angle between two vectors in degrees"""
    return np.degrees(
        math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))


def get_angle(p1, p2, p3):
    """
    Returns the angle between the line segment from p2 to p1
    and the line segment from p2 to p3 in degrees
    """
    a = np.radians(np.array(p1))
    b = np.radians(np.array(p2))
    c = np.radians(np.array(p3))

    avec = a - b
    cvec = c - b

    return angle_between_vectors_degrees(avec, cvec)


def most_probable_quadrilateral(corners):
    quads = []
    for quad in itertools.combinations(corners, 4):
        points = np.array(quad)
        points = transform.order_points(points)
        points = np.array([[p] for p in points], dtype="int32")
        quads.append(points)

    # get top five quadrilaterals by area
    quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
    # sort candidate quadrilaterals by their angle range, which helps remove outliers
    return sorted(quads, key=angle_range)[0]


def get_max_area_or_whole(contours, image_w, image_h):
    if not contours:
        # If we didn't find valuable contours, we take the whole image as contours
        top_right = (image_w, 0)
        bottom_right = (image_w, image_h)
        bottom_left = (0, image_h)
        top_left = (0, 0)
        cnt = np.array([[top_right], [bottom_right], [bottom_left], [top_left]])
    else:
        cnt = max(contours, key=cv2.contourArea)

    return cnt.reshape(4, 2)
