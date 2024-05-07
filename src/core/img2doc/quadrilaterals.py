import math
import cv2
import numpy as np


def is_valid_quadrilateral(quad, image_w, image_h, min_quad_area_ratio, max_quad_angle_range):
    """
    Checks if a quadrilateral is valid given the parameters
    :param quad: quadrilateral to be examined
    :param image_w: width of the image
    :param image_h: height of the image
    :param min_quad_area_ratio: percentage of the total area of the image to be considered valid for a quadrilateral
    :param max_quad_angle_range: internal angles range considered valid for a quadrilateral
    :return: ``True`` if the quadrilateral is valid, ``False`` otherwise
    """
    return (len(quad) == 4 and cv2.contourArea(quad) > image_w * image_h * min_quad_area_ratio
            and angle_range(quad) < max_quad_angle_range)


def angle_range(quad):
    """
    Calculates the range between max and min interior angles of quadrilateral.
    :param quad: numpy array with vertices ordered clockwise starting with the top left vertex.
    :return: the angle range between max and min interior angles.
    """
    # Reshapes are necessary to ensure quad has the required shape, no matter the method used to obtain it
    top_left, top_right, bottom_right, bottom_left = (quad[0].reshape(1, 2), quad[1].reshape(1, 2),
                                                      quad[2].reshape(1, 2), quad[3].reshape(1, 2))

    ura = get_angle(top_left[0], top_right[0], bottom_right[0])
    ula = get_angle(bottom_left[0], top_left[0], top_right[0])
    lra = get_angle(top_right[0], bottom_right[0], bottom_left[0])
    lla = get_angle(bottom_right[0], bottom_left[0], top_left[0])

    angles = [ura, ula, lra, lla]
    return np.ptp(angles)


def angle_between_vectors_degrees(u, v):
    """
    Calculates the angle between 2 vectors in degrees.
    :param u: first vector
    :param v: second vector
    :return: the angle between u and v.
    """
    return np.degrees(
        math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))


def get_angle(p1, p2, p3):
    """
    Calculates the angle between the line segment from p2 to p1
    and the line segment from p2 to p3.
    :param p1: first point
    :param p2: second point
    :param p3: third point
    :return: the angle in degrees
    """
    a = np.radians(np.array(p1))
    b = np.radians(np.array(p2))
    c = np.radians(np.array(p3))

    a_vector = a - b
    c_vector = c - b

    return angle_between_vectors_degrees(a_vector, c_vector)


def get_max_area_or_whole(contours, image_w, image_h):
    """
    Returns the contour with the largest area or the whole image if we didn't find a valid contour.
    :param contours: the already filtered contours
    :param image_w: width of the image
    :param image_h: height of the image
    :return: a 2D numpy array [[x1,y1], [x2, y2], ...] with the four vertexes of the quadrilateral and a flag that is
    ``True`` if a valid contour was found, ``False`` otherwise
    """
    flag = False
    if not contours:
        # If we didn't find valuable contours, we take the whole image as contours
        top_right = (image_w, 0)
        bottom_right = (image_w, image_h)
        bottom_left = (0, image_h)
        top_left = (0, 0)
        cnt = np.array([[top_right], [bottom_right], [bottom_left], [top_left]])
    else:
        cnt = max(contours, key=cv2.contourArea)
        flag = True
    return cnt.reshape(4, 2), flag
