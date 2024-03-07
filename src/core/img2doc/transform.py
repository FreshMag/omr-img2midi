# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import cv2


def order_points(pts):
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (top_left, bottom_left) = left_most

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    d = dist.cdist(top_left[np.newaxis], right_most, "euclidean")[0]
    (bottom_right, top_right) = right_most[np.argsort(d)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def get_max_distance(axis1, axis2):
    pt1, pt2 = axis1
    pt3, pt4 = axis2
    dim1 = dist.euclidean(pt1, pt2)
    dim2 = dist.euclidean(pt3, pt4)
    return max(int(dim1), int(dim2))


def get_containing_dims(quad):
    (top_left, top_right, bottom_right, bottom_left) = quad

    max_width = get_max_distance((bottom_right, bottom_left), (top_right, top_left))
    max_height = get_max_distance((top_right, bottom_right), (top_left, bottom_left))

    return max_width, max_height


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    quad = order_points(pts)

    new_width, new_height = get_containing_dims(quad)
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [new_width - 1, 0],
        [new_width - 1, new_height - 1],
        [0, new_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, M, (new_width, new_height))

    # return the warped image
    return warped
