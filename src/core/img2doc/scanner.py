import cv2
from util import show_image

from . import imutils
from . import transform
from .contours import get_contour


def scan(image, min_quad_area_ratio=0.25, max_quad_angle_range=40):
    """
    Scan an image by applying a perspective transformation and sharpening.
    Args:
        image (Mat): image to scan
        min_quad_area_ratio (float): A contour will be rejected if its corners
            do not form a quadrilateral that covers at least min_quad_area_ratio
            of the original image. Defaults to 0.25.
        max_quad_angle_range (int):  A contour will also be rejected if the range
            of its interior angles exceeds max_quad_angle_range. Defaults to 40.
    :return image (np.ndarray):
    """
    rescaled_height = 500.0

    assert (image is not None)

    ratio = image.shape[0] / rescaled_height
    orig = image.copy()
    rescaled_image = imutils.resize(image, height=int(rescaled_height))

    # get the contour of the document
    screen_cnt = get_contour(rescaled_image, min_quad_area_ratio, max_quad_angle_range)

    # apply the perspective transformation
    warped = transform.four_point_transform(orig, screen_cnt * ratio)
    thresh = imutils.warped2sharpened(warped)

    return thresh
