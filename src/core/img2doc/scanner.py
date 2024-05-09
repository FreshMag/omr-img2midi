import cv2
import numpy as np
from core.img2doc import imutils, transform
from core.img2doc.contours import approx_contours, harris


def scan(image, min_quad_area_ratio=0.25, max_quad_angle_range=40, warp=True, thresh_block_size_ratio=0.018,
         thresh_c_ratio=0.0050, component_pixel_thresh_ratio=6e-05):
    """
    Scan an image by applying a perspective transformation and sharpening.
    :param image: image to be scanned, usually a photo of the document
    :param min_quad_area_ratio: percentage of the total area of the image to be considered valid for a quadrilateral containing the document sheet
    :param max_quad_angle_range: internal angles range considered valid for a quadrilateral containing the document sheet
    :param warp (boolean) whether to warp the image or not
    :param thresh_block_size_ratio: percentage of the width of the image to be used as block size in the thresholding
    :param thresh_c_ratio: percentage of the width of the image to be used as constant in the thresholding
    :param component_pixel_thresh_ratio: percentage of the area of the image to be used as thresh for connected components pixel number
    :return: the scanned image
    """
    rescaled_height = 500.0

    assert (image is not None)

    ratio = image.shape[0] / rescaled_height
    if warp:
        warped = warp_image(image, max_quad_angle_range, min_quad_area_ratio, ratio, rescaled_height)
    else:
        warped = image.copy()

    final_image = clean_image(warped, component_pixel_thresh_ratio, thresh_block_size_ratio, thresh_c_ratio)
    return final_image


def clean_image(warped, component_pixel_thresh_ratio, thresh_block_size_ratio, thresh_c_ratio):
    """
    Internal function used by ``scan()`` to clean the image after warping. See ``scan()`` for details.
    :return: the cleaned image
    """

    def round_up_to_odd(f):
        return int(np.ceil(f) // 2 * 2 + 1)

    h, w, _ = warped.shape
    thresh = imutils.warped2sharpened(warped, thresh_block_size=round_up_to_odd(thresh_block_size_ratio * w),
                                      thresh_c=int(thresh_c_ratio * w))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(thresh), None, None,
                                                                  None, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]
    final_image = np.zeros(labels.shape, np.uint8)
    component_pixel_thresh = ((h * w) - cv2.countNonZero(thresh)) * component_pixel_thresh_ratio
    for i in range(0, n_labels - 1):
        if areas[i] >= component_pixel_thresh:  # keep
            final_image[labels == i + 1] = 255
    final_image = cv2.bitwise_not(final_image)
    return final_image


def warp_image(image, max_quad_angle_range, min_quad_area_ratio, ratio, rescaled_height):
    """
    Internal function used by ``scan()`` to warp the image to obtain a bird-eye view. See ``scan()`` for details.
    :return: the warped image
    """
    orig = image.copy()
    rescaled_image = imutils.resize(image, height=int(rescaled_height))
    quad, contour_found = approx_contours(rescaled_image, min_quad_area_ratio, max_quad_angle_range)
    if not contour_found:
        # if contour method didn't work we try harris method
        harris_quad, harris_found = harris(rescaled_image, min_quad_area_ratio, max_quad_angle_range)
        if harris_found:
            quad = np.flip(harris_quad, axis=None)

    # apply the perspective transformation
    warped = transform.four_point_transform(orig, quad * ratio)
    return warped


def light_scan(image):
    """
    Used for performing a delicate scan. Used for example on already scanned images to binarize them. In practice,
    it only applied a simple thresholding
    :param image: image to be scanned
    :return: the threshold image
    """
    assert (image is not None)
    img = image.copy()
    if len(image.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(img, 30, 255, cv2.THRESH_OTSU)[1]
    return thresh


