import cv2
import numpy as np
from core.img2doc import imutils, transform
from core.img2doc.contours import approx_contours, harris


def scan(image, min_quad_area_ratio=0.25, max_quad_angle_range=40, warp=True, thresh_block_size=21, thresh_c=15):
    """
    Scan an image by applying a perspective transformation and sharpening.
    Args:
        image (Mat): image to scan
        min_quad_area_ratio (float): A contour will be rejected if its corners
            do not form a quadrilateral that covers at least min_quad_area_ratio
            of the original image. Defaults to 0.25.
        max_quad_angle_range (int):  A contour will also be rejected if the range
            of its interior angles exceeds max_quad_angle_range. Defaults to 40.
    :param close_kernel:
    :param warp:
    :return image (np.ndarray):
    """
    rescaled_height = 500.0

    assert (image is not None)

    ratio = image.shape[0] / rescaled_height
    if warp:
        orig = image.copy()
        rescaled_image = imutils.resize(image, height=int(rescaled_height))

        # print("Trying contours approximation")
        quad, contour_found = approx_contours(rescaled_image, min_quad_area_ratio, max_quad_angle_range)
        if not contour_found:
            # print("Contour approximation failed, trying Harris method")
            harris_quad, harris_found = harris(rescaled_image, min_quad_area_ratio, max_quad_angle_range)
            if harris_found:
                # print("Harris found")
                quad = np.flip(harris_quad, axis=None)
            # else:
            # print("Harris failed, taking all the image as quadrilateral")

        # apply the perspective transformation
        warped = transform.four_point_transform(orig, quad * ratio)
    else:
        warped = image.copy()
    thresh = imutils.warped2sharpened(warped, thresh_block_size=thresh_block_size, thresh_c=thresh_c)

    return thresh


def light_scan(image):
    assert (image is not None)
    img = image.copy()
    if len(image.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(img, 30, 255, cv2.THRESH_OTSU)[1]
    return thresh


if __name__ == '__main__':
    img = cv2.imread("../../../data/dinput/document.jpg")
    cv2.imshow("image", scan(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
