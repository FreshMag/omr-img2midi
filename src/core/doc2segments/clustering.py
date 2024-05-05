import cv2

from .bounding_box import BoundingBox


def cluster_bboxes(image, min_cluster_area_ratio=0.005, min_cluster_height_ratio=0.03, safety_margin_ratio=0.03,
                   max_cluster_area_ratio=0.25, max_cluster_height_ratio=0.25):
    """
    Clusters the contours of the image, producing rectangular bounding boxes already merged with Non-Maxima suppression
    :param image: Image to be clustered
    :param min_cluster_area_ratio: Percentage of the minimum area of the image to be considered a cluster
    :param min_cluster_height_ratio: Percentage of the minimum height of the image to be considered a cluster
    :param safety_margin_ratio: Percentage of the area of the image to added as safety margin
    :param max_cluster_area_ratio: Percentage of the maximum area of the image to be considered a cluster
    :param max_cluster_height_ratio: Percentage of the maximum height of the image to be considered a cluster
    :return: Bounding boxes merged with Non-Maxima suppression
    """
    # Function to filter out contours with low area
    def filter_contours(cntr, min_area, min_height, max_area, max_height):
        return [contour for contour in cntr if
                min_area <= cv2.contourArea(contour) <= max_area and
                min_height <= cv2.boundingRect(contour)[3] <= max_height]

    # Invert the image so that lines are white and background is black
    inverted_image = cv2.bitwise_not(image)

    h, w = inverted_image.shape

    # Find contours
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with low area
    valid_contours = filter_contours(contours, min_cluster_area_ratio * h * w, min_cluster_height_ratio * h,
                                     max_cluster_area_ratio * h * w, max_cluster_height_ratio * h)

    # Get bounding boxes for valid contours
    bounding_boxes = [cv2.boundingRect(contour) for contour in valid_contours]

    # We then merge the overlapping bboxes
    bounding_boxes = BoundingBox.merge_overlapping_boxes(bounding_boxes)
    # And expand the boxes with a safety margin
    bounding_boxes = [bbox.expand(safety_margin_ratio*h, safety_margin_ratio*h) for bbox in bounding_boxes]

    return bounding_boxes
