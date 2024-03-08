import cv2
from .bounding_box import BoundingBox


def cluster_bboxes(image, min_cluster_area=2000, min_cluster_height=50):
    # Function to filter out contours with low area
    def filter_contours(cntr, min_area, min_height):
        return [contour for contour in cntr if cv2.contourArea(contour) >= min_area and
                cv2.boundingRect(contour)[3] >= min_height]


    # Invert the image so that lines are white and background is black
    inverted_image = cv2.bitwise_not(image)

    # Find contours
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with low area
    valid_contours = filter_contours(contours, min_cluster_area, min_cluster_height)

    # Get bounding boxes for valid contours
    bounding_boxes = [cv2.boundingRect(contour) for contour in valid_contours]

    # We then merge the overlapping bboxes
    bounding_boxes = BoundingBox.merge_overlapping_boxes(bounding_boxes)

    return bounding_boxes
