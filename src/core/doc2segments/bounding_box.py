import itertools

import cv2


class BoundingBox:
    """
    Bounding box used in doc2segments. Utility class to avoid dealing with tuple-based bounding boxes.
    """

    def __init__(self, top_left_x, top_left_y, width, height):
        """
        Instantiates a bounding box given the coordinates and width and height.
        :param top_left_x (int): Top left coordinate of the bounding box.
        :param top_left_y (int): Top left coordinate of the bounding box.
        :param width (int): Width of the bounding box:
        :param height (int): Height of the bounding box:
        """
        self.top_left = (top_left_x, top_left_y)
        self.bottom_right = (top_left_x + width, top_left_y + height)

    def apply_on_image(self, image):
        """
        Applies a bounding box on an image, returning the selected area.
        :param image: Image to apply bounding box on
        :return: The area of the image described by the bounding box.
        """
        return image[self.top_left[1]:self.bottom_right[1], self.top_left[0]:self.bottom_right[0]]

    def draw_on_image(self, image, color=(0, 0, 255), thickness=5):
        """
        Draws a bounding box on an image.
        :param image: Image to draw bounding box on
        :param color: BGR color of the bounding box
        :param thickness: Thickness of the bounding box lines
        :return: Image with the bounding box drawn on it.
        """
        return cv2.rectangle(image, self.top_left, self.bottom_right, color, thickness)

    def expand(self, width, height):
        """
        Expands the bounding box given a width and height.
        :param width: New width of the bounding box
        :param height: New height of the bounding box
        :return: A new bounding box with expanded width and height.
        """
        self.top_left = (max(self.top_left[0] - int(width / 2), 0), max(self.top_left[1] - int(height / 2), 0))
        self.bottom_right = (self.bottom_right[0] + int(width / 2), self.bottom_right[1] + int(height / 2))
        return self

    @staticmethod
    def from_opencv_rect(rect):
        """
        Converts a OpenCV rectangle into a bounding box.
        :param rect: OpenCV rectangle coordinates
        :return: The converted bounding box.
        """
        return BoundingBox(rect[0], rect[1], rect[2], rect[3])

    @staticmethod
    def merge_overlapping_boxes(boxes):
        """
        Merges overlapping bounding boxes (aka Non-Maxima suppression)
        :param boxes: List of bounding boxes
        :return: Merged bounding boxes
        """
        def union(a, b):
            x = min(a[0], b[0])
            y = min(a[1], b[1])
            w = max(a[0] + a[2], b[0] + b[2]) - x
            h = max(a[1] + a[3], b[1] + b[3]) - y
            return x, y, w, h

        def intersection(a, b):
            x = max(a[0], b[0])
            y = max(a[1], b[1])
            w = min(a[0] + a[2], b[0] + b[2]) - x
            h = min(a[1] + a[3], b[1] + b[3]) - y
            if w < 0 or h < 0:
                return ()
            return x, y, w, h

        rects = boxes.copy()
        it = 0
        while it < 100:
            found = 0
            for ra, rb in itertools.combinations(rects, 2):
                if intersection(ra, rb):
                    if ra in rects:
                        rects.remove(ra)
                    if rb in rects:
                        rects.remove(rb)
                    rects.append((union(ra, rb)))
                    found = 1
                    break
            if found == 0:
                break
            it = it + 1

        return [BoundingBox.from_opencv_rect(box) for box in rects]
