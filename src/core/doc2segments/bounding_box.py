import cv2


class BoundingBox:
    def __init__(self, top_left_x, top_left_y, width, height):
        self.top_left = (top_left_x, top_left_y)
        self.bottom_right = (top_left_x + width, top_left_y + height)

    def apply_on_image(self, image):
        return image[self.top_left[1]:self.bottom_right[1], self.top_left[0]:self.bottom_right[0]]

    def draw_on_image(self, image, color=(0, 0, 255), thickness=5):
        return cv2.rectangle(image, self.top_left, self.bottom_right, color, thickness)

    @staticmethod
    def from_opencv_rect(rect):
        return BoundingBox(rect[0], rect[1], rect[2], rect[3])

    @staticmethod
    def merge_overlapping_boxes(boxes):
        def are_overlapping(box1, box2):
            """
                Check if two rectangles overlap.

                Parameters:
                    box1 (tuple): First rectangle in the form (x, y, width, height).
                    box2 (tuple): Second rectangle in the form (x, y, width, height).

                Returns:
                    bool: True if rectangles overlap, False otherwise.
                """
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            # Check if one rectangle is to the left of the other
            if x1 + w1 < x2 or x2 + w2 < x1:
                return False

            # Check if one rectangle is above the other
            if y1 + h1 < y2 or y2 + h2 < y1:
                return False

            return True


