import itertools

import cv2


class BoundingBox:
    def __init__(self, top_left_x, top_left_y, width, height):
        self.top_left = (top_left_x, top_left_y)
        self.bottom_right = (top_left_x + width, top_left_y + height)

    def apply_on_image(self, image):
        return image[self.top_left[1]:self.bottom_right[1], self.top_left[0]:self.bottom_right[0]]

    def draw_on_image(self, image, color=(0, 0, 255), thickness=5):
        return cv2.rectangle(image, self.top_left, self.bottom_right, color, thickness)

    def expand(self, width, height):
        self.top_left = (max(self.top_left[0] - int(width / 2), 0), max(self.top_left[1] - int(height / 2), 0))
        self.bottom_right = (self.bottom_right[0] + int(width / 2), self.bottom_right[1] + int(height / 2))
        return self

    @staticmethod
    def from_opencv_rect(rect):
        return BoundingBox(rect[0], rect[1], rect[2], rect[3])

    @staticmethod
    def merge_overlapping_boxes(boxes):
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
