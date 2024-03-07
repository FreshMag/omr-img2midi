
class BoundingBox:
    def __init__(self, upper_left, lower_right):
        self.upper_left = upper_left
        self.lower_right = lower_right

    def apply_on_image(self, image):
        return image[self.upper_left[1]:self.lower_right[1], self.upper_left[0]:self.lower_right[0]]