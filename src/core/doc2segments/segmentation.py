import warnings

import cv2
import numpy as np
from util import show_image

from .clustering import cluster_bboxes


def segment_doc(doc):
    total_area = doc.shape[0] * doc.shape[1]
    segments = [bbox.apply_on_image(doc) for bbox in cluster_bboxes(doc, min_cluster_area=total_area / 300)]
    return split_segments(segments)


def optimal_half_split(segment, threshold=0.1):
    pixel_threshold = threshold * (segment.shape[0] * 10)
    bin_segment = cv2.threshold(segment, 127, 255, cv2.THRESH_OTSU)[1]
    bin_segment = cv2.bitwise_not(bin_segment)
    width = segment.shape[1]
    center_x = width // 2
    optimal_x = -1

    def is_optimal(x):
        if x >= width or x < 0:
            return False
        num_pixels_left = cv2.countNonZero(bin_segment[:, x - 5:x + 5])
        return num_pixels_left < pixel_threshold

    for offset in range(center_x + 1):
        left = center_x - offset
        if is_optimal(left):
            optimal_x = left
            break
        right = center_x + offset
        if is_optimal(right):
            optimal_x = right
            break

    if optimal_x <= 0 or optimal_x >= width - 1:
        warnings.warn("Optimal half split is not possible")
        show_image(bin_segment)
    # else:
    #show_image(cv2.line(bin_segment, (optimal_x, 0), (optimal_x, segment.shape[0]), (255, 0, 0), 2))
    left_part = segment[:, :optimal_x]
    right_part = segment[:, optimal_x:]

    return left_part, right_part


def smallest_boxed(image):
    assert len(image.shape) == 2
    ys = np.where(np.any(image == 0, axis=1))
    y_min, y_max = np.min(ys), np.max(ys)
    xs = np.where(np.any(image == 0, axis=0))
    x_min, x_max = np.min(xs), np.max(xs)
    return image[y_min:y_max, x_min:x_max]


def add_safety_border(image, border=30):
    assert len(image.shape) == 2
    return cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=(255,))


def is_binary(image):
    return np.all(np.logical_or(image == 0, image == 255), axis=None)


def unsharpen(image):
    assert len(image.shape) == 2
    return cv2.GaussianBlur(image, (1, 1), 0.25)


def split_segments(segments):
    splitted = []
    for segment in segments:
        shape = segment.shape
        if len(shape) != 2:
            raise ValueError(f"Segment shape {shape} does not have 2 dimensions. Please provide grayscale 2D segments")
        h, w = shape
        if h > w:
            warnings.warn("Ignoring one segment which is vertical. Size (%d, %d)" % (w, h))
            continue
        seg1, seg2 = optimal_half_split(segment)
        if is_binary(segment):
            seg1, seg2 = (unsharpen(add_safety_border(smallest_boxed(seg1))), unsharpen(add_safety_border(smallest_boxed(seg2))))
        splitted.append(seg1)
        splitted.append(seg2)
        #show_image(seg1)
        #show_image(seg2)
    return splitted
