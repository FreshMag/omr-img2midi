import warnings

import cv2
import numpy as np

from .clustering import cluster_bboxes


def segment_doc(doc):
    """
    The main function of doc2segments. It takes a document scanned with img2doc and returns the list of segments
    inside the doc
    :param doc: The scanned document
    :return: The list of segments, sorted according to theie position in the music sheet
    """
    segments = [bbox.apply_on_image(doc) for bbox in cluster_bboxes(doc)]
    return list(reversed(split_segments(segments)))


def optimal_half_split(segment, threshold=0.1):
    """
    This function finds the optimal half split of a segment. Optimal is defined as the vertical line that encounters the
    minimum number of black pixel (music symbol) as near as possible to the center of the segment.
    :param segment: The segment that needs to be split
    :param threshold: A value used as threshold to determine if the split is optimal. It is multiplied for the height of
    the segment
    :return: Two images, left and right side of the segment respect to the found optimal split
    """
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

    left_part = segment[:, :optimal_x]
    right_part = segment[:, optimal_x:]

    return left_part, right_part


def smallest_boxed(image):
    """
    This function finds the smallest bounding box containing all the black pixels (symbols) of a given image
    :param image: to be analyzed
    :return: Portion of the image determined by the found bounding box
    """
    assert len(image.shape) == 2
    ys = np.where(np.any(image == 0, axis=1))
    y_min, y_max = np.min(ys), np.max(ys)
    xs = np.where(np.any(image == 0, axis=0))
    x_min, x_max = np.min(xs), np.max(xs)
    return image[y_min:y_max, x_min:x_max]


def add_safety_border(image, border=30):
    """
    This function adds the safety border of a given image
    :param image: to add safety border
    :param border: size in pixel of the border
    :return: a new image with safety border
    """
    assert len(image.shape) == 2
    return cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=(255,))


def is_binary(image):
    """
    This function checks if the image is binary or not
    :param image: to be analyzed
    :return: true if the image is binary or false otherwise
    """
    return np.all(np.logical_or(image == 0, image == 255), axis=None)


def small_blur(image):
    """
    This function applies a small amount of blur the given image to "un-sharpen" the image
    :param image: to be blurred
    :return: the blurred image
    """
    assert len(image.shape) == 2
    return cv2.GaussianBlur(image, (1, 1), 0.25)


def split_segments(segments):
    """
    This function splits the given segments into two segments using the ``optimal_half_split`` function.
    It also checks for segment shapes, ignoring for example vertical segments mistakenly found.
    :param segments: iterable of segments (images)
    :return: the new list with each segment split in half (double sized)
    """
    split = []
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
            seg1, seg2 = (
                small_blur(add_safety_border(smallest_boxed(seg1))), small_blur(add_safety_border(smallest_boxed(seg2)))
            )
        split.append(seg2)
        split.append(seg1)
    return split
