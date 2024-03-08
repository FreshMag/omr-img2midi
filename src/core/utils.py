import cv2


def clean_doc(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed
