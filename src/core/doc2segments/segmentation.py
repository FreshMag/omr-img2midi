import cv2

from .clustering import cluster_bboxes
from src.util.imgutils import show_image


def segment_doc(doc):

    bboxes = cluster_bboxes(doc)

    img_with_bboxes = cv2.cvtColor(doc.copy(), cv2.COLOR_GRAY2BGR)
    for bbox in bboxes:
        bbox.draw_on_image(img_with_bboxes)

    show_image(img_with_bboxes)
    return []
