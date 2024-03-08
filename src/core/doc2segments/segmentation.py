from .clustering import cluster_bboxes


def segment_doc(doc):
    return [bbox.apply_on_image(doc) for bbox in cluster_bboxes(doc)]
