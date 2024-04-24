import math

import numpy as np
from util.imgutils import show_image


def divide_in_chunks(image, chunk_size=512):
    img_chunks = []
    for x in range(0, image.shape[1], chunk_size):
        for y in range(0, image.shape[0], chunk_size):
            y2 = min(y + chunk_size, image.shape[0])
            x2 = min(x + chunk_size, image.shape[1])
            img_chunk = image[y:y2, x:x2]
            img_chunks.append(img_chunk)

    return img_chunks


def reassemble_image(chunks, width, height, chunk_size):
    rows = (height // chunk_size) + 1

    # Initialize the assembled image
    assembled_image = np.empty((height, 0, 3), dtype=np.uint8)
    i = 0
    while i < len(chunks):
        column = chunks[i]
        j = i + 1
        while j % rows != 0:
            column = np.concatenate((column, chunks[j]), axis=0)
            j += 1
        assembled_image = np.hstack((assembled_image, column))
        i += rows

    return assembled_image
