"""
This file contains a collection of utility functions used by ctc_predict.py

All the code here was developed by Jorge Calvo Zaragoza <https://github.com/calvozaragoza> and can be found also
on the original GitHub repository <https://github.com/OMR-Research/tf-end-to-end>
"""
import cv2


def sparse_tensor_to_strs(sparse_tensor):
    indices = sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [[] for _ in range(dense_shape[0])]

    string = []
    ptr = 0
    b = 0

    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]

        string.append(values[ptr])

        ptr = ptr + 1

    strs[b] = string

    return strs


def normalize(image):
    return (255. - image) / 255.


def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img
