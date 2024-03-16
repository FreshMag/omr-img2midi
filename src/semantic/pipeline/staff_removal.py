import os

import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K

from src.util.imgutils import show_image


def get_model(model_path):
    block_filter_count = [8, 16, 32, 64, 128, 256]

    optimizer = tf.keras.optimizers.legacy.Adam()

    loss_function = 'binary_crossentropy'
    # Creazione del modello
    model = UNet((512, 512, 3), block_filter_count)

    # Configura il modello per il training
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['acc', jaccard_index])
    model.load_weights(model_path)
    return model


def down_block(x, filter_count):
    c = keras.layers.Conv2D(filter_count, 3, padding='same', activation='relu')(x)
    c = keras.layers.Conv2D(filter_count, 3, padding='same', activation='relu')(c)
    p = keras.layers.MaxPool2D(2, 2)(c)
    return c, p


def bottleneck(x, filter_count):
    c = keras.layers.Conv2D(filter_count, 3, padding='same', activation='relu')(x)
    c = keras.layers.Conv2D(filter_count, 3, padding='same', activation='relu')(c)
    return c


def up_block(x, skip, filter_count):
    us = keras.layers.UpSampling2D(2)(x)
    c = keras.layers.Conv2D(filter_count, 2, padding='same')(us)
    concat = keras.layers.Concatenate()([c, skip])
    c = keras.layers.Conv2D(filter_count, 3, padding='same', activation='relu')(concat)
    c = keras.layers.Conv2D(filter_count, 3, padding='same', activation='relu')(c)
    return c


def UNet(image_shape, block_filter_count=None):
    if block_filter_count is None:
        block_filter_count = [64, 128, 256, 512, 1024]
    inputs = keras.layers.Input(image_shape)

    # Downsampling path
    p0 = inputs
    c1, p1 = down_block(p0, block_filter_count[0])
    c2, p2 = down_block(p1, block_filter_count[1])
    c3, p3 = down_block(p2, block_filter_count[2])
    c4, p4 = down_block(p3, block_filter_count[3])

    # Bottleneck
    bn = bottleneck(p4, block_filter_count[4])

    # Upsampling path
    u1 = up_block(bn, c4, block_filter_count[3])
    u2 = up_block(u1, c3, block_filter_count[2])
    u3 = up_block(u2, c2, block_filter_count[1])
    u4 = up_block(u3, c1, block_filter_count[0])

    # Output
    outputs = keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(u4)

    return keras.models.Model(inputs, outputs)


def jaccard_index(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def normalize_image(image):
    img = image.copy()

    if len(image.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    border_v, border_h = (0, 0)
    if img.shape[0] < 512 <= img.shape[1]:
        img = img[:, :512]
        border_v = int((512 - img.shape[0]) / 2) + 1
    elif img.shape[1] < 512 <= img.shape[0]:
        img = img[:512, :]
        border_h = int((512 - img.shape[1]) / 2) + 1
    elif img.shape[0] < 512 and img.shape[1] < 512:
        border_v = int((512 - img.shape[0]) / 2) + 1
        border_h = int((512 - img.shape[1]) / 2) + 1

    img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    img = img[:512, :512]
    img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    img = cv2.resize(img, (512, 512))

    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # show_image(img)
    input = (np.array([img]) / 255)
    input = input.astype(np.float32)
    return input, border_h, border_v


class StaffRemover:
    def __init__(self, model_path):
        self.model = get_model(model_path)

    def remove_staffs(self, image):
        input, border_h, border_v = normalize_image(image)
        pred = self.model.predict(input)

        optimal_bin_thr = 0.6
        pred_bin = (pred > optimal_bin_thr).astype(np.uint8)
        pred_bin = np.squeeze(pred_bin[0], axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

        dilated = cv2.morphologyEx(pred_bin, cv2.MORPH_DILATE, kernel)
        mask = dilated == 1

        # Create a new array filled with black color
        black_color = np.zeros(shape=(512, 512, 3))

        # Apply the mask to retain BGR values where the mask is True
        result = ((np.where(mask[..., None], black_color, input)[0]) * 255).astype(np.uint8)
        _, result = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_not(result)
        # Remove the safety border
        result = cv2.resize(result, (512 + 100, 512 + 100))
        result = result[50:-50, 50:-50, :]
        print("After removing safe border: {0}".format(result.shape))
        h, w, _ = result.shape
        result = result[max(border_v-1, 0):min(h-border_v+1, h), max(border_h-1, 0):min(w-border_h+1, w)]
        print("Result shape is {0}".format(result.shape))
        print("")
        return result
