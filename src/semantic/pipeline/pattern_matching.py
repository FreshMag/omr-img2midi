import cv2
import numpy as np
from tqdm import tqdm

from src.semantic.pipeline.object_detection import show


def kp_dense_sampling(img, spacing, scale):
    points = []
    s = img.shape
    points_per_col = int(s[0] / spacing)
    points_per_row = int(s[1] / spacing)
    half_spacing = spacing / 2
    borderX = (s[1] - (points_per_row - 1) * spacing) / 2
    borderY = (s[0] - (points_per_col - 1) * spacing) / 2

    posY = borderY
    for y in range(points_per_col):
        if y % 2 == 0:
            posX = borderX
        else:
            posX = half_spacing + borderX
        for x in range(points_per_row):
            if s[1] - borderX >= posX >= 0 and s[0] - borderY >= posY >= 0:
                p = cv2.KeyPoint(posX, posY, scale)
                points.append(p)
                posX += spacing
        posY += spacing
    return points


bin_count = 20
colorhist_intersection_thr = 1.1


def compute_hist(img):
    img_YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    pixel_count = img.shape[0] * img.shape[1]
    channels = [1, 2]
    hists = [cv2.calcHist([img_YCbCr], [c], None, [bin_count], [0, 256]) / pixel_count for c in channels]
    hist = np.concatenate((hists[0], hists[1]), axis=0)
    return hist


def check_color_similarity(product_hist, img):
    img_hist = compute_hist(img)
    if np.minimum(product_hist, img_hist).sum() > colorhist_intersection_thr:
        return True
    return False


def compute_avg_dist(d1, d2):
    dist = 0.0
    for i in range(d1[1].shape[0]):
        dist += np.linalg.norm(d1[1][i] - d2[1][i])
    return dist / (d1[1].shape[0] * d1[1].shape[1])


def find_product_candidates(product_template, img, keypoints, scales):
    window_step_perc = 0.1
    dist_thr = 2.5
    s = img.shape
    p_shape = product_template.shape
    scaled_images = [cv2.resize(img, (round(s[1] * scale), round(s[0] * scale)), interpolation=cv2.INTER_CUBIC)
                     for scale in scales]

    y_step = int(round(p_shape[0] * window_step_perc))
    x_step = int(round(p_shape[1] * window_step_perc))

    candidates = []
    sift = cv2.SIFT_create()
    p_hist = compute_hist(product_template)
    product_descriptors = sift.compute(cv2.cvtColor(product_template, cv2.COLOR_BGR2GRAY), keypoints)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for s in tqdm(range(len(scales))):
        i_shape = scaled_images[s].shape
        for y in range(0, i_shape[0] - p_shape[0], y_step):
            for x in range(0, i_shape[1] - p_shape[1], x_step):
                patch = scaled_images[s][y:y + p_shape[0], x:x + p_shape[1]]
                if check_color_similarity(p_hist, patch):
                    patch_descriptors = sift.compute(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), keypoints)
                    dist = compute_avg_dist(product_descriptors, patch_descriptors)
                    if dist < dist_thr:
                        r = {"top": float(y) / scales[s], "bottom": float(y + p_shape[0]) / scales[s],
                             "left": float(x) / scales[s], "right": float(x + p_shape[1]) / scales[s], "dist": dist}
                        candidates.append(r)
    return candidates


if __name__ == "__main__":
    product_template = cv2.imread("../../../data/input/patterns/Note/symbol1.png")
    show(product_template)
    input_image = cv2.imread("../../../data/input/test14.png")
    show(input_image)
    keypoints = kp_dense_sampling(product_template, 16, 3.0)
    scales = [1]
    # Ricerca dei candidati
    candidates = find_product_candidates(product_template, input_image, keypoints, scales)
    # Visualizzazione dei candidati individuati
    initial_candidates = input_image.copy()
    for c in candidates:
        cv2.rectangle(initial_candidates, (round(c["left"]), round(c["top"])), (round(c["right"]), round(c["bottom"])),
                      (0, 0, 255))
    show(initial_candidates)
