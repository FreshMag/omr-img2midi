import glob
import os

import cv2 as cv2

from src.semantic.pipeline.object_detection import show
from src.semantic.pipeline.pattern_matching import kp_dense_sampling


class Matcher:
    def __init__(self, input_image_path):
        self.gray = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        self.color = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        show(self.gray)
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()
        self.keypoints = kp_dense_sampling(self.gray, 16, 3.0)
        kp, des = self.sift.detectAndCompute(self.gray, None)
        self.kp = kp
        self.des = des
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50)  # or pass empty dictionary
        self.matches = {}

    def __add_matches(self, matches, class_name):
        if not self.matches.get(class_name):
            self.matches[class_name] = []

        self.matches[class_name].extend(matches)

    def draw_matches(self):
        with_boxes = self.color.copy()
        for class_name, matches in self.matches.items():
            for match in matches:
                x1, y1, x2, y2 = match
                cv2.rectangle(with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                position = (x2, y2)
                font_scale = 1
                color = (255, 0, 0)
                thickness = 2
                # Using cv2.putText() method
                with_boxes = cv2.putText(with_boxes, class_name, position, font,
                                         font_scale, color, thickness, cv2.LINE_AA)

        return with_boxes

    def match(self, template_gray_image, class_name):
        print("Matching {}...".format(class_name))
        # plt.imshow(template_gray_image)
        # plt.show()
        template_kp, template_des = self.sift.detectAndCompute(template_gray_image, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(template_des, self.des, k=2)

        good_matches = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Extract coordinates of keypoints from matches
        # keypoints1 = [kp1[m.queryIdx].pt for m in good_matches]
        keypoints2 = [self.kp[m.trainIdx].pt for m in good_matches]

        boxes = []
        h, w = template_gray_image.shape
        for kp in keypoints2:
            x1, x2, y1, y2 = (int(kp[0] - (w / 2)), int(kp[0] + (w / 2)), int(kp[1] - (h / 2)), int(kp[1] + (h / 2)))
            boxes.append((x1, y1, x2, y2))
            # cv2.rectangle(with_boxes, (int(kp[0] - (w / 2)), int(kp[1] - (h / 2))),
            #              (int(kp[0] + (w / 2)), int(kp[1] + (h / 2))),
            #              (255, 0, 0), 2)
            # plt.imshow(img2[int(kp[1] - (h / 2)):int(kp[1] + (h / 2)), int(kp[0] - (w / 2)):int(kp[0] + (w / 2))])
            # plt.grid(False)
            # plt.show()

        self.__add_matches(boxes, class_name)


def get_symbols(symbols_path, filter=None, class_limit=5, template_limit=50):
    symbol_classes = os.listdir(symbols_path)
    symbols = {}
    i = 0
    for symbol_class in symbol_classes:
        if i >= class_limit:
            break
        if filter is not None and not filter(symbol_class):
            continue
        symbol_path = os.path.join(symbols_path, symbol_class)
        images = glob.glob(symbol_path + "/*.png")[:template_limit]
        symbols[symbol_class] = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in images]
        i += 1
    return symbols


symbol_path = "../../../data/input/patterns/"
symbols = get_symbols(symbol_path) #, filter=lambda symbol:symbol == "NotesOpen" or symbol == "Rests1" or symbol == "TrebleClef")

input_image = "../../../data/input/no_symbols.png"
matcher = Matcher(input_image)
scales = [(1, 1)]
templates = {}

for symbol_name, symbol_images in symbols.items():
    for image in symbol_images:
        show(image)
        rescaled = [cv2.resize(image, (round(scaley * image.shape[0]),
                                       round(scalex * image.shape[1]))) for scaley, scalex in scales]
        for img in rescaled:
            matcher.match(img, symbol_name)

with_matches = matcher.draw_matches()
cv2.imshow("matches", with_matches)
cv2.waitKey(0)
cv2.destroyWindow("matches")
# draw_params = dict(matchColor = (0,255,0),
#                  singlePointColor = (255,0,0),
#                  matchesMask = matchesMask,
#                  flags = cv.DrawMatchesFlags_DEFAULT)

# img3 = cv.drawMatchesKnn(template,kp1,img2,kp2,matches,None,**draw_params)

# plt.imshow(img3,),plt.show()
