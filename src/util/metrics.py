"""

This file contains a series of calculations to evaluate metrics. It has also been used to obtain optimal parameters
for the various algorithms. Here documentation and code are incomplete and are not be considered core of the project.

"""
import glob
import itertools
import os
import warnings

import numpy as np
from tqdm import tqdm

import cv2
import tabulate
from core.doc2segments.segmentation import segment_doc
from core.img2doc.scanner import scan, light_scan
from semantic import EncodedSheet, CTC
from util.one_pass import end2end_recognition


def symbol_error_rate(pred, ref):
    m = len(pred)
    n = len(ref)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                   dp[i][j - 1],  # insertion
                                   dp[i - 1][j - 1])  # substitution

    return dp[m][n] / max(m, n)


def jaccard_index(ref, pred):
    intersection = len(list(set(pred).intersection(ref)))
    union = (len(pred) + len(ref)) - intersection
    # print("Length Prediction: %d\nLength Reference: %d" % (len(pred),len(ref)))
    return float(intersection) / union


def get_test_set(directory, filter_files=None):
    filetypes = (".jpg", ".jpeg", ".png")
    images = []
    for type in filetypes:
        images.extend(glob.glob(os.path.join(directory, "*" + type)))

    test_set = {}

    # We want this kind of structure
    # {
    #     "mickeyzelda": {
    #         "scan" : "./input/test/mickeyzelda_scan.jpeg",
    #         "topnormal": "./input/test/mickeyzelda_topnormal.jpeg",
    #         "size": 5,
    #         "ref": "./input/test/mickeyzelda_ref.txt",
    #     }
    # }
    if filter is not None:
        images = filter(filter_files, images)
    for image in images:
        comp = os.path.basename(image).split("_")
        if len(comp) == 2:
            name, condition = comp[0], comp[1].split(".")[0]
        else:
            name, condition = comp[0], comp[1] + "-" + comp[2].split(".")[0]
        # print("Taking %s in condition %s" % (name, condition))
        if name in test_set:
            test_set[name][condition] = image
        else:
            test_set[name] = {condition: image}
            ref_file_path = os.path.join(os.path.dirname(image), name + "_ref.txt")
            if os.path.exists(ref_file_path):
                test_set[name]["ref"] = ref_file_path
            else:
                warnings.warn("Could not find reference file for %s" % name)
                # raise FileNotFoundError("Could not find reference file for %s" % name)

    return test_set


def show_image(img, title="Title"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_statistics(results, conditions):
    to_tabulate = []
    # print(results)
    for name in results:
        to_tabulate.append(
            [name + "_scanned", results[name]["ref-scan-jaccard"], "", results[name]["ref-scan-symbolerror"], ""])
        for condition in conditions:
            to_tabulate.append([name + "_" + condition,
                                results[name]["ref-" + condition + "-jaccard"],
                                results[name]["scan-" + condition + "-jaccard"],
                                results[name]["ref-" + condition + "-symbolerror"],
                                results[name]["scan-" + condition + "-symbolerror"]])

    # print(results)
    print(tabulate.tabulate(to_tabulate, headers=["Name of the sheet", "Jaccard Index(with reference)",
                                                  "Jaccard Index(with scanned)", "Symbol Error Rate(with reference)",
                                                  "Symbol Error Rate(with scanned)"], tablefmt="fancy_grid"))


def print_means(results, backgrounds, log=lambda x: print(x)):
    print("\nComputing means")
    backgrounds.append("scan")
    sums_jaccard = np.zeros(len(backgrounds))
    counts_jaccard = np.zeros(len(backgrounds))
    sums_symbolerror = np.zeros(len(backgrounds))
    counts_symbolerror = np.zeros(len(backgrounds))

    for name in results:
        for condition in filter(lambda x: "ref" in x, results[name].keys()):
            cond = condition.split("-")
            if len(cond) == 3:
                _, c, metric = cond
            else:
                _, c, hd, metric = cond
                c = c + hd
            for i in range(len(backgrounds)):
                back = backgrounds[i]
                if back in c:
                    if metric == "jaccard":
                        sums_jaccard[i] = sums_jaccard[i] + results[name][condition]
                        counts_jaccard[i] = counts_jaccard[i] + 1
                    elif metric == "symbolerror":
                        sums_symbolerror[i] = sums_symbolerror[i] + results[name][condition]
                        counts_symbolerror[i] = counts_symbolerror[i] + 1

    to_tabulate = []
    for i in range(len(backgrounds)):
        to_tabulate.append([backgrounds[i], float(sums_jaccard[i]) / (counts_jaccard[i]),
                            float(sums_symbolerror[i]) / counts_symbolerror[i]])

    log(tabulate.tabulate(to_tabulate,
                          headers=["Background", "Jaccard Index(with reference)",
                                   "Symbol Error Rate(with reference)"],
                          tablefmt="fancy_grid"))
    return to_tabulate


def test_set_with_metrics(parameters, directory="./", vocabulary_file_path="../../data/vocabulary_semantic.txt",
                          model_path="../../models/semantic/semantic_model.meta"):
    test_set = get_test_set(directory, filter_files=lambda x: "bona122" not in x and "bona90" not in x)

    model = CTC(model_path)

    def img2sheet(image_path, to_scan=True, to_print=False):
        print("Reading %s" % image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        if to_scan:
            img = scan(img,
                       thresh_block_size_ratio=parameters["thresh_block_ratio"],
                       thresh_c_ratio=parameters["thresh_c_ratio"],
                       component_pixel_thresh_ratio=parameters["component_thresh_ratio"])
        else:
            img = light_scan(img)
        if to_print:
            cv2.imshow(image_path, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        segments = segment_doc(img)
        #for segment in segments:
        #show_image(segment)
        sheet, _ = end2end_recognition(segments, model=model, vocabulary_path=vocabulary_file_path)
        return sheet

    cameras = ["front", "left", "right", "top", "bottom"]
    backgrounds = ["normal", "shadow"]  # , "dark"]
    quality = ["-hd"]
    conditions = [comb[0] + comb[1] + comb[2] for comb in itertools.product(cameras, backgrounds, quality)]
    results = {}

    for name in tqdm(test_set.keys()):
        case = test_set[name]
        true = EncodedSheet(vocabulary_file_path=vocabulary_file_path)
        try:
            true.add_from_semantic_file(case["ref"], file_format="symbol", separator="\n")
        except Exception as e:
            print(e)
            warnings.warn(f"Ignoring %s because of empty reference file for ref: %s" % (name, case["ref"]))
            continue

        results[name] = {}
        # print("==== Calculating %s %s" % (name, "scan"))
        # print("==== Reference file: %s" % case["ref"])
        scanned = img2sheet(case["scan"], to_scan=False)
        # scanned.write_to_file("../../data/output/%s.txt" % (os.path.basename(case["scan"])), output_format="symbol")
        results[name]["ref-scan-jaccard"] = jaccard_index(true.output_indexes, scanned.output_indexes)
        results[name]["ref-scan-symbolerror"] = symbol_error_rate(true.output_indexes, scanned.output_indexes)

        for condition in tqdm(case.keys()):
            if condition in conditions:
                # print("==== Calculating %s %s" % (name, condition))
                # print("==== Reference file: %s" % case["ref"])

                computed_sheet = img2sheet(case[condition], to_scan=True)
                # computed_sheet.write_to_file("../../data/output/%s.txt" % (os.path.basename(case[condition])), output_format="symbol")
                results[name]["ref-" + condition + "-jaccard"] = jaccard_index(true.output_indexes,
                                                                               computed_sheet.output_indexes)
                # print("==== Comparing with Scan image: %s" % case["scan"])
                results[name]["scan-" + condition + "-jaccard"] = jaccard_index(scanned.output_indexes,
                                                                                computed_sheet.output_indexes)
                results[name]["ref-" + condition + "-symbolerror"] = symbol_error_rate(true.output_indexes,
                                                                                       computed_sheet.output_indexes)
                results[name]["scan-" + condition + "-symbolerror"] = symbol_error_rate(scanned.output_indexes,
                                                                                        computed_sheet.output_indexes)

    print_statistics(results, conditions)

    return print_means(results, backgrounds)


def optimization():
    search_space = {"component_thresh_ratio": [0.00001, 0.000015], "thresh_block_ratio": [0.01224, 0.01356, 0.01488],
                    "thresh_c_ratio": [0.0033, 0.0066, 0.0099]}
    optimal = None
    min_error = float('inf')

    for component_thresh in search_space["component_thresh_ratio"]:
        for thresh_size in search_space["thresh_block_ratio"]:
            for thresh_c in search_space["thresh_c_ratio"]:
                param = {"component_thresh_ratio": component_thresh,
                         "thresh_block_ratio": thresh_size,
                         "thresh_c_ratio": thresh_c}
                print("Testing %s" % param)
                result = test_set_with_metrics(parameters=param, directory="../../data/input/test/")
                means_symbol_error = np.array(result)[:, 2:]
                means_symbol_error[means_symbol_error == ''] = 0.0
                means_symbol_error = means_symbol_error.astype(np.float32)
                error_sum = np.sum(means_symbol_error)

                if error_sum < min_error:
                    min_error = error_sum
                    optimal = param

    print("Optimal: %s" % optimal)
    print("Min. error: %f" % min_error)


if __name__ == "__main__":
    # optimization()
    test_set_with_metrics(parameters={'component_thresh_ratio': 2e-05,
                                      'thresh_block_ratio': 0.018,
                                      'thresh_c_ratio': 0.0050},
                          directory="../../data/input/test/")
