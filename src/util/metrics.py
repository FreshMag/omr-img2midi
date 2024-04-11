import numpy as np
from tabulate import tabulate
from sklearn.metrics import f1_score, jaccard_score


def min_operations(pred, ref):
    m = len(pred)
    n = len(ref)

    # Initialize a 2D array to store the minimum number of operations
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the rest of the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                   dp[i][j - 1],  # insertion
                                   dp[i - 1][j - 1])  # substitution

    # The bottom-right cell of the dp table contains the result
    return dp[m][n]


class Metrics:
    def __init__(self):
        self.jaccard_index = 0
        self.f1_score = 0

    def compute_from_semantics(self, predicted_semantics, expected_semantics):
        pred = np.array(predicted_semantics)
        true = np.array(expected_semantics)
        self.jaccard_index = jaccard_score(pred, true, average='micro')
        self.f1_score = f1_score(pred, true, average='micro')

    def print_table(self):
        print(tabulate([['Jaccard Index', self.jaccard_index],
                        ['F1-Score', self.f1_score]], headers=['Metric name', 'Metric value'], tablefmt='orgtbl'))
