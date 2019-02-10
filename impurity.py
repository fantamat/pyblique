# Module containing impurity metrics.

import numpy as np
import sys


def prob(data_labels):
    if len(data_labels) == 0:
        return []
    try:
        values, counts = np.unique(data_labels, return_counts=True)
        return counts/np.sum(counts)
    except TypeError:
        sys.stderr.write("Please use Numpy arrays!")
        sys.exit()


def gini(data_labels):
    freq = prob(data_labels)
    fs = np.square(freq)
    return 1 - np.sum(fs)


def entropy(data_labels):
    freq = prob(data_labels)
    return -np.sum(freq*np.log2(freq))

