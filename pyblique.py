# -*- coding: utf-8 -*-
# Koen Dercksen - 4215966

from random import randint, random
import impurity
import numpy as np
import sys


def get_data(fname):
    try:
        data = np.genfromtxt(fname, comments="#", delimiter=",", dtype=float)
        np.random.shuffle(data)
        return data
    except FileNotFoundError:
        sys.stderr.write("{} does not exist! Aborting.\n".format(fname))
        sys.exit(2)


class ObliqueClassifier:
    """Oblique classifier. Can be trained on a dataset and be used to
    predict unseen records.

    Currently, the classifier only uses the gini index as a metric.
    """

    def __init__(self, metric=impurity.gini, data=None, num_perturb=10, pr_perturb_equal=0.3, min_samples_split=2):
        """Metric can only be a minimizing function!
        """
        self.num_perturb = num_perturb
        self.min_samples_split = min_samples_split
        self.pr_perturb_equal = pr_perturb_equal
        self.num_classes = 0
        # optional add:
        #   other stopping criteria like max_depth, min_samples_leaf, min_impurity_decrease, min_impurity_split
        if data:
            self.fit(data)
        self.metric = metric
        self.tree = {}

    def fit(self, data):
        """
        :param data: float numpy array with dimensions (number of samples), (number of attributes +1)
            data[:, -1] are considered as labels from the set {1., 2., 3., ..., max(data[:, -1])}
        """
        self.num_classes = max(data[:, -1])
        self.tree = self.__create_decision_tree(data)

    def predict(self, record):
        cls = self.tree
        while type(cls) is dict:
            splitv = cls["split"]
            v = self.__checkrel(record, splitv) > 0
            if v:
                cls = cls["high"]
            else:
                cls = cls["low"]
        return cls

    def __create_decision_tree(self, data):
        if len(data) == 0:
            return -1
        isleaf, leaf = self.__is_leaf_node(data)
        if isleaf:
            return leaf
        else:
            splits = self.__get_all_splits(data)
            index, split = min(enumerate(splits), key=lambda x: x[1][1])
            # in order to make this oblique, we first have to build a vector
            # to enable the linear combination split
            sv = np.zeros((len(data[0]),))
            sv[-1] = -split[0]
            sv[index] = 1
            low, high = self.__split_data(data, sv)
            imp = self.metric(low[:, -1]) + self.metric(high[:, -1])
            # perturb a random attribute in split vector 20 times
            for c in range(self.num_perturb):
                r = randint(0, len(sv) - 1)
                imp, sv = self.__perturb(data, sv, r, imp)
            tree = {"split": sv}
            low, high = self.__split_data(data, sv)
            if len(low) > self.min_samples_split and len(high) > self.min_samples_split:
                # stop splitting when one of the subtrees has less than {min_samples_split} samples
                subtree_low = self.__create_decision_tree(low)
                subtree_high = self.__create_decision_tree(high)
            else:
                values, counts = np.unique(data[:, -1], return_counts=True)
                return values[np.argmax(counts)]
            tree["low"] = subtree_low
            tree["high"] = subtree_high
            tree["class_hits_low"] = self.__get_hits(low[:, -1])
            tree["class_hits_high"] = self.__get_hits(high[:, -1])
        return tree

    def __get_hits(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        out = np.zeros(self.num_classes)
        for i, label in enumerate(values):
            out[int(label)-1] = counts[i]
        return out

    def __get_splits(self, data, attr):
        attr_vals = np.sort(data[:, attr])
        weights = np.repeat(1.0, 2) / 2
        return np.convolve(attr_vals, weights)[1:-1]

    def __checkrel(self, record, splitv):
        return np.sum(np.multiply(record[:-1], splitv[:-1])) + splitv[-1]

    def __calc_u(self, record, splitv, attr):
        am = splitv[attr]
        top = am * record[attr] - self.__checkrel(record, splitv)
        return top / record[attr]

    def __perturb(self, data, splitv, attr, old_imp):
        # first calculate all values of U with the current value in splitv
        # for attr
        us = np.array(sorted([[self.__calc_u(r, splitv, attr)] for r in data]))
        possplits = self.__get_splits(us, 0)
        # now find the best of these splits...
        amvalues = {}
        for s in possplits:
            newsplitv = np.array(splitv)
            newsplitv[attr] = s
            low, high = self.__split_data(data, newsplitv)
            imp = self.metric(low[:, -1]) + self.metric(high[:, -1])
            amvalues[s] = (imp, newsplitv)
        bestnewimp, bestnewsplit = min(amvalues.values(), key=lambda x: x[0])
        if bestnewimp > old_imp:
            return bestnewimp, bestnewsplit
        elif bestnewimp == old_imp:
            if random() < self.pr_perturb_equal:
                return bestnewimp, bestnewsplit
        return old_imp, splitv

    def __best_split_on_attr(self, data, attr):
        # Will return a tuple of (split test, split value).
        split_values = self.__get_splits(data, attr)
        split_evals = {}
        for s in split_values:
            cond = data[:, attr] <= s
            left, right = data[cond], data[~cond]
            split_evals[s] = self.metric(left[:, -1]) + self.metric(right[:, -1])
        # Minimize because we're using gini index
        return min(split_evals.items(), key=lambda x: x[1])

    def __get_all_splits(self, data):
        n_attrs = data.shape[1] - 1
        result = [self.__best_split_on_attr(data, i) for i in range(n_attrs)]
        return np.array(result)

    def __split_data(self, data, splitv):
        high = np.zeros(data.shape)
        low = np.zeros(data.shape)
        ihigh, ilow = 0, 0
        for record in data:
            v = self.__checkrel(record, splitv) > 0
            if v:
                high[ihigh] = record
                ihigh += 1
            else:
                low[ilow] = record
                ilow += 1
        high = high[~np.all(high == 0, axis=1)]
        low = low[~np.all(low == 0, axis=1)]
        return low, high

    def __is_leaf_node(self, data):
        # Returns true/false and the class label (useful if this was a leaf)
        labels = data[:, -1]
        label_all = labels[0]
        return all(label == label_all for label in labels), label_all


def error_rate(predictions, labels):
    if len(predictions) != len(labels):
        sys.stderr.write("Incorrect array sizes ({} vs {}) please input evenly"
                         "sized arrays!".format(len(predictions), len(labels)))
    incorrect = 0
    for p, l in zip(predictions, labels):
        if p != l:
            incorrect += 1
    return incorrect/len(labels)
