#!/usr/bin/env python

from pyblique import error_rate, get_data, ObliqueClassifier
import impurity
import sklearn
import argparse
import os
import sys
import time


class Tee:

    def __init__(self, *args):
        self.outputs = args

    def __call__(self, s="", end="\n"):
        for o in self.outputs:
            o.write(s + end)


def run(fname, folds):
    st = time.clock()
    data = get_data("Data/{}.data".format(fname))
    with open("Results/{}_{}folds.txt".format(fname, folds), "a") as f:
        tee = Tee(sys.stdout, f)
        tee("Validating classifier with {}-fold test...".format(folds))
        if "cross_validation" in sklearn.__all__:
            from sklearn.cross_validation import KFold
            kf = KFold(len(data), n_folds=folds)
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=folds).split(data)
        avg_error = 0
        it = 1
        for train, test in kf:
            start = time.clock()
            tee("Iteration #{}".format(it))
            oc = ObliqueClassifier(metric=impurity.entropy, min_samples_split=2)
            oc.fit(data[train])
            predictions = [oc.predict(r) for r in data[test]]
            actual_labels = data[test][:, -1]
            error = error_rate(predictions, actual_labels)
            tee("Error: {:.3f}".format(error))
            tee("Elapsed time: {:.3f} seconds".format(time.clock() - start))
            tee()
            avg_error += error
            it += 1
        totaltime = time.clock() - st
        tee("Average error: {:.3f}".format(avg_error/folds))
        tee("Total elapsed time: {:.3f} seconds.".format(totaltime))
        tee("Average elapsed time: {:.3f} seconds.".format(totaltime/folds))


if __name__ == "__main__":
    files = os.listdir("Data")
    files = [f.split(".")[0] for f in files]
    if not os.path.exists("Results"):
        os.makedirs("Results")
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folds", default=5, type=int,
                        help="Amount of folds")
    parser.add_argument("data", type=str, help="Name of dataset",
                        choices=files)

    args = parser.parse_args()
    run(args.data, args.folds)
