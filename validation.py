# Author: Philipp Wagner <bytefish@gmx.de>
# Released to public domain under terms of the BSD Simplified license.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the organization nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#   See <http://www.opensource.org/licenses/bsd-license>

import os
import sys
import cv2
import numpy as np

# from sklearn import cross_validation as cval
from sklearn.model_selection import cross_validate as cval
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is
    given.

    Args: path: Path to a folder with subfolders representing the subjects
    (persons).  sz: A tuple with the size Resizes

    Returns: A list [X,y]

            X: The images, which is a Python list of numpy arrays.  y: The
            corresponding labels (the unique number of the subject, person) in
            a Python list.  """
    c = 0
    X, y = [], []
    for dirname, dirnames, _filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(
                        subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print("I/O error")
                except Exception:
                    print(("Unexpected error:", sys.exc_info()[0]))
                    raise
            c = c+1
    return [X, y]


class FaceRecognizerModel(BaseEstimator):

    def __init__(self):
        self.model = cv2.face.FisherFaceRecognizer_create()

    def fit(self, X, y):
        self.model.train(X, y)

    def predict(self, T):
        #
        # self.model.predict returns (label, confidence) tuple
        #
        return [self.model.predict(T[i])[0] for i in range(0, T.shape[0])]


if __name__ == "__main__":
    # You'll need at least some images to perform the validation on:
    if len(sys.argv) < 2:
        print(
            "USAGE: facerec_demo.py </path/to/images>")
        sys.exit()
    # Read the images and corresponding labels into X and y.
    print("Read  images ....")
    [X, y] = read_images(sys.argv[1])
    # Convert labels to 32bit integers. This is a workaround for 64bit
    # machines, because the labels will truncated else. This is fixed in recent
    # OpenCV revisions already, I just leave it here for people on older
    # revisions.
    #
    # Thanks to Leo Dirac for reporting:
    X = np.asarray(X,  dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    # Then we create a 10-fold cross validation iterator:
    print("splitting the data...")
    skf = StratifiedKFold(n_splits=10)
    cv = skf.split(X, y)
    # Now we'll create a classifier, note we wrap it up in the
    # FaceRecognizerModel we have defined in this file. This is
    # done, so we can use it in the awesome scikit-learn library:
    estimator = FaceRecognizerModel()
    # And getting the precision_scores is then as easy as writing:
    print("cross validating...")
    precision_scores = cross_val_score(estimator, X, y,
                                       scoring=make_scorer(precision_score,
                                                           average='micro'),
                                       cv=cv)
    # Let's print them:
    #
    # return Array of scores of the estimator for each run of the cross
    # validation.
    #
    print("finish")
    print(precision_scores)
