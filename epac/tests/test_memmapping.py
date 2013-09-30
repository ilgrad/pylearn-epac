# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:56:37 2013

@author: laure
"""

import numpy as np
from sklearn import datasets
import random
import datetime
import sys
import os
import tempfile


def create_mmat(nrows, ncols, default_values=None, dir=None):
    now = datetime.datetime.now()
    np.random.seed(now.second)
    hfile = tempfile.NamedTemporaryFile("w+", dir=dir)
    filename = hfile.name
    hfile.close()
    mem_mat = np.memmap(filename,\
                     dtype='float32',\
                     mode='w+',\
                     shape=(nrows, ncols))
    for i in xrange(nrows):
        if not default_values:
            mem_mat[i, :] = np.random.random(size=ncols)
        elif type(default_values) is list:
            insert_row = np.zeros(ncols)
            for j in xrange(len(insert_row)):
                pos = random.randint(0, len(default_values) - 1)
                insert_row[j] = default_values[pos]
            mem_mat[i, :] = insert_row
    return mem_mat


def create_array(size, default_values=None, dir=None):
    ret_array = create_mmat(size, 1, default_values, dir=dir)
    ret_array = ret_array[:, 0]
    return ret_array


# @profile
def func_memm_local(n_samples, n_features, memmap, n_proc):
    ''' Test the capacity of the computer

    Parameters
    ----------
    n_samples: number of rows of the X matrix

    n_features: number of columns of th X matrix

    memmap: if True, use memory mapping to reduce memory cost

    n_proc: number of processes
    '''

    ## 1) Building dataset
    ## ============================================================
    print " -> Pt1 : Beginning with", n_features, "features, memmap =",\
        memmap, ",", n_proc, "processes"
    if memmap:
        X = create_mmat(n_samples, n_features, dir="/volatile")
        y = create_array(n_samples, [0, 1], dir="/volatile")

        print "X matrix file size =", os.path.getsize(X.filename), "bytes"

        Xy = dict(X=X, y=y)
    else:

        X, y = datasets.make_classification(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=2,
                                            random_state=1)

        Xy = dict(X=X, y=y)
    ## 2) Build two workflows respectively
    ## =======================================================
    print " -> Pt2 : X and y created, building workflow"
    from sklearn.svm import SVC
    from epac import CV, Methods
    cv_svm_local = CV(Methods(*[SVC(kernel="linear"),
                          SVC(kernel="rbf")]),
                          n_folds=3)
    print " -> Pt3 : Workflow built, running"

#    # Single process on local engine
#    cv_svm_local.run(**Xy)
#    print " -> Pt4 : Finished running single-process, reducing"
#    cv_svm_local.reduce()

    # Multiple processes on local engine
    from epac import LocalEngine
    local_engine = LocalEngine(cv_svm_local, num_processes=n_proc)
    cv_svm = local_engine.run(**Xy)
    print " -> Pt4 : Finished running multi-processes, reducing"
    cv_svm.reduce()

    print " -> Pt5 : Finished with", n_features, "features"


if __name__ == "__main__":
    args = sys.argv[1:]
#    args = [500, 70000, 'True', 1]
    args[2] = (args[2] == 'True')
    func_memm_local(int(args[0]), int(args[1]), args[2], int(args[3]))
