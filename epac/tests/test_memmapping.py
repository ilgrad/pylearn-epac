# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:56:37 2013

@author: laure
"""

from memory_profiler import profile
import numpy as np
import os.path as path
from tempfile import mkdtemp
from sklearn import datasets
import random
import datetime
import sys
import os

def create_mmat(nrows, ncols, default_values=None):
    now = datetime.datetime.now()
    np.random.seed(now.second)
    filename = path.join(mkdtemp(), 'newfile.dat')
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


def create_array(size, default_values=None):
    ret_array = create_mmat(size, 1, default_values)
    ret_array = ret_array[:, 0]
    return ret_array


def convert2memmap(np_mat):
    filename = path.join(mkdtemp(), 'newfile.dat')
    mem_mat = np.memmap(filename,\
                     dtype='float32',\
                     mode='w+',\
                     shape=np_mat.shape)
    mem_mat[:] = np_mat[:]
    return mem_mat


# @profile
def func_memm_local(n_samples, n_features):
    ''' Test the capacity of the computer

    Parameters
    ----------
    n_samples: number of rows of the X matrix

    n_features: number of columns of th X matrix
    '''
    print " ------- memm_local pt1 : beginning with", n_features, "features -------"
    ## 1) Build a np.memmap dataset for big matrix
    ## ============================================================
    X = create_mmat(n_samples, n_features)
    y = create_array(n_samples, [0,1])

    print "X matrix file size =", os.path.getsize(X.filename), "bytes"
    print "y vector file size =", os.path.getsize(y.filename), "bytes"

    Xy = dict(X=X, y=y)
    ## 2) Build two workflows respectively
    ## =======================================================
    print " -> memm_local pt2 : X and y created, building workflow"
    from sklearn.svm import SVC
    from epac import CV, Methods
    cv_svm_local = CV(Methods(*[SVC(kernel="linear"),
                          SVC(kernel="rbf")]),
                          n_folds=3)
    print " -> memm_local pt3 : Workflow built, running"
    cv_svm_local.run(**Xy)
#    from epac import LocalEngine
#    local_engine = LocalEngine(cv_svm_local, num_processes=1)
#    cv_svm = local_engine.run(**Xy)
    print " -> memm_local pt4 : Finished running, reducing"
    cv_svm_local.reduce()
    print " -> memm_local pt5 : finished with", n_features, "features"


# @profile
def func_no_memm_local(n_samples, n_features):
    ''' Test the capacity of the computer

    Parameters
    ----------
    n_samples: number of rows of the X matrix

    n_features: number of columns of th X matrix
    '''
    print " ------- no_memm_local pt1 : beginning with", n_features, "features -------"
    ## 1) Build a np.memmap dataset for big matrix
    ## ============================================================
    X, y = datasets.make_classification(n_samples=n_samples,
                                        n_features=n_features,
                                        n_informative=2,
                                        random_state=1)
    print "X matrix file size =", sys.getsizeof(X), "bytes"

    Xy = dict(X=X, y=y)
    ## 2) Build two workflows respectively
    ## =======================================================
    print " -> no_memm_local pt2 : X and y created, building workflow"
    from sklearn.svm import SVC
    from epac import CV, Methods
    cv_svm_local = CV(Methods(*[SVC(kernel="linear"),
                          SVC(kernel="rbf")]),
                          n_folds=3)
    print " -> no_memm_local pt3 : Workflow built, running"
    cv_svm_local.run(**Xy)
#    from epac import LocalEngine
#    local_engine = LocalEngine(cv_svm_local, num_processes=1)
#    cv_svm = local_engine.run(**Xy)
    print " -> no_memm_local pt4 : Finished running, reducing"
    cv_svm_local.reduce()
    print " -> no_memm_local pt5 : finished with", n_features, "features"


if __name__ == "__main__":
    args = sys.argv[1:]
    func_memm_local(int(args[0]), int(args[1]))