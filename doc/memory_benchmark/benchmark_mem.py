# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:47:17 2013

@author: jinpeng.li@cea.fr

python -m memory_profiler benchmark_mem.py
"""

from memory_profiler import profile
import numpy as np
import os.path as path
from tempfile import mkdtemp
from sklearn import datasets


def convert2memmap(np_mat):
    filename = path.join(mkdtemp(), 'newfile.dat')
    mem_mat = np.memmap(filename,
                        dtype='float32',
                        mode='w+',
                        shape=np_mat.shape)
    mem_mat[:] = np_mat[:]
    return mem_mat


@profile
def func_memm_local():
    print("memm_local pt1")
    ## 1) Build a dataset and convert to np.memmap (for big matrix)
    ## ============================================================
    X, y = datasets.make_classification(n_samples=500,
                                        n_features=5000,
                                        n_informative=2,
                                        random_state=1)
    print("memm_local pt2")
    X = convert2memmap(X)
    y = convert2memmap(y)
    Xy = dict(X=X, y=y)
    ## 2) Build two workflows respectively
    ## =======================================================
    print("memm_local pt3")
    from sklearn.svm import SVC
    from epac import CV, Methods
    cv_svm_local = CV(Methods(*[SVC(kernel="linear"),
                                SVC(kernel="rbf")]),
                      n_folds=3)
    print("memm_local pt4")
#    from epac import LocalEngine
#    local_engine = LocalEngine(cv_svm_local, num_processes=2)
#    cv_svm = local_engine.run(**Xy)
    cv_svm_local.run(**Xy)
    print(cv_svm_local.reduce())
    print("memm_local pt5")


@profile
def func_no_memm_local():
    print("no_memm_local pt1")
    ## 1) Build a dataset and convert to np.memmap (for big matrix)
    ## ============================================================
    X, y = datasets.make_classification(n_samples=500,
                                        n_features=5000,
                                        n_informative=2,
                                        random_state=1)
    print("no_memm_local pt2")
    Xy = dict(X=X, y=y)
    ## 2) Build two workflows respectively
    ## =======================================================
    print("no_memm_local pt3")
    from sklearn.svm import SVC
    from epac import CV, Methods
    cv_svm_local = CV(Methods(*[SVC(kernel="linear"),
                                SVC(kernel="rbf")]),
                      n_folds=3)
    print("no_memm_local pt4")
    cv_svm_local.run(**Xy)
    print(cv_svm_local.reduce())
    print("no_memm_local pt5")

# func_no_memm_local()
func_memm_local()
