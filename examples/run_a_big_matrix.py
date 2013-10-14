# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:47:17 2013

@author: jinpeng.li@cea.fr
"""

import numpy as np
import os.path as path
from tempfile import mkdtemp
from sklearn import datasets


## 0) Convert an array matrix to memory mapping (np.memmap)
## =======================================================
def convert2memmap(np_mat):
    filename = path.join(mkdtemp(), 'newfile.dat')
    mem_mat = np.memmap(filename,
                        dtype='float32',
                        mode='w+',
                        shape=np_mat.shape)
    mem_mat[:] = np_mat[:]
    return mem_mat

## 1) Build a dataset and convert to np.memmap (for big matrix)
## ============================================================
X, y = datasets.make_classification(n_samples=500,
                                    n_features=10000,
                                    n_informative=2,
                                    random_state=1)
X = convert2memmap(X)
y = convert2memmap(y)

Xy = dict(X=X, y=y)

## 2) Build two workflows respectively
## =======================================================

from sklearn.svm import SVC
from epac import CV, Methods
cv_svm_local = CV(Methods(*[SVC(kernel="linear"),
                            SVC(kernel="rbf")]),
                  n_folds=3)
cv_svm_swf = CV(Methods(*[SVC(kernel="linear"),
                          SVC(kernel="rbf")]),
                n_folds=3)

## 3) Run two workflows using local engine and soma-workflow
## =========================================================

from epac import LocalEngine
local_engine = LocalEngine(cv_svm_local, num_processes=2)
cv_svm = local_engine.run(X=X, y=y)
print cv_svm.reduce()

from epac import SomaWorkflowEngine
swf_engine = SomaWorkflowEngine(cv_svm_swf,
                                num_processes=2,
                                #resource_id="jl237561@gabriel",
                                #login="jl237561",
                                remove_finished_wf=False)
cv_svm = swf_engine.run(**Xy)
print cv_svm.reduce()
