# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:47:17 2013

@author: jinpeng.li@cea.fr
"""

import numpy as np
import os.path as path
from tempfile import mkdtemp
from sklearn import datasets


def convert2memmap(np_mat):
    filename = path.join(mkdtemp(), 'newfile.dat')
    mem_mat = np.memmap(filename,\
                     dtype='float32',\
                     mode='w+',\
                     shape=np_mat.shape)
    mem_mat[:] = np_mat[:]
    return mem_mat


X, y = datasets.make_classification(n_samples=12,
                                    n_features=10,
                                    n_informative=2,
                                    random_state=1)
X = convert2memmap(X)
y = convert2memmap(y)

from sklearn.svm import SVC
from epac import CV, Methods
cv_svm = CV(Methods(*[SVC(kernel="linear"),
                      SVC(kernel="rbf")]),
                      n_folds=3)

from epac import LocalEngine
local_engine = LocalEngine(cv_svm, num_processes=2)
cv_svm = local_engine.run(X=X, y=y)
print cv_svm.reduce()

cv_svm2 = CV(Methods(*[SVC(kernel="linear"),
                      SVC(kernel="rbf")]),
                      n_folds=3)
cv_svm2.run(X=X, y=y)


import copy

leaf_res1 = []
for leaf1 in cv_svm.walk_leaves():
    res = copy.copy(leaf1.load_results())
    leaf_res1.append(res)

leaf_res2 = []
for leaf2 in cv_svm.walk_leaves():
    res = copy.copy(leaf2.load_results())
    leaf_res2.append(res)


for i in range(len(leaf_res1)):
    for key in leaf_res1[i][leaf_res1[i].keys()[0]].keys():
        print np.all(leaf_res1[i][leaf_res1[i].keys()[0]][key]
            == leaf_res2[i][leaf_res2[i].keys()[0]][key])

