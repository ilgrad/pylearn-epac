# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:47:17 2013

@author: jinpeng.li@cea.fr
"""

import numpy as np
import os
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


X, y = datasets.make_classification(n_samples=50,
                                    n_features=10000,
                                    n_informative=2,
                                    random_state=1)
X = convert2memmap(X)
y = convert2memmap(y)

Xy = dict(X=X, y=y)

for k in Xy:
    print k
    print type(Xy[k]) is np.core.memmap
    print Xy[k]


np.savez("/tmp/data.dat", **Xy)





def load_datasets(datasets_filepath):
    Xy = np.load(datasets_filepath)
    return {k: Xy[k] for k in Xy.keys()}


Xy = load_datasets("/tmp/data.dat.npz")

from sklearn.svm import SVC
from epac import CV, Methods
cv_svm = CV(Methods(*[SVC(kernel="linear"),
                      SVC(kernel="rbf")]),
                      n_folds=3)


from epac import LocalEngine
local_engine = LocalEngine(cv_svm, num_processes=2)
cv_svm = local_engine.run(X=X, y=y)
print cv_svm.reduce()


from epac import SomaWorkflowEngine
swf_engine = SomaWorkflowEngine(cv_svm, num_processes=2)
cv_svm = swf_engine.run(X=X, y=y)
print cv_svm.reduce()
