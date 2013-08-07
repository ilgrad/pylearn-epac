# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:47:17 2013

@author: jinpeng
"""

import numpy as np
from tempfile import mkdtemp
import os.path as path
from sklearn import datasets


def convert2memmap(np_mat):
    filename = path.join(mkdtemp(), 'newfile.dat')
    mem_mat = np.memmap(filename,\
                     dtype='float32',\
                     mode='w+',\
                     shape=(np_mat.shape[0], np_mat.shape[1]))
    mem_mat[:] = np_mat[:]
    return mem_mat


data = np.arange(12, dtype='float32')
data.resize((3,4))


## 1) Build dataset
## ===========================================================================
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
cv_svm.run(X=X, y=y) # Top-down process: computing recognition rates, etc.
cv_svm.reduce() # Bottom-up process: computing p-values, etc.