# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:47:17 2013

@author: jinpeng.li@cea.fr
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
                     shape=np_mat.shape)
    mem_mat[:] = np_mat[:]
    return mem_mat


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

#cv_svm.run(X=X, y=y) # Top-down process: computing recognition rates, etc.
#for leaf in cv_svm.walk_leaves():
#    print leaf.load_results()
#cv_svm.reduce() # Bottom-up process: computing p-values, etc.

from epac import LocalEngine
local_engine = LocalEngine(cv_svm, num_processes=2)
cv_svm = local_engine.run(X=X, y=y)
cv_svm.reduce()


mmatrx = MemmapMatrix(X)
X_cp = mmatrx.get()


def test(X):
    print X.filename
    return 0

from multiprocessing import Pool

pool = Pool(processes=2)
res = pool.map(test, [X, X])
pool.map()


import dill as pickle
dumpstr = pickle.dumps(X)
f = open("/tmp/dump.data", "w+")
f.write(dumpstr)
f.close()

f = open("/tmp/dump.data", "rb")
dumpstr = f.read()
f.close()
X_dill = pickle.loads(dumpstr)
X_dill.filename
