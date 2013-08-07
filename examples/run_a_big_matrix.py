# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:47:17 2013

@author: jinpeng
"""

## 1) Build dataset
## ===========================================================================
from sklearn import datasets
X, y = datasets.make_classification(n_samples=12,
                                    n_features=10,
                                    n_informative=2,
                                    random_state=1)
from sklearn.svm import SVC
from epac import CV, Methods
cv_svm = CV(Methods(*[SVC(kernel="linear"),
                      SVC(kernel="rbf")]),
                 n_folds=3)
cv_svm.run(X=X, y=y) # Top-down process: computing recognition rates, etc.
cv_svm.reduce() # Bottom-up process: computing p-values, etc.