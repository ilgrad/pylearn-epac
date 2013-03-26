# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:55:46 2013

@author: edouard
"""
# run workflow.py

import numpy as np
X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2], [-3, -4], [-5, -6], [-7, -8]])
y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])

from sklearn import datasets
iris = datasets.load_iris()

# Add the noisy data to the informative features
X = np.hstack((iris.data, np.random.normal(size=(len(iris.data), 20))))
y = iris.target


from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest

## Build sequential Pipeline
## =========================

#from workflow import Seq
# Simple sequential pipeline
# 2  SelectKBest
# |
# SVM Classifier
from epac import Seq
pipe = Seq(SelectKBest(k=2), SVC(kernel="linear"))
pipe.fit(X=X, y=y).predict(X=X)
# The downstream data-flow is a keyword arguments (dict) containing X and y.
# It will pass through each processing node, SelectKBest(k=2) and SVC.
# The Fit:
# Each non-leaf (here SelectKBest  node call the fit method, then apply
# the transformation on the downstream and pass it to the next node. The leaf
# node (here SVC) do not call the transformation.
# The predict:
# Similar sequential tranformation are applied on X, except that the leaf node
# call the predict method.

## Parallelization
## ===============

#from epac import Par
# Multi-classifiers
# -----------------

#    Par    ParMethods (Splitter)
#  /   \
# LDA  SVM  Classifiers (Estimator)
from epac import ParMethods
multi = ParMethods(LDA(),  SVC(kernel="linear"))
multi.fit(X=X, y=y)
multi.predict(X=X)
# Do both
multi.fit_predict(X=X, y=y)

#           Par          ParMethods (Splitter)
#          /  \
# SVM(linear)  SVM(rbf)  Classifiers (Estimator)
svms = ParMethods(*[SVC(kernel=kernel) for kernel in ("linear", "rbf")])
svms.fit_predict(X=X, y=y)
svms.reduce()

# Parallelize sequential Pipeline: Anova(k best selection) + SVM.
# No collisions between upstream keys, then no aggretation.
#     Par      MultiMethod (Splitter)
#  /   |   \
# 1    5   10  SelectKBest (Estimator)
# |    |    |
# SVM SVM SVM  Classifiers (Estimator)
anovas_svm = ParMethods(*[Seq(SelectKBest(k=k), SVC(kernel="linear")) for k in 
    [1, 5, 10]])
anovas_svm.fit_predict(X=X, y=y)
anovas_svm.reduce()
[l.get_key() for l in anovas_svm]
[l.get_key(2) for l in anovas_svm]  # No key 2 collisions, no aggregation


# Parallelize SVM with several parameters.
# Collisions between upstream keys, trig aggretation.

#                    Par                ParGrid (Splitter)
#                  /     \
# SVM(linear, C=1)  .... SVM(rbf, C=10) Classifiers (Estimator)
# ParGrid and PArMethods differ onlys the way they process the upstream
# flow. With ParGrid Children differs only by theire arguments, and thus
# are aggregated toggether
from epac import ParGrid
svms = ParGrid(*[SVC(kernel=kernel, C=C) for \
    kernel in ("linear", "rbf") for C in [1, 10]])
svms.fit_predict(X=X, y=y)
svms.reduce()
[l.get_key() for l in svms]
[l.get_key(2) for l in svms]  # key 2 collisions trig aggregation

# Cross-validation
# ---------------
# CV of LDA
#     ParCV               (Splitter)
#  /   |   \
# 0    1    2  Folds      (Slicer)
# |    |    |
# LDA LDA LDA  Classifier (Estimator)
from epac import ParCV
from epac import SummaryStat
cv_lda = ParCV(LDA(), n_folds=3, y=y, reducer=SummaryStat())
cv_lda.fit_predict(X=X, y=y)
cv_lda.reduce()

# A CV node is a Splitter: it as one child per fold. Each child is a slicer
# ie.: it re-slices the downstream data-flow according into train or test
# sample. When it is called with "fit" it uses the train samples. When it is 
# called with "predict" it uses the test samples.
# If it is called with transform, user has to precise wich sample to use. To
# do that just add a argument sample_set="train" or "test" in the downstream
# data-flow. This argument will be catched by the slicer.
cv_lda.transform(X=X, sample_set="train")
cv_lda.transform(X=X, sample_set="test")

#run epac.py
#self = cv_lda
#task=LDA(); n_folds=3; reducer=None; kwargs = dict(n=X.shape[0])
#self.children = []
#ds_kwargs = kwargs

# ParParPermutations + Cross-validation
# -------------------------------
#           ParPerm                  CV (Splitter)
#         /     |       \
#        0      1       2            Samples (Slicer)
#       |
#     ParCV                          CV (Splitter)
#  /   |   \
# 0    1    2                        Folds (Slicer)
# |    |    |
# LDA LDA LDA                        Classifier (Estimator)

from epac import ParPerm, ParCV, WF
from epac import SummaryStat, PvalPermutations
#from stores import
# _obj_to_dict, _dict_to_obj

perms_cv_lda = ParPerm(ParCV(LDA(), n_folds=3, reducer=SummaryStat()),
                    n_perms=3, permute="y", y=y, reducer=PvalPermutations())
# Save tree
import tempfile
perms_cv_lda.save(store=tempfile.mktemp())
# Fit & Predict
perms_cv_lda.fit_predict(X=X, y=y)
# Save results
perms_cv_lda.save(attr="results")
key = perms_cv_lda.get_key()
# Reload tree, all you need to know is the key
tree = WF.load(key)
# Reduces results
tree.reduce()

## DEBUGGING
## =========
from epac import ParMethods
multi = ParMethods(LDA(),  SVC(kernel="linear"))
multi.fit(X=X, y=y)
multi.predict(X=X)
# Do both
multi.fit_predict(X=X, y=y)
from epac.workflow import conf, debug
conf.DEBUG = True  # set debug to True
multi.fit_predict(X=X, y=y)  # re-run
ds_kwargs = dict(X=X, y=y)  # build the down-stream data flow
# get all nodes from root to the current node (stored in debug.current)
node_iterator = debug.current.get_path_from_root().__iter__()
# Manually iterate from root to current node, until desire node
self = node_iterator.next()
print self
ds_kwargs = self.fit_predict(recursion=False, **ds_kwargs)
print ds_kwargs