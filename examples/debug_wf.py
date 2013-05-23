# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:27:30 2013

@author: ed203246
"""

import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from epac import Pipe, CV, Methods, Permutations
from epac.sklearn_plugins import Permutation

from sklearn import datasets
from sklearn.lda import LDA
from epac import CV,

X, y = datasets.make_classification(n_samples=12, n_features=4)
wf = CV(SVC(), n_folds=2, reducer=SummaryStat(keep=True))
wf.fit_predict(X=X, y=y)
print wf.reduce()

wf = CV(SVC(), n_folds=2, reducer=SummaryStat(keep=False))
wf.fit_predict(X=X, y=y)
print wf.reduce()

from epac import SummaryStat
result = {'score_tr': [1, .8], 'score_te': [.9, .7]}
print SummaryStat(keep=True).reduce(result)
print SummaryStat(keep=False).reduce(result)

import pickle
file_path = "/tmp/toto.pkl"
output = open(file_path, 'wb')
pickle.dump(results, output)
output.close()

import pickle
file_path = "/tmp/toto.pkl"
inputf = open(file_path, 'rb')
results = pickle.load(inputf)
inputf.close()
from epac import SummaryStat
reducer=SummaryStat()

key2 = 'CVGridSearchRefit/CV/CV(*)/Grid/SVC(*)'
result = results[key2]
reducer.reduce(result)

results = {key2: self.reducer.reduce(results[key2]) for
                key2 in results}
                