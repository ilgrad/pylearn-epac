# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:13:42 2013

@author: jinpeng
"""
from epac.map_reduce.reducers import Reducer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC


## 2) Design your classifier
## ===========================================================================
class MySVC:
    def __init__(self, C=1.0):
        self.C = C
    def transform(self, X, y):
        svc = SVC(C=self.C)
        svc.fit(X, y)
        # "transform" should return a dictionary
        return {"y/pred": svc.predict(X), "y": y}


## 3) Design your reducer which compute, precision, recall, f1_score, etc.
## ===========================================================================
class MyReducer(Reducer):
    def reduce(self, result):
        pred_list = []
        # iterate all the results of each classifier
        # then you can design you own reducer!
        for res in result:
            precision, recall, f1_score, support = \
                    precision_recall_fscore_support(res['y'], res['y/pred'])
            pred_list.append({res['key']: recall})
        return pred_list
