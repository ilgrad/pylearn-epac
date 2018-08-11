# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:37:58 2013

@author: jinpeng
"""

import unittest
from sklearn import datasets
from epac.map_reduce.reducers import Reducer
from epac import Methods
from sklearn.metrics import precision_recall_fscore_support


## 1) Design your classifier
## ===========================================================================
class MySVC:
    def __init__(self, C=1.0):
        self.C = C

    def transform(self, X, y):
        from sklearn.svm import SVC
        svc = SVC(C=self.C)
        svc.fit(X, y)
        # "transform" should return a dictionary
        return {"y/pred": svc.predict(X), "y": y}


## 2) Design your reducer which compute, precision, recall, f1_score, etc.
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


class TestDumpClass(unittest.TestCase):
    def test_mysvc_reducer(self):
        ## 1) Build dataset
        ## ===================================================================
        X, y = datasets.make_classification(n_samples=12,
                                            n_features=10,
                                            n_informative=2,
                                            random_state=1)

        ## 2) run with Methods
        ## ===================================================================
        my_svc1 = MySVC(C=1.0)
        my_svc2 = MySVC(C=2.0)

        two_svc_single = Methods(my_svc1, my_svc2)
        two_svc_local = Methods(my_svc1, my_svc2)
        two_svc_swf = Methods(my_svc1, my_svc2)

        two_svc_single.reducer = MyReducer()
        two_svc_local.reducer = MyReducer()
        two_svc_swf.reducer = MyReducer()

        for leaf in two_svc_single.walk_leaves():
            print(leaf.get_key())
        for leaf in two_svc_local.walk_leaves():
            print(leaf.get_key())
        for leaf in two_svc_swf.walk_leaves():
            print(leaf.get_key())

        # top-down process to call transform
        two_svc_single.run(X=X, y=y)
        # buttom-up process to compute scores
        res_single = two_svc_single.reduce()

        ### You can get below results:
        ### ==================================================================
        ### [{'MySVC(C=1.0)': array([ 1.,  1.])}, {'MySVC(C=2.0)': array([ 1.,  1.])}]

        ### 3) Run using local multi-processes
        ### ==================================================================
        from epac.map_reduce.engine import LocalEngine
        local_engine = LocalEngine(two_svc_local, num_processes=2)
        two_svc_local = local_engine.run(**dict(X=X, y=y))
        res_local = two_svc_local.reduce()

        ### 4) Run using soma-workflow
        ### ==================================================================
        from epac.map_reduce.engine import SomaWorkflowEngine
        sfw_engine = SomaWorkflowEngine(tree_root=two_svc_swf,
                                        num_processes=2)
        two_svc_swf = sfw_engine.run(**dict(X=X, y=y))
        res_swf = two_svc_swf.reduce()
        if not repr(res_swf) == repr(res_local):
            raise ValueError("Cannot dump class definition")
        if not repr(res_swf) == repr(res_single):
            raise ValueError("Cannot dump class definition")

if __name__ == '__main__':
    unittest.main()
