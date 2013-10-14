# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:43:39 2013

@author: jinpeng
"""

import unittest
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from epac import Pipe, CV, Methods
from epac import CVBestSearchRefitParallel
from epac import SomaWorkflowEngine
from sklearn import datasets
from epac.tests.utils import comp_2wf_reduce_res
from epac.tests.utils import compare_two_node


class TestCVBestSearchRefitParallelReduce(unittest.TestCase):
    def test_cv_best_search_refit_parallel(self):
        n_folds = 2
        n_folds_nested = 3
        k_values = [1, 2]
        C_values = [1, 2]
        n_samples = 500
        n_features = 10000
        n_cores = 2
        X, y = datasets.make_classification(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=5)
        # epac workflow for paralle computing
        pipelines = Methods(*[Pipe(SelectKBest(k=k),
                              Methods(*[SVC(kernel="linear", C=C)
                              for C in C_values]))
                              for k in k_values])
        pipeline = CVBestSearchRefitParallel(pipelines,
                                             n_folds=n_folds_nested)
        wf = CV(pipeline, n_folds=n_folds)

        sfw_engine = SomaWorkflowEngine(tree_root=wf,
                                        num_processes=n_cores,
                                        remove_finished_wf=False,
                                        remove_local_tree=False)
        sfw_engine_wf = sfw_engine.run(X=X, y=y)

        # epac workflow for normal node computing
        pipelines2 = Methods(*[Pipe(SelectKBest(k=k),
                              Methods(*[SVC(kernel="linear", C=C)
                              for C in C_values]))
                              for k in k_values])
        pipeline2 = CVBestSearchRefitParallel(pipelines2,
                                             n_folds=n_folds_nested)
        wf2 = CV(pipeline2, n_folds=n_folds)
        wf2.run(X=X, y=y)

        self.assertTrue(compare_two_node(sfw_engine_wf, wf2))
        self.assertTrue(comp_2wf_reduce_res(sfw_engine_wf, wf2))

if __name__ == '__main__':
    unittest.main()
