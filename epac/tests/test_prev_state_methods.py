# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:44:30 2013

@author: jinpeng.li@cea.fr
"""

import unittest
import numpy as np
from sklearn import datasets
from epac import Methods
from epac.workflow.splitters import WarmStartMethods
from epac.tests.utils import comp_2wf_reduce_res
from epac.tests.utils import compare_two_node


class TOY_CLF:
    def __init__(self, v_lambda):
        self.v_lambda = v_lambda
        self.v_beta = None

    def _get_error(self, v_beta, v_lambda, X, y):
        pred_y = np.dot(X, v_beta)
        ss_error = np.sum((y - pred_y) ** 2, axis=0)
        return ss_error

    def transform(self, X, y):
        len_beta = X.shape[1]

        min_err = 0
        if not (self.v_beta is None):
            min_err = self._get_error(self.v_beta, self.v_lambda, X, y)

        # Search the beta which minimizes the error function
        # ==================================================
        for i in xrange(10):
            v_beta = np.random.random(len_beta)
            err = self._get_error(v_beta, self.v_lambda, X, y)
            if (self.v_beta is None) or err < min_err:
                self.v_beta = v_beta
                min_err = err

        pred_y = np.dot(X, self.v_beta)
        return {"y/pred": pred_y, "y/true": y, "best_beta": self.v_beta}


class TestWorkFlow(unittest.TestCase):
    def test_prev_state_methods(self):
        ## 1) Build dataset
        ## ================================================
        X, y = datasets.make_classification(n_samples=5,
                                            n_features=20,
                                            n_informative=2)
        Xy = {"X": X, "y": y}
        methods = Methods(*[TOY_CLF(v_lambda=v_lambda)
                            for v_lambda in [2, 1]])
        methods.run(**Xy)

        ps_methods = WarmStartMethods(*[TOY_CLF(v_lambda=v_lambda)
                                        for v_lambda in [2, 1]])
        ps_methods.run(**Xy)
        self.assertTrue(compare_two_node(methods, ps_methods))
        self.assertTrue(comp_2wf_reduce_res(methods, ps_methods))


if __name__ == '__main__':
    unittest.main()
