# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:18:13 2013

@author: jinpeng.li@cea.fr
"""
import numpy as np
from sklearn import datasets
from epac import Methods
from epac.workflow.splitters import PrevStateMethods


# PrevStateMethods can implement algorithm like ISTA
# (iterative shrinkage-thresholding algorithm)
# see http://mechroom.technion.ac.il/~becka/papers/71654.pdf
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
        if not (self.v_beta == None):
            print "v_beta has been initialized as = ", self.v_beta
            min_err = self._get_error(self.v_beta, self.v_lambda, X, y)
        else:
            print "v_beta is None"

        # Search the beta which minimizes the error function
        # ==================================================
        for i in xrange(10):
            v_beta = np.random.random(len_beta)
            err = self._get_error(v_beta, self.v_lambda, X, y)
            if (self.v_beta == None) or err < min_err:
                self.v_beta = v_beta
                min_err = err

        print "Best v_beta =", self.v_beta
        pred_y = np.dot(X, self.v_beta)
        return {"y/pred": pred_y, "y/true": y, "best_beta": self.v_beta}


if __name__ == "__main__":
    ## 1) Build dataset
    ## ================================================
    X, y = datasets.make_classification(n_samples=10,
                                        n_features=5,
                                        n_informative=2,
                                        random_state=1)
    Xy = {"X": X, "y": y}

    ## 2) Build Methods
    ## ================================================
    print "Methods ==================================="
    methods = Methods(*[TOY_CLF(v_lambda=v_lambda)
                                for v_lambda in [2, 1]])
    print methods.run(**Xy)

    ## 3) Build PrevStateMethods like Methods
    ## ================================================
    ##               PrevStateMethods
    ##             /                  \
    ##  TOY_CLF(v_lambda=2)    TOY_CLF(v_lambda=1)
    ##
    ##  1. PrevStateMethods will look for different argumenets as signature
    ##     For example, here is v_lambda, there are different for each leaf
    ##  2. And then run TOY_CLF(v_lambda=2).transform
    ##  3. Except v_lambda, PrevStateMethods copy all the other parameters
    ##     from TOY_CLF(v_lambda=2) to TOY_CLF(v_lambda=1) as initialization
    ##  4. Finally call TOY_CLF(v_lambda=1).transform
    print "PrevStateMethods =========================="
    ps_methods = PrevStateMethods(*[TOY_CLF(v_lambda=v_lambda)
                                for v_lambda in [2, 1]])
    print ps_methods.run(**Xy)
