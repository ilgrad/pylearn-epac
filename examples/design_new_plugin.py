# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:05:28 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay.li@cea.fr
"""

import numpy as np
from sklearn import datasets
from epac import CV
from epac.map_reduce.reducers import Reducer
from epac.map_reduce.results import Result
from epac import Methods
from sklearn.metrics import accuracy_score

X, y = datasets.make_classification(n_samples=10,
                                    n_features=50,
                                    n_informative=2,
                                    random_state=None)
#import numpy as np
#y = y[np.random.permutation(len(y))]

# A User node should implement fit + predict or fit + transfrom
# ===============================================================

# Predict can return an array. In this case EPAC will
# put the prediction in a Result (a dictionnary). with key = "y/pred". y being the
# difference between input agrument of fit and predict. The true y will also figure
# in the result with key "y/true"
class MySVM:
    def __init__(self, C=1.0):
        self.C = C
    def fit(self, X, y):
        from sklearn.svm import SVC
        self.svc = SVC(C=self.C)
        self.svc.fit(X, y)
    def predict(self, X):
        return self.svc.predict(X)

svms = Methods(MySVM(C=1.0), MySVM(C=2.0))
cv = CV(svms, cv_key="y", cv_type="stratified", n_folds=2,
        reducer=None)
cv.run(X=X, y=y)  # top-down process to call transform
cv.reduce()       # buttom-up process

from sklearn.decomposition import PCA
class MyPCA(PCA):
    """PCA with predict method"""
    def predict(self, X):
        """Project to X PCs then project back to original space
        If X is not singular, self.fit(X).predict(X) == X"""
        return np.dot(self.transform(X), self.components_) + self.mean_

pcas = Methods(MyPCA(n_components=1), MyPCA(n_components=2))
cv = CV(pcas, n_folds=2, reducer=None)
cv.run(X=X, y=y)  # top-down process to call transform
cv.reduce()       # buttom-up process


# User defined reducer receive a ResultSet (list of dict) and returns a Result
# ============================================================================

class KeepBest(Reducer):
    """This reducer keep only the best classifier and return a single result"""
    def reduce(self, result_set):
        from epac.utils import train_test_split
        # Iterate over the result_set: a list of results (see transform).
        # Each result contains an additional unique key called "key". Example:
        # "MySVC(C=1.0)" or "MySVC(C=2.0)"
        # then you can design you own reducer!
        max_accuracy = -1
        for result in result_set:
            # Each result is the dictionary returned by "transform".
            # If there is a CV in the workflow, EPAC suffixes keys 
            # with /test or /train.
            # function train_test_split split result (dict) into two sub-dicts
            # removing /test or /train suffix. It returns two reference of the same
            # dict if no /test or /train suffix where found.
            output = dict()  # output result is a dictonary
            result_train, result_test = train_test_split(result)
            if result_train is result_test:  # No CV in the EPAC workflow
                accuracy = accuracy_score(result['y/true'], result['y/pred'])
                output["acc/y"] = accuracy
            else:  # there was a CV in the EPAC workflow
                accuracy = accuracy_score(result_test['y/true'], result_test['y/pred'])
                output["acc/y/test"] = accuracy
                output["acc/y/train"] = accuracy_score(result_train['y/true'], result_train['y/pred'])
            if accuracy > max_accuracy:
                # keep the key in the reduced result
                best_result = Result(key=result['key'], **output)
        return best_result  # reducer return a single result


best_svc = Methods(SVMTransform(C=1.0), SVMTransform(C=2.0))
best_svc.reducer = KeepBest()
cv = CV(best_svc, cv_key="y", cv_type="stratified", n_folds=2,
        reducer=None)
cv.run(X=X, y=y)  # top-down process to call transform
cv.reduce()       # buttom-up process

# User defined reducer receive a ResultSet (list of dict) and returns a ResultSet
# ===============================================================================

class AccuracySummary(Reducer):
    """This reducer summarize the results by accuracies and return a set of results"""
    def reduce(self, result_set):
        # if you want to a remote execution of your code, import should be done
        # within methods
        from epac.utils import train_test_split
        from epac.map_reduce.results import ResultSet
        outputs = list()  # output result is a dictonary
        for result in result_set:
            output = dict()  # output result is a dictonary
            result_train, result_test = train_test_split(result)
            if result_train is result_test:
                accuracy = accuracy_score(result['y/true'], result['y/pred'])
                output["acc/y"] = accuracy
            else:
                accuracy = accuracy_score(result_test['y/true'], result_test['y/pred'])
                output["acc/y/test"] = accuracy
                output["acc/y/train"] = accuracy_score(result_train['y/true'], result_train['y/pred'])
            outputs.append(Result(key=result['key'], **output))
        return ResultSet(*outputs)

best_svc.reducer = AccuracySummary()  # Modify the reducer
cv = CV(best_svc, cv_key="y", cv_type="stratified", n_folds=2,
        reducer=None)
cv.run(X=X, y=y)  # top-down process to call transform
cv.reduce()       # buttom-up process


# A User node implements "transform" that return a dictionary
# ===========================================================

class SVMTransform:
    def __init__(self, C=1.0):
        self.C = C
    def transform(self, X, y):
        from sklearn.svm import SVC
        svc = SVC(C=self.C)
        svc.fit(X, y)
        # "transform" should return a dictionary: ie.: a result, keys are abritrary
        return {"y/pred": svc.predict(X), "y/true": y}

best_svc_tranform = Methods(SVMTransform(C=1.0), SVMTransform(C=2.0))
cv = CV(best_svc_tranform, cv_key="y", cv_type="stratified", n_folds=2,
        reducer=None)
cv.run(X=X, y=y)  # top-down process to call transform
cv.reduce()       # buttom-up process


#
## 4) Run using local multi-processes
## ==================================

from epac.map_reduce.engine import LocalEngine
local_engine = LocalEngine(best_svc, num_processes=2)
best_svc = local_engine.run(**dict(X=X, y=y))
best_svc_tranform.reduce()

## 5) Run using soma-workflow
## ==========================

from epac.map_reduce.engine import SomaWorkflowEngine
sfw_engine = SomaWorkflowEngine(tree_root=best_svc,
                                num_processes=2)
best_svc = sfw_engine.run(**dict(X=X, y=y))
best_svc.reduce()
