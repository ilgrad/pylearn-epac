# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:05:28 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay.li@cea.fr
"""

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from epac import CV
from epac.map_reduce.reducers import Reducer
from epac.map_reduce.results import Result
from epac import Methods, Pipe
from sklearn.metrics import accuracy_score

X, y = datasets.make_classification(n_samples=10,
                                    n_features=50,
                                    n_informative=2,
                                    random_state=None)

# User defined node
# =================

# Methods: fit + predict or fit + transfrom
# Input: arrays
# Output: fit None, predict array or ad dictionary
# EPAC puts the prediction in a Result (a dictionnary)

# Create a new classifier

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

# Extends PCA with predict

class MyPCA(PCA):
    """PCA with predict method"""
    def predict(self, X):
        """Project to X PCs then project back to original space
        If X is not singular, self.fit(X).predict(X) == X"""
        return np.dot(self.transform(X), self.components_) + self.mean_

pcas = Methods(MyPCA(n_components=1), MyPCA(n_components=2))
cv = CV(pcas, n_folds=2, cv_key="X", cv_type="random", reducer=None)
cv.run(X=X)  # top-down process to call transform
cv.reduce()       # buttom-up process

# Result and ResultSet 
# ====================

# Reducer take input results (ResultSet) and produce result (Result) or
# ResultSet. So it is necessary to understand how EPAC handle results using
# Result and ResultSet.
#
# Result
# ------
#
# EPAC puts the prediction in a Result (a dictionnary), the keys are formed
# as <data_name>[/<train|test>]/<true|pred>
#
# <data_name>: is the difference between the input agruments of fit and predict.
# Example, fit(X, y), predict(X), key will be "y".
# If no differences, fit(X), predict(X), key will be "X". 
#
# <true|pred>. Prediction on test/train set are suffixed by test/train.
#
# <train|test>. If the workflow contains a cross-validation, Results keys
# will be suffixed by train or test: example "y/train" and "y/test". 
#
# Finally, EPAC add a key "key" corresponding to the signature of the node.
# Example: "SVC(C=1)" or "SVC(C=2)"
#
# ResultSet
# ---------
#
# Through the reducing phase (bottum-up process) EPAC agregate Result into
# ResultSet which is a collection of Result.
# Examples
# --------

from sklearn.svm import SVC
from epac import CV, Methods
X, y = datasets.make_classification(n_samples=4, random_state=0)
svms.run(X=X, y=y)
result_set = svms.reduce()
print result_set
#ResultSet(
#[{'key': MySVM(C=1.0), 'y/true': [0 1 0 1], 'y/pred': [0 1 0 1]},
# {'key': MySVM(C=2.0), 'y/true': [0 1 0 1], 'y/pred': [0 1 0 1]}]) 

# With a CV
cv_svm = CV(svms, reducer=None, n_folds=2, random_state=0)
cv_svm.run(X=X, y=y)
result_set = cv_svm.reduce()  # Return a ResultSet ie.: a list of Results
print result_set
#ResultSet(
#[{'key': CV(nb=0)/MySVM(C=1.0), 'y/true/test': [0 1], 'y/pred/test': [0 0], 'y/pred/train': [0 1]},
# {'key': CV(nb=0)/MySVM(C=2.0), 'y/true/test': [0 1], 'y/pred/test': [0 0], 'y/pred/train': [0 1]},
# {'key': CV(nb=1)/MySVM(C=1.0), 'y/true/test': [0 1], 'y/pred/test': [0 0], 'y/pred/train': [0 1]},
# {'key': CV(nb=1)/MySVM(C=2.0), 'y/true/test': [0 1], 'y/pred/test': [0 0], 'y/pred/train': [0 1]}])

result = result_set['CV(nb=0)/MySVM(C=1.0)']
print result
from epac.utils import train_test_split
result_train, result_test = train_test_split(result)

 
# User defined reducer
# ====================

# Method: reduce
# Input: ResultSet (list of dict)
# Output: single Result (dict) or ResultSet (list of dict)

# Details
# Each Result is a dict with keys 
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
