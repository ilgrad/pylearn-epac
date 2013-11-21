from joblib import Parallel, delayed
from epac import Methods
import numpy as np


from sklearn import datasets
from sklearn.svm import SVC


X, y = datasets.make_classification(n_samples=500,
                                    n_features=200000,
                                    n_informative=2,
                                    random_state=1)



methods = Methods(*[SVC(C=1, kernel='linear'), SVC(C=1, kernel='rbf')])   

data = {"X":X, 'y':y, "methods": methods}

# X = np.random.random((500, 200000))

def map_func(data):
  from sklearn.cross_validation import StratifiedKFold
  from sklearn import svm, cross_validation
  kfold = StratifiedKFold(y=data['y'], n_folds=3)
  # kfold = cross_validation.KFold(n=data.X.shape[0], n_folds=3)
  # svc = SVC(C=1, kernel='linear')
  for train, test in kfold:
      # svc.fit(data['X'][train], data['y'][train])
      # svc.predict(data['X'][test])
      data['methods'].run(X=data["X"][train], y=data['y'][train])
  return None


data_list = [data, data, data, data, data, data]

Parallel(n_jobs=4, verbose=100)(delayed(map_func)(d)
                    for d in data_list)
