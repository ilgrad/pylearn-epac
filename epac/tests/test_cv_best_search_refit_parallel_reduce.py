
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from epac import Pipe, CV, Methods
from epac import CVBestSearchRefitParallel
from epac import SomaWorkflowEngine
from sklearn import datasets

n_folds = 2
n_folds_nested = 3
k_values = [1, 2]
C_values = [1, 2]
n_samples = 50
n_features = 500
n_cores = 3


X, y = datasets.make_classification(n_samples=n_samples,
                                    n_features=n_features,
                                    n_informative=5)
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
print sfw_engine_wf.reduce()
