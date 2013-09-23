.. _tutorials:


Building Dataset
================

In this section, we start with small examples to understand how to use EPAC libary. The source code of this tutorial is avaible in **./pylearn-epac/examples/small_toy.py**

In order to build the EPAC tree, we first need a dataset. Let's build *X* matrix (samples) and *y* vector (labels) as below.

.. code-block:: python


    from sklearn.datasets import samples_generator
    X, y = samples_generator.make_classification(
                             n_samples=10,
                             n_features=5,
                             n_informative=5,
                             n_redundant=0,
                             random_state=42)


In the next section, we can use *X* and *y* to run machine learning algorithm.


A simple SVM example
====================

Conventionally, we want to learn a mapping from *X* to *y* as a machine learning task as shown in the figure below.

.. figure:: ./images/simple_ml_example.png
   :scale: 100 %
   :align: center
   :alt: machine learning task


Here is an example of SVM from scikit-learn:

.. code-block:: python

    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    clf.fit(X=X, y=y)
    prediction = clf.predict(X)


The first step is to fit the SVM model and then we can predict *y* using *X*.


.. figure:: ./images/svc_input_output.png
   :scale: 100 %
   :align: center
   :alt: input and output of SVC

The input of SVM is *X* and *y* which is a dictionary. The output is the prediction. We first call the fit function and then call the predict function of SVM. In fact, here will a basic node in EPAC tree.


Preprocessing
=============

Before the classification step, sometimes we need a preprocessing step, for example, feature selection (*SelectKBest* in sklearn) as shown below. 


.. figure:: ./images/select_k_best.png
   :scale: 100 %
   :align: center
   :alt: select k best features based on univariate statistical tests 

It can reduce the *X* feature dimension so that the classifier (SVM in this example) takes less time.


.. figure:: ./images/select_k_best_example.png
   :scale: 100 %
   :align: center
   :alt: reduce X feature dimension

To implement these two processes without Pipeline in sklearn and without Pipe in EPAC.


.. code-block:: python


    # without epac
    # 0. build two processes
    from sklearn.feature_selection import SelectKBest
    select_k_best = SelectKBest(k=2)
    svm = SVC(kernel='linear')
    # 1. fit model from X and y
    select_k_best.fit(X=X, y=y)
    # select features from X
    tr_X = select_k_best.transform(X)
    # 2. fit model from output of select_k_best 
    svm.fit(X=tr_X, y=y)
    # predict results from output of select_k_best
    svm.predict(X=tr_X)


In the next section, we will introduce how to simply the codes using EPAC.


Basic units: Pipe and Methods
=============================

In EPAC, there are two basic elements, *Pipe* (sequential pipeline) and *Methods* (parallel methods). *Pipe* is used for running a sequence of nodes while *Methods* is designed for parallelly run nodes or sequences of nodes.

Pipe
----
 
Now, we first introduce *Pipe*, which executes a sequential of BaseNode. These nodes should implement 

either as non-leaf node (non-termianl node),

- fit and transform, e.g. *SelectKBest(k=2)*,

or as leaf node (terminal node)

- fit and predict, e.g. *SVM()*,

or as any nodes with only transform function.

.. figure:: ./images/epac_nodes.png
   :scale: 100 %
   :align: center
   :alt: reduce X feature dimension

As the example shown above, you can find that the output of ``SelectKBest(k=2)`` becomes the input of ``SVM()``. It is a sequential process. Using EPAC, the codes are much more simple as shown below. After building EPAC tree, we can call *run* which is a top-down process. The input *X* and *y* will pass from *SelectKBest* to SVM. The output of *SelectKBest* will become the input of *SVM* automatically. All the input and output are a dictionary. For example, we want to run  

::

    >>> # Build sequential Pipeline
    >>> # -------------------------
    >>> # SelectKBest (Transformer)
    >>> # |
    >>> # SVM Classifier (Estimator)
    >>> from epac import Pipe
    >>> pipe = Pipe(SelectKBest(k=2), SVC())
    >>> pipe.run(X=X, y=y)
    {'y/true': array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0]), 'y/pred': array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0])}


The downstream data-flow is a keyword arguments (dict) containing *X* and *y*. It will pass through each processing node, *SelectKBest(k=2)* and *SVM*. Each non-terminal node call fit and transform, that take a dictionnary as input and produces a dictionnary as output. The output is passed to the next node till terminal node. The return value of the run is simply agregation of the outputs (dict) of the leaf nodes (terminal nodes). Here ``y/true`` means the labels in original dataset while ``y/pred`` means the prediction from pipe. ``y/true`` and ``y/pred`` are use for the reducing step. In the next section, we will present parallel ``Methods``. 

Methods
-------

In this section, ``Methods`` will be described to run several classifiers in parallel as below codes. It will make copy of the input ``{X=X, y=y}``, and then pass it into the children nodes respectively.

::

    >>> ## Parallelization
    >>> ## ===============
    >>> # Multi-classifiers
    >>> # -----------------
    >>> #         Methods       Methods (Splitter)
    >>> #        /   \
    >>> # SVM(C=1)  SVM(C=10)   Classifiers (Estimator)
    >>> from epac import Methods
    >>> multi = Methods(SVC(C=1), SVC(C=10))
    >>> multi.run(X=X, y=y)
    [{'y/true': array([ 0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.]), 'y/pred': array([ 0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.])}, {'y/true': array([ 0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.]), 'y/pred': array([ 0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.])}]
    >>> print multi.reduce()
    ResultSet(
    [{'key': SVC(C=1), 'y/true': [1 0 0 1 1 0 1 0 1 0], 'y/pred': [1 0 0 1 1 0 1 0 1 0]},
     {'key': SVC(C=10), 'y/true': [1 0 0 1 1 0 1 0 1 0], 'y/pred': [1 0 0 1 1 0 1 0 1 0]}])    

In these codes, ``Methods`` set the input of dictionary ``{X=X, y=y}`` to ``SVM(C=1)`` and to ``SVM(C=10)`` respectively. ``multi.reduce()`` outputs into "ResultSet" which is a dict-like structure which contains the "keys" of the methods that as been used. In EPAC, **run** means the top-down process, and **reduce** means bottom-up process. For this moment, the **reduce** process returen only the collection of results from classifiers. We will show more meaningful examples using **reduce** later.  A more complicated ``Methods`` example using two arguments is shown as below.

 
::    
    
    >>> #                         Methods                  Methods (Splitter)
    >>> #          /                        \
    >>> # SVM(l1, C=1)  SVM(l1, C=10)  ..... SVM(l2, C=10) Classifiers (Estimator)
    >>> from sklearn.svm import LinearSVC as SVM
    >>> svms = Methods(*[SVM(loss=loss, C=C) for loss in ("l1", "l2") for C in [1, 10]])
    >>> svms.run(X=X, y=y)
    [{'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.])}, {'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.])}, {'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.])}, {'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.])}]
    >>> print svms.reduce()
    ResultSet(
    [{'key': LinearSVC(loss=l1,C=1), 'y/true': [ 1.  0.  0.  1.  0.  0.  1.  0.  1.  1.  0.  1.], 'y/pred': [ 0.  0.  0.  1.  0.  0.  1.  0.  1.  0.  0.  1.]},
     {'key': LinearSVC(loss=l1,C=10), 'y/true': [ 1.  0.  0.  1.  0.  0.  1.  0.  1.  1.  0.  1.], 'y/pred': [ 1.  0.  0.  1.  0.  0.  1.  0.  1.  1.  0.  1.]},
     {'key': LinearSVC(loss=l2,C=1), 'y/true': [ 1.  0.  0.  1.  0.  0.  1.  0.  1.  1.  0.  1.], 'y/pred': [ 0.  0.  0.  1.  0.  0.  1.  0.  1.  0.  0.  1.]},
     {'key': LinearSVC(loss=l2,C=10), 'y/true': [ 1.  0.  0.  1.  0.  0.  1.  0.  1.  1.  0.  1.], 'y/pred': [ 1.  0.  0.  1.  0.  0.  1.  0.  1.  1.  0.  1.]}])



This example illustrates how to iterate two argument arrays using EPAC. We can computes all the results from all the combinations. In the next section, we will show how to combine ``Pipe`` and ``Methods``.

Pipe and Methods Combination
----------------------------


An example is shown in this section to combine ``Methods`` and ``Pipe``. When you need some common preprocessing, e.g. feature selection, for all classifiers, you can concatenate a preprocessing with the combination of pipe and methods, in order to save your computing resource. An example that concatenating ``SelectKBest`` and two svm (``SVC(C=1)`` and ``SVC(C=2)``) is shown below: 


.. figure:: ./images/pipe_and_method_combination.png
   :scale: 100 %
   :align: center
   :alt: save your computing resource using pipe


::

   >>> pipe_methods = Pipe(SelectKBest(k=2), Methods(SVM(C=1), SVM(C=10)))
   >>> pipe_methods.run(X=X, y=y)
   [{'y/true': array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0]), 'y/pred': array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0])}, {'y/true': array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0]), 'y/pred': array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0])}] 
   >>> pipe_methods.reduce()
   ResultSet(
   [{'key': SelectKBest/LinearSVC(C=1), 'y/true': [1 0 0 1 1 0 1 0 1 0], 'y/pred': [1 0 0 1 0 0 1 0 1 0]},
    {'key': SelectKBest/LinearSVC(C=10), 'y/true': [1 0 0 1 1 0 1 0 1 0], 'y/pred': [1 0 0 1 0 0 1 0 1 0]}]) 


Therefore, two basic units have been presented in this section. You can start to construct your own EPAC tree for many machine learning processes. 
In the next section, we will introduce some predefined nodes in EPAC, for instance, Cross-Validation.

Cross-validation
================

In this section, we will introduce the cross-validation as below.

::
    
    >>> # Cross-validation
    >>> # ----------------
    >>> # CV of LDA
    >>> #      CV                 (Splitter)
    >>> #  /   |   \
    >>> # 0    1    2  Folds      (Slicer)
    >>> # |    |
    >>> #   Methods               (Splitter)
    >>> #    /   \
    >>> #  LDA  SVM    Classifier (Estimator)
    >>> from epac import CV, Methods
    >>> from sklearn.lda import LDA
    >>> cv = CV(Methods(LDA(), SVM()))
    >>> cv.run(X=X, y=y)
    [[{'y/test/pred': array([1, 0]), 'y/train/pred': array([0, 1, 1, 0, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}, {'y/test/pred': array([1, 0]), 'y/train/pred': array([0, 1, 1, 0, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}], [{'y/test/pred': array([0, 1]), 'y/train/pred': array([1, 0, 1, 0, 1, 0, 1, 0]), 'y/test/true': array([0, 1])}, {'y/test/pred': array([0, 1]), 'y/train/pred': array([1, 0, 1, 0, 1, 0, 1, 0]), 'y/test/true': array([0, 1])}], [{'y/test/pred': array([0, 0]), 'y/train/pred': array([1, 0, 0, 1, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}, {'y/test/pred': array([0, 0]), 'y/train/pred': array([1, 0, 0, 1, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}], [{'y/test/pred': array([1, 0]), 'y/train/pred': array([1, 0, 0, 1, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}, {'y/test/pred': array([1, 0]), 'y/train/pred': array([1, 0, 0, 1, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}], [{'y/test/pred': array([1, 0]), 'y/train/pred': array([1, 0, 0, 1, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}, {'y/test/pred': array([1, 0]), 'y/train/pred': array([1, 0, 0, 1, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}]] 
    >>> print cv.reduce()
    ResultSet(
    [{'key': LDA, 'y/test/score_precision': [ 0.83333333  1.        ], 'y/test/score_recall': [ 1.   0.8], 'y/test/score_accuracy': 0.9, 'y/test/score_f1': [ 0.90909091  0.88888889], 'y/test/score_recall_mean': 0.9},
     {'key': LinearSVC, 'y/test/score_precision': [ 0.83333333  1.        ], 'y/test/score_recall': [ 1.   0.8], 'y/test/score_accuracy': 0.9, 'y/test/score_f1': [ 0.90909091  0.88888889], 'y/test/score_recall_mean': 0.9}]) 

``cv.run(X=X, y=y)`` run the top-down process so that we get all the results, and ``cv.reduce()`` compute different scores, accuracies, etc. For instance, ``y/test/score_precision`` denotes the precision on the test part for the prediction on *y*. 

Model Selection using Cross-validation
======================================

We have several classifiers and we need to select the best classifier using the cross-validation. 

::

    >>> # Model selection using CV
    >>> # ------------------------
    >>> # CVBestSearchRefit
    >>> #      Methods       (Splitter)
    >>> #      /    \
    >>> # SVM(C=1)  SVM(C=10)   Classifier (Estimator)
    >>> from epac import Pipe, CVBestSearchRefit, Methods
    >>> # CV + Grid search of a simple classifier
    >>> wf = CVBestSearchRefit(Methods(SVM(C=1), SVM(C=10)))
    >>> wf.run(X=X, y=y)
    {'best_params': [{'C': 1, 'name': 'LinearSVC'}], 'y/true': array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0]), 'y/pred': array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0])}
    >>> print wf.reduce()
    ResultSet(
    [{'key': CVBestSearchRefit, 'best_params': [{'C': 1, 'name': 'LinearSVC'}], 'y/true': [1 0 0 1 1 0 1 0 1 0], 'y/pred': [1 0 0 1 1 0 1 0 1 0]}]) 

This example shows how to select model from several classifiers. ``wf.run(X=X, y=y)`` and ``wf.reduce()`` return the same results which are the best parameters and its prediction on ``y`` vector. A more complicated example, which select model from ``SelectKBest -> LDA()`` and ``SelectKBest -> SVM()``,  is shown as below.   

::

    >>> # Feature selection combined with SVM and LDA
    >>> # CVBestSearchRefit
    >>> #                     Methods          (Splitter)
    >>> #               /              \
    >>> #            KBest(1)         KBest(5) SelectKBest (Estimator)
    >>> #              |
    >>> #            Methods                   (Splitter)
    >>> #        /          \
    >>> #    LDA()          SVM() ...          Classifiers (Estimator)
    >>> pipelines = Methods(*[Pipe(SelectKBest(k=k), Methods(LDA(), SVM())) for k in [1, 5]])
    >>> print [n for n in pipelines.walk_leaves()]
    [Methods/SelectKBest(k=1)/Methods/LDA, Methods/SelectKBest(k=1)/Methods/LinearSVC, Methods/SelectKBest(k=5)/Methods/LDA, Methods/SelectKBest(k=5)/Methods/LinearSVC] 
    >>> best_cv = CVBestSearchRefit(pipelines)
    >>> best_cv.run(X=X, y=y)
    {'best_params': [{'k': 1, 'name': 'SelectKBest'}, {'name': 'LDA'}], 'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 1.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.])} 
    >>> best_cv.reduce()
    ResultSet(
    [{'key': CVBestSearchRefit, 'best_params': [{'k': 5, 'name': 'SelectKBest'}, {'name': 'LDA'}], 'y/true': [1 0 0 1 1 0 1 0 1 0], 'y/pred': [1 0 0 1 1 0 1 0 1 0]}]) 
    

We can use EPAC like playing "lego". ``best_cv`` can be put in cross-validation as shown below.   

::
 
    >>> # Put it in an outer CV
    >>> cv = CV(best_cv)
    >>> cv.run(X=X, y=y)
    [{'best_params': [{'k': 5, 'name': 'SelectKBest'}, {'name': 'LinearSVC'}], 'y/test/pred': array([1, 0]), 'y/train/pred': array([0, 1, 1, 0, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}, {'best_params': [{'k': 5, 'name': 'SelectKBest'}, {'name': 'LinearSVC'}], 'y/test/pred': array([0, 1]), 'y/train/pred': array([1, 0, 1, 0, 1, 0, 1, 0]), 'y/test/true': array([0, 1])}, {'best_params': [{'k': 5, 'name': 'SelectKBest'}, {'name': 'LDA'}], 'y/test/pred': array([0, 0]), 'y/train/pred': array([1, 0, 0, 1, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}, {'best_params': [{'k': 5, 'name': 'SelectKBest'}, {'name': 'LDA'}], 'y/test/pred': array([1, 0]), 'y/train/pred': array([1, 0, 0, 1, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}, {'best_params': [{'k': 5, 'name': 'SelectKBest'}, {'name': 'LDA'}], 'y/test/pred': array([1, 0]), 'y/train/pred': array([1, 0, 0, 1, 1, 0, 1, 0]), 'y/test/true': array([1, 0])}] 
    >>> cv.reduce()
    ResultSet(
    [{'key': CVBestSearchRefit, 'y/test/score_precision': [ 0.83333333  1.        ], 'y/test/score_recall': [ 1.   0.8], 'y/test/score_accuracy': 0.9, 'y/test/score_f1': [ 0.90909091  0.88888889], 'y/test/score_recall_mean': 0.9}])

 
Running in Parallel
===================

In order to take advantage of multi-cores machine, EPAC can be run in parallel. We can first create a EPAC tree as below

::

    >>> # Perms + Cross-validation of SVM(linear) and SVM(rbf)
    >>> # -------------------------------------
    >>> #           Perms        Perm (Splitter)
    >>> #      /     |       \
    >>> #     0      1       2   Samples (Slicer)
    >>> #            |
    >>> #           CV           CV (Splitter)
    >>> #       /   |   \
    >>> #      0    1    2       Folds (Slicer)
    >>> #           |
    >>> #        Methods         Methods (Splitter)
    >>> #    /           \
    >>> # SVM(linear)  SVM(rbf)  Classifiers (Estimator) 
    >>> from sklearn.svm import SVC
    >>> from epac import Perms, CV, Methods
    >>> perms_cv_svm = Perms(CV(Methods(*[SVC(kernel="linear"), SVC(kernel="rbf")])))

You can use multi-processes to take advantage of multi-cores machine so that machine learning can be run more faster.

::

    >>> # Without multi-processes
    >>> # perms_cv_svm.run(X=X, y=y)
    >>> # perms_cv_svm.reduce()
    >>> # With multi-processes
    >>> from epac import LocalEngine
    >>> local_engine = LocalEngine(tree_root=perms_cv_svm, num_processes=2)
    >>> perms_cv_svm = local_engine.run(X=X, y=y)
    >>> perms_cv_svm.reduce() 

You can run your algorithms even on HPC on which DRMAA has been installed.

::

    >>> # Run with soma-workflow for multi-processes
    >>> from epac import SomaWorkflowEngine
    >>> sfw_engine = SomaWorkflowEngine(
    >>>                     tree_root=perms_cv_svm,
    >>>                     num_processes=2
    >>>                     #, resource_id="jl237561@gabriel",
    >>>                     # login="jl237561"
    >>>                     )
    >>> perms_cv_svm = sfw_engine.run(X=X, y=y)
    >>> perms_cv_svm.reduce()


Design your own plug-in
=======================

Design your own machine learning algorithm as a plug-in in EPAC tree.

::

   from sklearn.metrics import precision_recall_fscore_support
   from sklearn.svm import SVC
   from epac.map_reduce.reducers import Reducer
   from epac import Methods


   ## 1) Design your classifier
   ## =========================
   class MySVC:
       def __init__(self, C=1.0):
           self.C = C
       def transform(self, X, y):
           svc = SVC(C=self.C)
           svc.fit(X, y)
           # "transform" should return a dictionary
           return {"y/pred": svc.predict(X), "y": y}

   ## 2) Design your reducer which recall rate
   ## ========================================
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

   ## 3) Build a tree, and then compute results 
   ## =========================================
   my_svc1 = MySVC(C=1.0)
   my_svc2 = MySVC(C=2.0)
   two_svc = Methods(my_svc1, my_svc2)
   two_svc.reducer = MyReducer()
   #           Methods
   #          /      \
   # MySVC(C=1.0)  MySVC(C=2.0) 
   # top-down process to call transform
   two_svc.run(X=X, y=y)
   # buttom-up process to compute scores
   two_svc.reduce()

