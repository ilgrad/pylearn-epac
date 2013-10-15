#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Spliters divide the work to do into several parallel sub-tasks.
They are of two types data spliters (CV, Perms) or tasks
splitter (Methods, Grid).


@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

## Abreviations
## tr: train
## te: test
import collections
import numpy as np
import copy

from epac.workflow.base import BaseNode, key_push, key_pop
from epac.workflow.base import key_split
from epac.workflow.factory import NodeFactory
from epac.workflow.wrappers import Wrapper
from epac.stores import StoreMem
from epac.utils import train_test_split
from epac.utils import _list_indices, dict_diff, _sub_dict
from epac.utils import get_list_from_lists
from epac.utils import copy_parameters
from epac.map_reduce.results import Result, ResultSet
from epac.map_reduce.reducers import CVBestSearchRefitPReducer
from epac.map_reduce.reducers import ClassificationReport, PvalPerms
from epac.configuration import conf
from epac import Pipe
import warnings

## ======================================================================== ##
## ==                                                                    == ##
## == Parallelization nodes
## ==
## ======================================================================== ##


# -------------------------------- #
# -- Splitter                   -- #
# -------------------------------- #


class BaseNodeSplitter(BaseNode):
    """Splitters are are non leaf node (degree >= 1) with children.
    They split the downstream data-flow to their children.
    They agregate upstream data-flow from their children.
    """
    def __init__(self, need_group_key=True):
        super(BaseNodeSplitter, self).__init__()
        self.need_group_key = need_group_key

    def reduce(self, store_results=True):
        # Terminaison (leaf) node return results
        if not self.children:
            return self.load_results()
        # 1) Build sub-aggregates over children
        children_results = [child.reduce(store_results=False) for
                            child in self.children]
        result_set = ResultSet(*children_results)
        if not self.reducer:
            return result_set

        if not self.need_group_key:
            reduced = ResultSet()
            reduced.add(self.reducer.reduce(result_set))
            return reduced

        # Group by key, without consideration of the fold/permutation number
        # which is the head of the key
        # use OrderedDict to preserve runing order
        from collections import OrderedDict
        groups = OrderedDict()
        for result in result_set:
            # remove the head of the key
            _, key_tail = key_pop(result["key"], index=0)
            result["key"] = key_tail
            if not key_tail in groups:
                groups[key_tail] = list()
            groups[key_tail].append(result)
        # For each key, stack results
        reduced = ResultSet()
        for key in groups:
            result_stacked = Result.stack(*groups[key])
            reduced.add(self.reducer.reduce(result_stacked))
        return reduced


class CV(BaseNodeSplitter):
    """Cross-validation parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Pipe | Par*

    n_folds: int
        Number of folds. (Default 5)

    cv_type: string
        Values: "stratified", "random", "loo". Default "stratified".

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    reducer: Reducer
        A Reducer should implement the reduce(node, key2, val) method.
        Default ClassificationReport() with default arguments.

    Example
    -------
    >>> from sklearn import datasets
    >>> from epac import CV, Methods
    >>> from sklearn.lda import LDA
    >>> from sklearn.svm import LinearSVC as SVM
    >>> X, y = datasets.make_classification(n_samples=20,
    ...                                     n_features=20,
    ...                                     n_informative=2,
    ...                                     random_state=1)
    >>> cv = CV(Methods(LDA(), SVM()))
    >>> res = cv.run(X=X, y=y)
    >>> print cv.reduce()
    ResultSet(
    [{'key': LDA, 'y/test/score_precision': [ 0.33333333  0.36363636], 'y/test/score_recall': [ 0.3  0.4], 'y/test/score_accuracy': 0.35, 'y/test/score_f1': [ 0.31578947  0.38095238], 'y/test/score_recall_mean': 0.35},
     {'key': LinearSVC, 'y/test/score_precision': [ 0.5  0.5], 'y/test/score_recall': [ 0.6  0.4], 'y/test/score_accuracy': 0.5, 'y/test/score_f1': [ 0.54545455  0.44444444], 'y/test/score_recall_mean': 0.5}])

     """

    def __init__(self, node, n_folds=5, random_state=None,
                 cv_type="stratified", reducer=ClassificationReport(),
                 **kwargs):
        super(CV, self).__init__(**kwargs)
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_type = cv_type
        self.reducer = reducer
        self.slicer = CRSlicer(signature_name="CV",
                               nb=0,
                               apply_on=None,
                               col_or_row=False)
        self.children = VirtualList(size=n_folds,
                                    parent=self)
        self.slicer.parent = self
        subtree = NodeFactory.build(node)
        # subtree = node if isinstance(node, BaseNode) else LeafEstimator(node)
        self.slicer.add_child(subtree)

    def move_to_child(self, nb):
        self.slicer.set_nb(nb)
        if hasattr(self, "_sclices"):
            cpt = 0
            for train, test in self._sclices:
                if cpt == nb:
                    break
                cpt += 1
            self.slicer.set_sclices({conf.TRAIN: train, conf.TEST: test})
        return self.slicer

    def transform(self, **Xy):
        # Set the slicing
        if not "y" in Xy:
            raise ValueError('"y" should be provided')
        if self.cv_type == "stratified":
            if not self.n_folds:
                raise ValueError('"n_folds" should be set')
            from sklearn.cross_validation import StratifiedKFold
            self._sclices = StratifiedKFold(y=Xy["y"], n_folds=self.n_folds)
        elif self.cv_type == "random":
            if not self.n_folds:
                raise ValueError('"n_folds" should be set')
            from sklearn.cross_validation import KFold
            self._sclices = KFold(n=Xy["y"].shape[0], n_folds=self.n_folds,
                                  random_state=self.random_state)
        elif self.cv_type == "loo":
            from sklearn.cross_validation import LeaveOneOut
            self._sclices = LeaveOneOut(n=Xy["y"].shape[0])
        return Xy

    def get_parameters(self):
        return dict(n_folds=self.n_folds)


class Perms(BaseNodeSplitter):
    """Permutation parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Pipe | Par*

    n_perms: int
        Number permutations.

    reducer: Reducer
        A Reducer should inmplement the reduce(key2, val) method.

    permute: string
        The name of the data to be permuted (default "y").

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    reducer: Reducer
        A Reducer should inmplement the reduce(key2, val) method.

    col_or_row: boolean value
            If col_or_row is True means that column permutation,
            If col_or_row is False means that row permutation
    """
    def __init__(self, node, n_perms=100, permute="y", random_state=None,
                 reducer=PvalPerms(), col_or_row=False, **kwargs):
        super(Perms, self).__init__(**kwargs)
        self.n_perms = n_perms
        self.permute = permute  # the name of the bloc to be permuted
        self.random_state = random_state
        self.reducer = reducer
        self.slicer = CRSlicer(signature_name="Perm",
                               nb=0,
                               apply_on=permute,
                               col_or_row=col_or_row)
        self.children = VirtualList(size=n_perms,
                                    parent=self)
        self.slicer.parent = self
        subtree = NodeFactory.build(node)
        # subtree = node if isinstance(node, BaseNode) else LeafEstimator(node)
        self.slicer.add_child(subtree)
        self.col_or_row = col_or_row

    def move_to_child(self, nb):
        self.slicer.set_nb(nb)
        if hasattr(self, "_sclices"):
            cpt = 0
            for perm in self._sclices:
                if cpt == nb:
                    break
                cpt += 1
            self.slicer.set_sclices({self.permute: perm})
        return self.slicer

    def get_parameters(self):
        return dict(n_perms=self.n_perms, permute=self.permute)

    def transform(self, **Xy):
        # Set the slicing
        if not self.permute in Xy:
            raise ValueError('"%s" should be provided' % self.permute)
        from epac.sklearn_plugins import Permutations
        if len(Xy[self.permute].shape) == 2:
            if not self.col_or_row:
                self._sclices = Permutations(n=Xy[self.permute].shape[0],
                                             n_perms=self.n_perms,
                                             random_state=self.random_state)
            else:
                self._sclices = Permutations(n=Xy[self.permute].shape[1],
                                             n_perms=self.n_perms,
                                             random_state=self.random_state)
        else:
            self._sclices = Permutations(n=Xy[self.permute].shape[0],
                                         n_perms=self.n_perms,
                                         random_state=self.random_state)
        return Xy


class Methods(BaseNodeSplitter):
    """Parallelization is based on several runs of different methods
    """
    def __init__(self, *nodes):
        super(Methods, self).__init__()
        for node in nodes:
            node_cp = copy.deepcopy(node)
            node_cp = NodeFactory.build(node_cp)
            self.add_child(node_cp)
        curr_nodes = self.children
        leaves_key = [l.get_key() for l in self.walk_leaves()]
        curr_nodes_key = [c.get_key() for c in curr_nodes]
        while len(leaves_key) != len(set(leaves_key)) and curr_nodes:
            curr_nodes_state = [c.get_parameters() for c in curr_nodes]
            curr_nodes_next = list()
            for key in set(curr_nodes_key):
                collision_indices = _list_indices(curr_nodes_key, key)
                if len(collision_indices) == 1:  # no collision for this cls
                    continue
                diff_arg_keys = dict_diff(*[curr_nodes_state[i] for i
                                            in collision_indices]).keys()
                for curr_node_idx in collision_indices:
                    if diff_arg_keys:
                        curr_nodes[curr_node_idx].signature_args = \
                            _sub_dict(curr_nodes_state[curr_node_idx],
                                      diff_arg_keys)
                    curr_nodes_next += curr_nodes[curr_node_idx].children
            curr_nodes = curr_nodes_next
            curr_nodes_key = [c.get_key() for c in curr_nodes]
            leaves_key = [l.get_key() for l in self.walk_leaves()]
        leaves_key = [l.get_key() for l in self.walk_leaves()]
        if len(leaves_key) != len(set(leaves_key)):
            raise ValueError("Some methods are identical, they could not be "
                             "differentiated according to their arguments")

    def transform(self, **Xy):
        return Xy

    def reduce(self, store_results=True):
        # 1) Build sub-aggregates over children
        children_results = [child.reduce(store_results=False) for
                            child in self.children]
        results = ResultSet(*children_results)
        if self.reducer:
            return self.reducer.reduce(results)
        return results


class WarmStartMethods(Methods):
    """Run like methods but with previous state for initialization
    """
    def __init__(self, *nodes):
        super(WarmStartMethods, self).__init__(*nodes)
        self.stop_top_down = True

    def transform(self, **Xy):
        prev_node = None
        rets = []
        for node in self.children:
            cpXy = Xy
            if not (prev_node is None):
                # from_obj, to_obj, exclude_parameters
                copy_parameters(from_obj=prev_node.wrapped_node,
                                to_obj=node.wrapped_node,
                                exclude_parameters=node.signature_args)
            ret = node.top_down(**cpXy)
            rets.append(ret)
            prev_node = node
        if len(rets) > 0:
            Xy = rets[0] if len(rets) == 1 else rets
        return Xy


# -------------------------------- #
# -- Slicers                    -- #
# -------------------------------- #


class VirtualList(collections.Sequence):
    def __init__(self, size, parent):
        self.size = size
        self.parent = parent

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i >= self.size:
            raise IndexError("%s index out of range" % self.__class__.__name__)
        return self.parent.move_to_child(nb=i)
        #return self.parent.move_to_child(i, self.slicer)

    def __iter__(self):
        """ Iterate over leaves"""
        for i in xrange(self.size):
            yield self.__getitem__(i)

    def append(self, o):
        pass


class Slicer(BaseNode):
    """ Slicers are Splitters' children, they re-sclice the downstream blocs.
    """
    def __init__(self, signature_name, nb):
        super(Slicer, self).__init__()
        self.signature_name = signature_name
        self.signature_args = dict(nb=nb)

    def set_nb(self, nb):
        self.signature_args["nb"] = nb

    def get_parameters(self):
        return dict(slices=self.slices)

    def get_signature(self, nb=1):
        """Overload the base name method: use self.signature_name"""
        return self.signature_name + \
            "(nb=" + str(self.signature_args["nb"]) + ")"

    def get_signature_args(self):
        """overried get_signature_args to return a copy"""
        return copy.copy(self.signature_args)

    def reduce(self, store_results=True):
        results = ResultSet(self.children[0].reduce(store_results=False))
        for result in results:
            result["key"] = key_push(self.get_signature(), result["key"])
        return results


class CRSlicer(Slicer):
    """column or row sampling
    Parameters
    ----------
    signature_name: string

    nb: integer
        nb is used for the key value that distinguishs thier sibling node

    apply_on: string or list of strings (or None)
        The name(s) of the downstream blocs to be re-slicing. If
        None, all downstream blocs are sampling (slicing).

    col_or_row: boolean value
        If col_or_row is True means that column splitter,
        If col_or_row is False means that row splitter
    """

    def __init__(self, signature_name, nb, apply_on, col_or_row=True):
        super(CRSlicer, self).__init__(signature_name, nb)
        self.slices = None
        self.col_or_row = col_or_row
        if not apply_on:  # None is an acceptable value here
            self.apply_on = apply_on
        elif isinstance(apply_on, list):
            self.apply_on = apply_on
        elif isinstance(apply_on, str):
            self.apply_on = [apply_on]
        else:
            raise ValueError("apply_on must be a string or a "
                             "list of strings or None")

    def set_sclices(self, slices):
        """
        """
        # convert as a list if required
        if isinstance(slices, dict):
            self.slices =\
                {k: slices[k].tolist() if isinstance(slices[k], np.ndarray)
                 else slices[k] for k in slices}
        else:
            self.slices = \
                slices.tolist() if isinstance(slices, np.ndarray) else slices

    def transform(self, **Xy):
        if not self.slices:
            raise ValueError("Slicing hasn't been initialized. ")
        data_keys = self.apply_on if self.apply_on else Xy.keys()
        for slice_key in self.slices.keys():
            if slice_key in data_keys:
                data_key = slice_key
                dat = Xy.pop(data_key)
                if len(dat.shape) == 2:
                    if self.col_or_row:
                        Xy[data_key] = dat[:, self.slices[data_key]]
                    else:
                        Xy[data_key] = dat[self.slices[data_key], :]
                else:
                    Xy[data_key] = dat[self.slices[data_key]]
        # only for cross-validation
        if conf.TRAIN in self.slices.keys() \
                and conf.TEST in self.slices.keys():
            Xy[conf.KW_SPLIT_TRAIN_TEST] = True
            for data_key in data_keys:
                dat = Xy.pop(data_key)
                for sample_set in self.slices:
                    if len(dat.shape) == 2:
                        if self.col_or_row:
                            Xy[key_push(data_key, sample_set)] = \
                                dat[:, self.slices[sample_set]]
                        else:
                            Xy[key_push(data_key, sample_set)] = \
                                dat[self.slices[sample_set], :]
                    else:
                        Xy[key_push(data_key, sample_set)] = \
                            dat[self.slices[sample_set]]
        return Xy


class CRSplitter(BaseNodeSplitter):
    """Column or Row Splitter parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Pipe | Par*

    indices_of_groups: dictionary
        The name of the data to be splited and its gourp indices

    col_or_row: boolean value
        If col_or_row is True means that column splitter,
        If col_or_row is False means that row splitter

    Example
    -------
    See Example in ColumnSplitter

    """

    def __init__(self, node, indices_of_groups, col_or_row=True):
        super(CRSplitter, self).__init__()
        self.indices_of_groups = indices_of_groups
        self.slicer = CRSlicer(signature_name=self.__class__.__name__,
                               nb=0,
                               apply_on=None,
                               col_or_row=col_or_row)

        self.uni_indices_of_groups = {}
        for key_indices_of_groups in indices_of_groups:
            self.uni_indices_of_groups[key_indices_of_groups] = \
                list(set(indices_of_groups[key_indices_of_groups]))

        self.size = 1
        for key_indices_of_groups in self.uni_indices_of_groups:
            tmp_data = list(self.uni_indices_of_groups[key_indices_of_groups])
            self.size = self.size * len(tmp_data)

        self.children = VirtualList(size=self.size, parent=self)
        self.slicer.parent = self
        subtree = NodeFactory.build(node)
        # subtree = node if isinstance(node, BaseNode) else LeafEstimator(node)
        self.slicer.add_child(subtree)

    def convert_dict2list(self, dict_data):
        list_data = []
        for key in dict_data:
            list_data.append(dict_data[key])
        return list_data

    def move_to_child(self, nb):
        self.slicer.set_nb(nb)
        lists = self.convert_dict2list(self.uni_indices_of_groups)
        list_data = get_list_from_lists(lists, nb)
        i = 0
        slices = {}
        for key in self.uni_indices_of_groups:
            indices = np.nonzero(np.asarray(self.indices_of_groups[key]) ==
                                 np.asarray(list_data[i]))
            indices = indices[0]
            slices[key] = indices
            i += 1
        self.slicer.set_sclices(slices)
        return self.slicer

    def transform(self, **Xy):
        self._sclices = None
        return Xy

    def get_parameters(self):
        return dict(size=self.size)


class RowSplitter(CRSplitter):
    """Column Splitter parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Pipe | Par*

    indices_of_groups: dictionary
        group index for data

    Example
    -------
    See Example in ColumnSplitter
    """

    def __init__(self, node, indices_of_groups):
        super(RowSplitter, self).__init__(node,
                                          indices_of_groups,
                                          col_or_row=False)


class ColumnSplitter(CRSplitter):
    """Column Splitter parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Pipe | Par*

    indices_of_groups: dictionary
        group index for data

    Example
    -------
    >>> import numpy as np
    >>> from epac import ColumnSplitter
    >>> class TestNode:
    ...     def transform(self, X, Y):
    ...         print "---------------------"
    ...         print "X=", X.shape
    ...         print "Y=", Y.shape
    ...         return {"X": X, "Y": Y}
    ...
    >>> n_samples = 10
    >>> n_xfeatures = 11
    >>> n_yfeatures = 12
    >>> X = np.random.randn(n_samples, n_xfeatures)
    >>> Y = np.random.randn(n_samples, n_yfeatures)
    >>> print "1. We want to split X into 3 groups (0, 1, 2) column by column"
    1. We want to split X into 3 groups (0, 1, 2) column by column
    >>> print "Therefore we need to define a group indices for X"
    Therefore we need to define a group indices for X
    >>> X_group_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    >>> print 'The key of "X" denotes the data you want to split'
    The key of "X" denotes the data you want to split
    >>> print 'X_group_indices denotes how to split by group indices'
    X_group_indices denotes how to split by group indices
    >>> indices_of_groups = {"X": X_group_indices}
    >>> print 'Build spliter for TestNode'
    Build spliter for TestNode
    >>> column_splitter = ColumnSplitter(TestNode(),
    ...                                  indices_of_groups)
    >>> print 'Run top-down process'
    Run top-down process
    >>> print "*********************************"
    *********************************
    >>> print "Split X"
    Split X
    >>> res = column_splitter.run(X=X, Y=Y)
    ---------------------
    X= (10, 4)
    Y= (10, 12)
    ---------------------
    X= (10, 3)
    Y= (10, 12)
    ---------------------
    X= (10, 4)
    Y= (10, 12)
    >>> print '2. Similarily we want to split Y'
    2. Similarily we want to split Y
    >>> Y_group_indices = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    >>> indices_of_groups = {"Y": Y_group_indices}
    >>> column_splitter = ColumnSplitter(TestNode(),
    ...                                  indices_of_groups)
    >>> print "*********************************"
    *********************************
    >>> print "Split Y"
    Split Y
    >>> res = column_splitter.run(X=X, Y=Y)
    ---------------------
    X= (10, 11)
    Y= (10, 3)
    ---------------------
    X= (10, 11)
    Y= (10, 2)
    ---------------------
    X= (10, 11)
    Y= (10, 4)
    ---------------------
    X= (10, 11)
    Y= (10, 3)
    >>> print '3. We want to split X and Y'
    3. We want to split X and Y
    >>> indices_of_groups = {"X": X_group_indices,
    ...                      "Y": Y_group_indices}
    >>> column_splitter = ColumnSplitter(TestNode(),
    ...                                  indices_of_groups)
    >>> print "*********************************"
    *********************************
    >>> print "Split X and Y"
    Split X and Y
    >>> res = column_splitter.run(X=X, Y=Y)
    ---------------------
    X= (10, 4)
    Y= (10, 3)
    ---------------------
    X= (10, 4)
    Y= (10, 2)
    ---------------------
    X= (10, 4)
    Y= (10, 4)
    ---------------------
    X= (10, 4)
    Y= (10, 3)
    ---------------------
    X= (10, 3)
    Y= (10, 3)
    ---------------------
    X= (10, 3)
    Y= (10, 2)
    ---------------------
    X= (10, 3)
    Y= (10, 4)
    ---------------------
    X= (10, 3)
    Y= (10, 3)
    ---------------------
    X= (10, 4)
    Y= (10, 3)
    ---------------------
    X= (10, 4)
    Y= (10, 2)
    ---------------------
    X= (10, 4)
    Y= (10, 4)
    ---------------------
    X= (10, 4)
    Y= (10, 3)

    """

    def __init__(self, node, indices_of_groups):
        super(ColumnSplitter, self).__init__(node,
                                             indices_of_groups,
                                             col_or_row=True)


class CVBestSearchRefitParallel(Wrapper):
    """Cross-validation + grid-search then refit with optimals parameters.

    Average results over first axis, then find the arguments that maximize or
    minimise a "score" over other axis.

    Parameters
    ----------

    See CV parameters, plus other parameters:

    score: string
        the score name to be optimized (default "mean_score_te").

    arg_max: boolean
        True/False take parameters that maximize/minimize the score. Default
        is True.

    Example
    -------

    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from epac import Methods
    >>> from epac.workflow.splitters import CVBestSearchRefitParallel
    >>> X, y = datasets.make_classification(n_samples=12,
    ...                                     n_features=10,
    ...                                     n_informative=2,
    ...                                     random_state=1)
    >>> n_folds_nested = 2
    >>> C_values = [.1, 0.5, 1, 2, 5]
    >>> kernels = ["linear", "rbf"]
    >>> methods = Methods(*[SVC(C=C, kernel=kernel)
    ...     for C in C_values for kernel in kernels])
    >>> wf = CVBestSearchRefitParallel(methods, n_folds=n_folds_nested)
    >>> wf.run(X=X, y=y)
    [[{'y/test/pred': array([0, 0, 1, 0, 0, 0]), 'y/train/pred': array([0, 0, 0, 0, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}, {'y/test/pred': array([1, 1, 1, 0, 1, 1]), 'y/train/pred': array([0, 0, 1, 1, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}, {'y/test/pred': array([1, 1, 1, 0, 1, 1]), 'y/train/pred': array([0, 0, 1, 1, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}, {'y/test/pred': array([1, 1, 1, 0, 1, 1]), 'y/train/pred': array([0, 0, 1, 1, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}, {'y/test/pred': array([1, 1, 1, 0, 1, 1]), 'y/train/pred': array([0, 0, 1, 1, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}, {'y/test/pred': array([1, 1, 1, 0, 1, 1]), 'y/train/pred': array([0, 0, 1, 1, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}, {'y/test/pred': array([1, 1, 1, 0, 1, 1]), 'y/train/pred': array([0, 0, 1, 1, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}, {'y/test/pred': array([1, 1, 1, 0, 1, 1]), 'y/train/pred': array([0, 0, 1, 1, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}, {'y/test/pred': array([1, 1, 1, 0, 1, 1]), 'y/train/pred': array([0, 0, 1, 1, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}, {'y/test/pred': array([1, 1, 1, 0, 1, 1]), 'y/train/pred': array([0, 0, 1, 1, 0, 1]), 'y/test/true': array([1, 0, 0, 1, 0, 1])}], [{'y/test/pred': array([0, 1, 1, 0, 1, 1]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}, {'y/test/pred': array([0, 1, 1, 1, 1, 1]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}, {'y/test/pred': array([0, 1, 0, 0, 1, 1]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}, {'y/test/pred': array([0, 1, 1, 1, 1, 1]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}, {'y/test/pred': array([0, 1, 0, 0, 1, 1]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}, {'y/test/pred': array([0, 1, 1, 1, 1, 1]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}, {'y/test/pred': array([0, 1, 0, 0, 1, 1]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}, {'y/test/pred': array([0, 1, 1, 0, 1, 0]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}, {'y/test/pred': array([0, 1, 0, 0, 1, 1]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}, {'y/test/pred': array([0, 1, 1, 0, 1, 0]), 'y/train/pred': array([1, 0, 0, 1, 0, 1]), 'y/test/true': array([0, 0, 1, 1, 0, 1])}]]
    >>> wf.reduce()
    ResultSet(
    [{'key': CVBestSearchRefitParallel, 'best_params': [{'kernel': 'rbf', 'C': 0.1, 'name': 'SVC'}], 'y/true': [1 0 0 1 0 0 1 0 1 1 0 1], 'y/pred': [1 0 0 1 0 0 1 0 1 1 0 1]}])

    """

    def __init__(self, node, **kwargs):
        super(CVBestSearchRefitParallel, self).__init__(wrapped_node=None)
        #### 'y/test/score_recall_mean'
        default_score = "y" + conf.SEP + \
                        conf.TEST + conf.SEP + \
                        conf.SCORE_RECALL_MEAN
        score = kwargs.pop("score") if "score" in kwargs else default_score
        arg_max = kwargs.pop("arg_max") if "arg_max" in kwargs else True
        from epac.workflow.splitters import CV
        # methods = Methods(*tasks)
        cv_node = CV(node=node,
                     reducer=ClassificationReport(keep=False),
                     **kwargs)
        self.add_child(cv_node)
        self.score = score
        self.arg_max = arg_max
        self.refited = None
        self.best_params = None
        self.reducer = CVBestSearchRefitPReducer(self)

    def get_signature(self):
        return self.__class__.__name__

    def transform(self, **Xy):
        Xy_train, Xy_test = train_test_split(Xy)
        result = Result(key=self.get_signature(), **Xy)
        if not self.store:
            self.store = StoreMem()
        self.save_results(ResultSet(result))
        if Xy_train is Xy_test:
            return Xy
        else:
            return Xy_train

    def _results2dict(self, **cpXy):
        res_dict = {}
        for key in cpXy[self.get_signature()]:
            if not key == "key":
                res_dict[key] = cpXy[self.get_signature()][key]
        return res_dict

    def reduce(self, store_results=True):
        children_results = [child.reduce(store_results=False) for
                            child in self.children]
        results = ResultSet(*children_results)
        if self.reducer:
            to_refit, best_params = self.reducer.reduce(results)
            Xy = self.load_results()
            Xy = self._results2dict(**Xy)
            self.refited = to_refit
            self.best_params = best_params
            out = self.refited.top_down(**Xy)
            out[conf.BEST_PARAMS] = best_params
            result = Result(key=self.get_signature(), **out)
            return ResultSet(result)
        return results


class CVBestSearchRefit(Wrapper):
    """Cross-validation + grid-search then refit with optimals parameters.

    Average results over first axis, then find the arguments that maximize or
    minimise a "score" over other axis.

    Parameters
    ----------

    See CV parameters, plus other parameters:

    score: string
        the score name to be optimized (default "mean_score_te").

    arg_max: boolean
        True/False take parameters that maximize/minimize the score. Default
        is True.

    Example
    -------
    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from epac import Methods
    >>> from epac.workflow.splitters import CVBestSearchRefit
    >>> X, y = datasets.make_classification(n_samples=12,
    ... n_features=10,
    ... n_informative=2,
    ... random_state=1)
    >>> n_folds_nested = 2
    >>> C_values = [.1, 0.5, 1, 2, 5]
    >>> kernels = ["linear", "rbf"]
    >>> methods = Methods(*[SVC(C=C, kernel=kernel)
    ...     for C in C_values for kernel in kernels])
    >>> wf = CVBestSearchRefit(methods, n_folds=n_folds_nested)
    >>> wf.transform(X=X, y=y)
    {'best_params': [{'kernel': 'rbf', 'C': 0.1, 'name': 'SVC'}], 'y/true': array([1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]), 'y/pred': array([1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1])}
    >>> wf.reduce()
    >>> wf.run(X=X, y=y)
    {'best_params': [{'kernel': 'rbf', 'C': 0.1, 'name': 'SVC'}], 'y/true': array([1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]), 'y/pred': array([1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1])}
    >>> wf.reduce()
    ResultSet(
    [{'key': CVBestSearchRefit, 'best_params': [{'kernel': 'rbf', 'C': 0.1, 'name': 'SVC'}], 'y/true': [1 0 0 1 0 0 1 0 1 1 0 1], 'y/pred': [1 0 0 1 0 0 1 0 1 1 0 1]}])

    """

    def __init__(self, node, **kwargs):
        super(CVBestSearchRefit, self).__init__(wrapped_node=None)
        #### 'y/test/score_recall_mean'
        default_score = "y" + conf.SEP + \
                        conf.TEST + conf.SEP + \
                        conf.SCORE_RECALL_MEAN
        score = kwargs.pop("score") if "score" in kwargs else default_score
        arg_max = kwargs.pop("arg_max") if "arg_max" in kwargs else True
        from epac.workflow.splitters import CV
        #methods = Methods(*tasks)
        self.cv = CV(node=node, reducer=ClassificationReport(keep=False),
                     **kwargs)
        self.score = score
        self.arg_max = arg_max
#        warnings.warn("%s is deprecated. Please use %s instead." % \
#                        (self.__class__.__name__,\
#                         CVBestSearchRefitParallel.__name__),
#                        category=DeprecationWarning)

    def get_signature(self):
        return self.__class__.__name__

    def transform(self, **Xy):
        Xy_train, Xy_test = train_test_split(Xy)
        if Xy_train is Xy_test:
            to_refit, best_params = self._search_best(**Xy)
        else:
            to_refit, best_params = self._search_best(**Xy_train)
        out = to_refit.top_down(**Xy)
        out[conf.BEST_PARAMS] = best_params
        self.refited = to_refit
        self.best_params = best_params
        return out

    def _search_best(self, **Xy):
        # Fit/predict CV grid search
        self.cv.store = StoreMem()  # local store erased at each fit
        from epac.workflow.pipeline import Pipe
        self.cv.top_down(**Xy)
        #  Pump-up results
        cv_result_set = self.cv.reduce(store_results=False)
        key_val = [(result.key(), result[self.score])
                   for result in cv_result_set]
        scores = np.asarray(zip(*key_val)[1])
        scores_opt = np.max(scores) if self.arg_max else np.min(scores)
        idx_best = np.where(scores == scores_opt)[0][0]
        best_key = key_val[idx_best][0]
        # Find nodes that match the best
        nodes_dict = {n.get_signature(): n for n in self.cv.walk_true_nodes()
                      if n.get_signature() in key_split(best_key)}
        to_refit = Pipe(*[nodes_dict[k].wrapped_node
                          for k in key_split(best_key)])
        best_params = [dict(sig) for sig in key_split(best_key, eval=True)]
        return to_refit, best_params

    def reduce(self, store_results=True):
        # Terminaison (leaf) node return result_set
        return self.load_results()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
