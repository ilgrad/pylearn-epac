# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:12:26 2013

Reducers for EPAC
@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import re
from abc import abstractmethod
from epac.map_reduce.results import Result
from epac.configuration import conf
from epac.workflow.base import key_push, key_pop
from epac.workflow.base import key_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

## ======================================================================== ##
## == Reducers                                                           == ##
## ======================================================================== ##


class Reducer:
    """ Reducer abstract class, called within the reduce method to process
    up-stream data flow of Result.

    Inherited classes should implement reduce(result)."""
    @abstractmethod
    def reduce(self, result):
        """Reduce abstract method

        Parameters
        ----------

        result: (dict)
            A result
        """


class ClassificationReport(Reducer):
    """Reducer that computes classification statistics.

    Parameters
    ----------
    
    select_regexp: srt
      A string to select items (defaults "test"). It must match two items:
      "true/test" and "pred/test".

    keep: boolean
      Should other items be kept (False) into summarized results.
      (default False)

    Example
    -------
    >>> from epac import ClassificationReport
    >>> reducer = ClassificationReport()
    >>> result = reducer.reduce({'key': "SVC", 'y/test/pred': [0, 1, 1, 1], 'y/test/true': [0, 0, 1, 1]})
    >>> result = result.items()
    >>> result.sort()
    >>> result
    [('key', 'SVC'), ('y/test/score_accuracy', 0.75), ('y/test/score_f1', array([ 0.66666667,  0.8       ])), ('y/test/score_precision', array([ 1.        ,  0.66666667])), ('y/test/score_recall', array([ 0.5,  1. ])), ('y/test/score_recall_mean', 0.75)]
    """
    
    def __init__(self, select_regexp=conf.TEST,
                 keep=False):
        self.select_regexp = select_regexp
        self.keep = keep

    def reduce(self, result):
        if self.select_regexp:
            inputs = [key3 for key3 in result
                if re.search(self.select_regexp, str(key3))]
        else:
            inputs = result.keys()
        if len(inputs) != 2:
            raise KeyError("Need to find exactly two results to compute a score."
            " Found %i: %s" % (len(inputs), inputs))
        key_true = [k for k in inputs if k.find(conf.TRUE) != -1][0]
        key_pred = [k for k in inputs if k.find(conf.PREDICTION) != -1][0]
        y_true = result[key_true]
        y_pred = result[key_pred]
        try:  # If list of arrays (CV, LOO, etc.) concatenate them
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
        except ValueError:
            pass
        out = Result(key=result["key"])
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None)
        key, _ = key_pop(key_pred, -1)
        out[key_push(key, conf.SCORE_PRECISION)] = p
        out[key_push(key, conf.SCORE_RECALL)] = r
        out[key_push(key, conf.SCORE_RECALL_MEAN)] = r.mean()
        out[key_push(key, conf.SCORE_F1)] = f1
        out[key_push(key, conf.SCORE_ACCURACY)] = accuracy_score(y_true, y_pred)
        if self.keep:
            out.update(result)
        return out


class PvalPerms(Reducer):
    """Reducer that computes p-values of stattistics.

    select_regexp: srt
      A string to select staitics (defaults ".*score.+"). on which to compute
      p-values.
    """
    def __init__(self, select_regexp='.*score.+',
                 keep=False):
        self.select_regexp = select_regexp
        self.keep = keep

    def reduce(self, result):
        if self.select_regexp:
            select_keys = [key for key in result
                if re.search(self.select_regexp, str(key))]
                #if re.search(self.select_regexp) != -1]
        else:
            select_keys = result.keys()
        out = Result(key=result.key())
        for key in select_keys:
            out[key] = result[key][0]
            randm_res = np.vstack(result[key][1:])
            count = np.sum(randm_res > result[key][0], axis=0).astype("float")
            pval = count / (randm_res.shape[0])
            out[key_push(key, "pval")] = pval
        if self.keep:
            out.update(result)
        return out


class CVBestSearchRefitPReducer(Reducer):
    def __init__(self, NodeBestSearchRefit):
        self.NodeBestSearchRefit = NodeBestSearchRefit

    def reduce(self, result):
        from epac.workflow.pipeline import Pipe
        #  Pump-up results
        cv_result_set = result
        key_val = [(result.key(), result[self.NodeBestSearchRefit.score]) \
                for result in cv_result_set]
        scores = np.asarray(zip(*key_val)[1])
        scores_opt = np.max(scores)\
            if self.NodeBestSearchRefit.arg_max else np.min(scores)
        idx_best = np.where(scores == scores_opt)[0][0]
        best_key = key_val[idx_best][0]
        # Find nodes that match the best
        nodes_dict = \
            {n.get_signature(): \
            n for n in self.NodeBestSearchRefit.children[0].walk_true_nodes() \
            if n.get_signature() in key_split(best_key)}
        to_refit = Pipe(*[nodes_dict[k].wrapped_node \
            for k in key_split(best_key)])
        best_params = [dict(sig) for sig in key_split(best_key, eval=True)]
        return to_refit, best_params
