# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:12:26 2013

Reducers for EPAC
@author: edouard.duchesnay@cea.fr
"""
import numpy as np
from scipy.stats import binom_test
import re
from abc import abstractmethod
from epac.map_reduce.results import Result
from epac.configuration import conf
from epac.workflow.base import key_push, key_pop, key_join, key_split
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
    >>> result = reducer.reduce({'key': "SVC", 'y/pred/test': [0, 1, 1, 1], 'y/true/test': [0, 0, 1, 1], 'y/pred/train': [0, 1, 1, 1]})
    >>> result = result.items()
    >>> result.sort()
    >>> result
    [('key', 'SVC'), ('y/recall_mean_pvalue/test', 0.625), ('y/recall_pvalues/test', array([ 0.5 ,  0.25])), ('y/score_accuracy/test', 0.75), ('y/score_f1/test', array([ 0.66666667,  0.8       ])), ('y/score_precision/test', array([ 1.        ,  0.66666667])), ('y/score_recall/test', array([ 0.5,  1. ])), ('y/score_recall_mean/test', 0.75)]
    """

    def __init__(self,# select_regexp=conf.TEST,
                 keep=False):
        #self.select_regexp = select_regexp
        self.keep = keep

    def reduce(self, result):
        from epac.utils import train_test_split
        from epac import key_contain_item
        
        result_train, result_test = train_test_split(result)
        key_true = [k for k in result_test if key_contain_item(k, conf.TRUE)]
        key_pred = [k for k in result_test if key_contain_item(k, conf.PREDICTION)]
        if len(key_true) != 1 or len(key_true) != 1:
            raise KeyError("Need to find exactly two results to compute a "
                           "score. Found %s %s" % \
                           (", ".join(key_true), ", ".join(key_true)))
        key_true = key_true[0]
        key_pred = key_pred[0]
        y_true = result_test[key_true]
        y_pred = result_test[key_pred]
#        if self.select_regexp:
#            
#            inputs = [key3 for key3 in result
#                      if re.search(self.select_regexp, str(key3))]
#        else:
#            inputs = result.keys()
#        if len(inputs) != 2:
#            raise KeyError("Need to find exactly two results to compute a "
#                           "score. Found %i: %s" % (len(inputs), inputs))
#                           
#        key_true = [k for k in inputs if k.find(conf.TRUE) != -1][0]
#        key_pred = [k for k in inputs if k.find(conf.PREDICTION) != -1][0]
#        y_true = result[key_true]
#        y_pred = result[key_pred]
        try:  # If list of arrays (CV, LOO, etc.) concatenate them
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
        except ValueError:
            pass
        out = Result(key=result["key"])
        p, r, f1, s = precision_recall_fscore_support(y_true,
                                                      y_pred,
                                                      average=None)

        # Compute p-value of recall for each class
        def recall_test(recall, n_trials, apriori_p):
            n_success = recall * n_trials
            pval = binom_test(n_success, n=n_trials, p=apriori_p)
            if recall > apriori_p:
                return (pval / 2)
            else:
                return 1 - (pval / 2)

        n_classes = len(s)  # Number of classes
        n_obs = len(y_true)
        prior_p = s.astype(np.float)/s.sum()  # A priori probability of each class
        r_pvalues = np.zeros_like(r)
        for class_index in range(n_classes):
            n_trials = s[class_index]
            #print "Class {class_index}: {n_success} success on {n_trials} trials".format(n_success=n_success, n_trials=n_trials, class_index=class_index)
            r_pvalues[class_index] = recall_test(r[class_index],
                                                 n_trials,
                                                 prior_p[class_index])

        # Compute p-value of mean recall
        mean_r = r.mean()
        mean_r_pvalue = binom_test(int(mean_r * n_obs), n=n_obs, p=.5)

        key, _ = key_pop(key_pred, 0)
        out[key_join(key, conf.SCORE_PRECISION, conf.TEST)] = p
        out[key_join(key, conf.SCORE_RECALL, conf.TEST)] = r
        out[key_join(key, conf.SCORE_RECALL_PVALUES, conf.TEST)] = r_pvalues
        out[key_join(key, conf.SCORE_RECALL_MEAN, conf.TEST)] = mean_r
        out[key_join(key, conf.SCORE_RECALL_MEAN_PVALUE, conf.TEST)] = mean_r_pvalue
        out[key_join(key, conf.SCORE_F1, conf.TEST)] = f1
        out[key_join(key, conf.SCORE_ACCURACY, conf.TEST)] = \
            accuracy_score(y_true, y_pred)
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
        key_val = [(result.key(), result[self.NodeBestSearchRefit.score])
                   for result in cv_result_set]
        scores = np.asarray(zip(*key_val)[1])
        scores_opt = np.max(scores)\
            if self.NodeBestSearchRefit.arg_max else np.min(scores)
        idx_best = np.where(scores == scores_opt)[0][0]
        best_key = key_val[idx_best][0]
        # Find nodes that match the best
        nodes_dict = \
            {n.get_signature():
             n for n in self.NodeBestSearchRefit.children[0].walk_true_nodes()
             if n.get_signature() in key_split(best_key)}
        to_refit = Pipe(*[nodes_dict[k].wrapped_node
                        for k in key_split(best_key)])
        best_params = [dict(sig) for sig in key_split(best_key, eval=True)]
        return to_refit, best_params

if __name__ == "__main__":
    import doctest
    doctest.testmod()
