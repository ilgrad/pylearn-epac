# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:04:34 2013

@author: edouard.duchesnay@cea.fr
"""

## ================================= ##
## == Configuration class         == ##
## ================================= ##
import numpy as np


class conf:
    TRACE_TOPDOWN = False
    REDUCE_TAB_FILENAME = "reduce_tab.csv"
    STORE_FS_PICKLE_SUFFIX = ".pkl"
    STORE_FS_JSON_SUFFIX = ".json"
    STORE_EXECUTION_TREE_PREFIX = "execution_tree"
    STORE_STORE_PREFIX = "store"
    SEP = "/"
    SUFFIX_JOB = "job"
    KW_SPLIT_TRAIN_TEST = "split_train_test"
    TRAIN = "train"
    TEST = "test"
    TRUE = "true"
    PREDICTION = "pred"
    SCORE_PRECISION = "score_precision"
    SCORE_RECALL = "score_recall"
    SCORE_RECALL_PVALUES = 'recall_pvalues'
    SCORE_RECALL_MEAN = "score_recall_mean"
    SCORE_RECALL_MEAN_PVALUE = 'recall_mean_pvalue'
    SCORE_F1 = "score_f1"
    SCORE_ACCURACY = "score_accuracy"
    BEST_PARAMS = "best_params"
    RESULT_SET = "result_set"
    MEMMAP = "memmap"
    MEMOBJ_SUFFIX = "_memobj.enpy"
    NOROBJ_SUFFIX = "_norobj.enpy"
    ML_CLASSIFICATION_MODE = None  # Set to True to force classification mode
    DICT_INDEX_FILE = "dict_index.txt"
    # when the data larger than 100MB, it needs memmory mapping
    MEMM_THRESHOLD = 100000000L
    # When split tree for parallel computing, the max depth we can split
    MAX_DEPTH_SPLIT_TREE = 4

    @classmethod
    def init_ml(cls, **Xy):
        ## Try to guess if ML tasl is of classification or regression
        # try to guess classif or regression task
        if cls.ML_CLASSIFICATION_MODE is None:
            if "y" in Xy:
                y = Xy["y"]
                y_int = y.astype(int)
                if not np.array_equal(y_int, y):
                    cls.ML_CLASSIFICATION_MODE = False
                if len(y_int.shape) > 1:
                    if y_int.shape[0] > y_int.shape[1]: 
                        y_int = y_int[:, 0]
                    else:
                        y_int = y_int[0, :]
                if np.min(np.bincount(y_int)) < 2:
                    cls.ML_CLASSIFICATION_MODE = False
                cls.ML_CLASSIFICATION_MODE = True


class debug:
    DEBUG = False
    current = None
