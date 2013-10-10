# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:56:37 2013

@author: laure.hugo@cea.fr
"""

import numpy as np
from sklearn import datasets
import random
import time
#import datetime
import sys
import os
#import tempfile
import unittest
import dill as pickle
#import getopt
from epac import StoreFs

from epac.tests.utils import isequal, compare_two_node


def create_mmat(nrows, ncols, default_values=None, dir=None, n_proc=None):
    filename = 'tmp_rows_' + str(nrows) + '_cols_' + str(ncols)
    if dir is None:
        filepath = '/tmp/' + filename
    else:
        if not os.path.isdir(dir):
            os.mkdir(dir)
        filepath = os.path.join(dir, filename)

    if n_proc == 1:
        mem_mat = np.memmap(filepath,
                            dtype='float32',
                            mode='w+',
                            shape=(nrows, ncols))
        for i in xrange(nrows):
            if not default_values:
                mem_mat[i, :] = np.random.random(size=ncols)
            elif type(default_values) is list:
                insert_row = np.zeros(ncols)
                for j in xrange(len(insert_row)):
                    pos = random.randint(0, len(default_values) - 1)
                    insert_row[j] = default_values[pos]
                mem_mat[i, :] = insert_row
    else:
        mem_mat = np.memmap(filepath,
                            dtype='float32',
                            mode='r+',
                            shape=(nrows, ncols))
    return mem_mat


def create_array(size, default_values=None, dir=None, n_proc=None):
    ret_array = create_mmat(size, 1, default_values, dir=dir, n_proc=n_proc)
    ret_array = ret_array[:, 0]
    return ret_array


# @profile
class TestMemMapping(unittest.TestCase):
    ''' Test the capacity of the computer

    Parameters
    ----------
    n_samples: int
        Number of rows of the X matrix

    n_features: int
        Number of columns of the X matrix

    memmap: boolean
        If True, use memory mapping to reduce memory cost

    n_proc: int
        Number of processes

    is_swf: boolean
        If True, run the processes on the cluster
        If False, run on the local machine

    dir: directory path
        Path of the directory where you want to save the temporary files
        If None, save in /tmp
    '''

    def __init__(self, n_samples, n_features, memmap=False,
                 n_proc=1, is_swf=False, dir=None, testname='test_memmapping'):
        super(TestMemMapping, self).__init__(testname)
        self.n_samples = n_samples
        self.n_features = n_features
        self.memmap = memmap
        self.n_proc = n_proc
        self.is_swf = is_swf
        self.dir = dir

    def test_memmapping(self):
        ## 1) Building dataset
        ## ============================================================
        if self.memmap:
            X = create_mmat(self.n_samples, self.n_features, dir=self.dir,
                            n_proc=self.n_proc)
            y = create_array(self.n_samples, [0, 1], dir=self.dir,
                             n_proc=self.n_proc)
            Xy = dict(X=X, y=y)
        else:
            X, y = datasets.make_classification(n_samples=self.n_samples,
                                                n_features=self.n_features,
                                                n_informative=2,
                                                random_state=1)
            Xy = dict(X=X, y=y)
        ## 2) Building workflow
        ## =======================================================
        from sklearn.svm import SVC
        from epac import CV, Methods
        cv_svm_local = CV(Methods(*[SVC(kernel="linear"),
                                    SVC(kernel="rbf")]), n_folds=3)

        cv_svm = None
        if self.is_swf:
            # Running on the cluster
            from epac import SomaWorkflowEngine
            mmap_mode = None
            if self.memmap:
                mmap_mode = "r+"
            swf_engine = SomaWorkflowEngine(cv_svm_local,
                                            num_processes=self.n_proc,
                                            resource_id="jl237561@gabriel",
                                            login="jl237561",
#                                            remove_finished_wf=False,
#                                            remove_local_tree=False,
                                            mmap_mode=mmap_mode,
                                            queue="Global_long")

            cv_svm = swf_engine.run(**Xy)
            time.sleep(2)
            print ''
            sum_memory = 0
            max_time_cost = 0
            for job_info in swf_engine.engine_info:
                print "mem_cost=", job_info.mem_cost, \
                      ", vmem_cost=", job_info.vmem_cost, \
                      ", time_cost=", job_info.time_cost
                sum_memory += job_info.mem_cost
                if max_time_cost < job_info.time_cost:
                    max_time_cost = job_info.time_cost
            print "sum_memory =", sum_memory
            print "max_time_cost =", max_time_cost
        else:
            # Running on the local machine
            from epac import LocalEngine
            local_engine = LocalEngine(cv_svm_local, num_processes=self.n_proc)
            cv_svm = local_engine.run(**Xy)

        cv_svm_reduce = cv_svm.reduce()
        print "\n -> Reducing results"
        print cv_svm_reduce

        dirpath = "/tmp/tmp_save_tree/"
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)

        if self.n_proc == 1:
            ## 4.1) Saving results on the disk for one process
            ## ===================================================
            store = StoreFs(dirpath=dirpath, clear=True)
            cv_svm.save_tree(store=store)

            with open("/tmp/tmp_save_results", 'w+') as filename:
                pickle.dump(cv_svm_reduce, filename)

        else:
            ## 4.2) Compare results to the results for one process
            ## ===================================================
            store = StoreFs(dirpath=dirpath, clear=False)
            cv_svm_one_proc = store.load()

            with open("/tmp/tmp_save_results", 'r+') as filename:
                cv_svm_reduce_one_proc = pickle.load(filename)

            print "\nComparing %i proc with one proc" % self.n_proc
            self.assertTrue(compare_two_node(cv_svm, cv_svm_one_proc))
            self.assertTrue(isequal(cv_svm_reduce, cv_svm_reduce_one_proc))


if __name__ == "__main__":
    args = sys.argv[1:]
    #print repr(args)
#    args = [10, 70, 'True', 1, 'False', None]
    args[2] = (args[2] == 'True')
    args[4] = (args[4] == 'True')
    if len(args) == 5:
        args.append(None)

    suite = unittest.TestSuite()
    suite.addTest(TestMemMapping(int(args[0]), int(args[1]),
                                 args[2], int(args[3]), args[4], args[5]))
#    suite.addTest(TestMemMapping(10, 80, True, 1, True, None))
#    suite.addTest(TestMemMapping(10, 80, True, 2, True, None))

    unittest.TextTestRunner().run(suite)
#
#
#    optlist, args = getopt.getopt(sys.argv[1:], ["n_samples=",
#                                                 "n_features=",
#                                                 "memmap=",
#                                                 "n_proc=",
#                                                 "is_swf=",
#                                                 "dir="])
#
#    for opt in optlist:
#        if opt[0] == '--n_samples':
#            n_samples = int(opt[1])
#        elif opt[0] == '--n_features':
#            n_features = int(opt[1])
#        elif opt[0] == '--memmap':
#            memmap = (opt[1] == 'True')
#        elif opt[0] == '--n_proc':
#            n_proc = int(opt[1])
#        elif opt[0] == '--is_swf':
#            is_swf = (opt[1] == 'True')
#        elif opt[0] == '--dir':
#            dir = opt[1]
