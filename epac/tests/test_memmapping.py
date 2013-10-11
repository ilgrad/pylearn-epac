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
import getopt
from epac import StoreFs

from epac.tests.utils import isequal, compare_two_node


def create_mmat(nrows, ncols, default_values=None, dir=None,
                writing_mode=True):
    ''' Create a random matrix with memory mapping (saved on the disk)

    Create a matrix of the desired number of rows and columns, and fill it with
    random numbers taken from the defaults_values list.

    Parameters
    ----------
    nrows: int
        number of rows the matrix

    ncols: int
        number of columns of the matrix

    default_values: list of integers
        Choose the random integers from this list to fill the matrix

    dir: directory path
        Path of the directory where the matrix will be saved
        If None, save in /tmp

    writing_mode: boolean
        If True, generate the matrix
        Otherwise, test if there is an existing matrix. If there is, load the
        previously generated matrix, if not, generate it
    '''
    # Define the name of the matrix, depending on its size
    filename = 'tmp_rows_' + str(nrows) + '_cols_' + str(ncols)
    if dir is None:
        filepath = '/tmp/' + filename
    else:
        if not os.path.isdir(dir):
            os.mkdir(dir)
        filepath = os.path.join(dir, filename)

    # Test if the file already exists
    existing_file = os.path.isfile(filepath)

    if writing_mode or not existing_file:
        # If the user wants, or if the file doesn't exist already,
        # generate the matrix and fill it row by row
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
        # Load the matrix previously generated
        mem_mat = np.memmap(filepath,
                            dtype='float32',
                            mode='r+',
                            shape=(nrows, ncols))
    return mem_mat


def create_array(size, default_values=None, dir=None, writing_mode=True):
    ''' Create a random array with memory mapping (saved on the disk)

    Create a array of the desired size, and fill it with
    random numbers taken from the defaults_values list.

    Parameters
    ----------
    size: int
        size of the array

    default_values: list of integers
        Choose the random integers from this list to fill the matrix

    dir: directory path
        Path of the directory where the matrix will be saved
        If None, save in /tmp

    writing_mode: boolean
        If True, generate the matrix
        Otherwise, load a previously generated matrix
    '''
    ret_array = create_mmat(size, 1, default_values=default_values, dir=dir,
                            writing_mode=writing_mode)
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

    def __init__(self, n_samples, n_features, memmap,
                 n_proc, is_swf, dir, testname='test_memmapping'):
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
            # If the proc is 1, always generate the matrix
            # Otherwise, load it if it exists, or create it if it doesn't
            writing_mode = (self.n_proc == 1)
            X = create_mmat(self.n_samples, self.n_features, dir=self.dir,
                            writing_mode=writing_mode)
            y = create_array(self.n_samples, [0, 1], dir=self.dir,
                             writing_mode=writing_mode)
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
                                            # remove_finished_wf=False,
                                            # remove_local_tree=False,
                                            mmap_mode=mmap_mode,
                                            queue="Global_long")

            cv_svm = swf_engine.run(**Xy)

            # Printing information about the jobs
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

        # Creating the directory to save results, if it doesn't exist
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
            ## 4.2) Loading the results for one process
            ## ===================================================
            try:
                store = StoreFs(dirpath=dirpath, clear=False)
                cv_svm_one_proc = store.load()

                with open("/tmp/tmp_save_results", 'r+') as filename:
                    cv_svm_reduce_one_proc = pickle.load(filename)

                ## 5.2) Comparing results to the results for one process
                ## ===================================================
                print "\nComparing %i proc with one proc" % self.n_proc
                self.assertTrue(compare_two_node(cv_svm, cv_svm_one_proc))
                self.assertTrue(isequal(cv_svm_reduce, cv_svm_reduce_one_proc))
            except KeyError:
                print "Warning: "
                print "No previous tree detected, no possible "\
                    "comparison of results"


if __name__ == "__main__":
    # Default values on the test
    n_samples = 50
    n_features = 500
    memmap = True
    n_proc = 3
    is_swf = False
    directory = None

    # Getting the arguments from the shell
    optlist, args = getopt.gnu_getopt(sys.argv[1:], "", ["n_samples=",
                                                         "n_features=",
                                                         "memmap=",
                                                         "n_proc=",
                                                         "is_swf=",
                                                         "dir="])
    # Changing the default values depending on the given arguments
    for opt in optlist:
        if opt[0] == '--n_samples':
            n_samples = int(opt[1])
        elif opt[0] == '--n_features':
            n_features = int(opt[1])
        elif opt[0] == '--memmap':
            memmap = (opt[1] == 'True')
        elif opt[0] == '--n_proc':
            n_proc = int(opt[1])
        elif opt[0] == '--is_swf':
            is_swf = (opt[1] == 'True')
        elif opt[0] == '--dir':
            directory = opt[1]

    # Running the test with the given arguments
    suite = unittest.TestSuite()
    suite.addTest(TestMemMapping(n_samples=n_samples, n_features=n_features,
                                 memmap=memmap, n_proc=n_proc, is_swf=is_swf,
                                 dir=directory))
    unittest.TextTestRunner().run(suite)
