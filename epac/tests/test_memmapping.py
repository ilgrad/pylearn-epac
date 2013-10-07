# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:56:37 2013

@author: laure.hugo@cea.fr
"""

import numpy as np
from sklearn import datasets
import random
import datetime
import sys
import os
import tempfile


def create_mmat(nrows, ncols, default_values=None, dir=None):
    now = datetime.datetime.now()
    np.random.seed(now.second)
    hfile = tempfile.NamedTemporaryFile("w+", dir=dir)
    filename = hfile.name
    print "Temporary path :", filename
    hfile.close()
    mem_mat = np.memmap(filename,
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
    return mem_mat


def create_array(size, default_values=None, dir=None):
    ret_array = create_mmat(size, 1, default_values, dir=dir)
    ret_array = ret_array[:, 0]
    return ret_array


# @profile
def func_memm_local(n_samples, n_features, memmap=False,
                    n_proc=1, is_swf=False, dir=None):
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

    ## 1) Building dataset
    ## ============================================================
    print "\n \n"
    print " -> Pt1 : Beginning with", n_features, "features, memmap =",\
        memmap, ",", n_proc, "processes", "is_swf = ", is_swf
    if memmap:
        X = create_mmat(n_samples, n_features, dir=dir)
        y = create_array(n_samples, [0, 1], dir=dir)

        print "X matrix file size =", os.path.getsize(X.filename), "bytes"

        Xy = dict(X=X, y=y)
    else:

        X, y = datasets.make_classification(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=2,
                                            random_state=1)

        Xy = dict(X=X, y=y)
    ## 2) Building workflow
    ## =======================================================
    print " -> Pt2 : X and y created, building workflow"
    from sklearn.svm import SVC
    from epac import CV, Methods
    cv_svm_local = CV(Methods(*[SVC(kernel="linear"),
                                SVC(kernel="rbf")]),
                      n_folds=3)
    print " -> Pt3 : Workflow built, running"

    cv_svm = None
    if is_swf:
        # Running on the cluster
        from epac import SomaWorkflowEngine
        mmap_mode = None
        if memmap:
            mmap_mode = "r+"
        swf_engine = SomaWorkflowEngine(cv_svm_local,
                                        num_processes=n_proc,
                                        resource_id="jl237561@gabriel",
                                        login="jl237561",
                                        # remove_finished_wf=False,
                                        # remove_local_tree=False,
                                        mmap_mode=mmap_mode,
                                        queue="Global_long")
        cv_svm = swf_engine.run(**Xy)
    else:
        # Running on the local machine
        from epac import LocalEngine
        local_engine = LocalEngine(cv_svm_local, num_processes=n_proc)
        cv_svm = local_engine.run(**Xy)
    print " -> Pt4 : Finished running reducing"
    print cv_svm.reduce()

    print " -> Pt5 : Finished with", n_features, "features"

    ## 3) Removing files
    ## =======================================================
    if hasattr(X, "filename"):
        os.remove(X.filename)
    if hasattr(y, "filename"):
        os.remove(y.filename)


if __name__ == "__main__":
    args = sys.argv[1:]
    #print repr(args)
    # args = [500, 70000, 'True', 2, 'True']
    args[2] = (args[2] == 'True')
    args[4] = (args[4] == 'True')
    func_memm_local(int(args[0]), int(args[1]), args[2], int(args[3]), args[4])
