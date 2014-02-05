# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:43:51 2013

@author: laure.hugo@cea.fr
@author: jinpeng.li@cea.fr

"""

import unittest

from sklearn import datasets
from epac import SomaWorkflowEngine
from sklearn.svm import SVC
from epac import CV, Methods


class TestGetSWFEngineInfo(unittest.TestCase):
    def test_engine_info(self):
        n_samples = 20
        n_features = 100
        n_proc = 2
        X, y = datasets.make_classification(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=2,
                                            random_state=1)
        Xy = dict(X=X, y=y)
        cv_svm_local = CV(Methods(*[SVC(kernel="linear"),
                                    SVC(kernel="rbf")]),
                          n_folds=3)
        swf_engine = SomaWorkflowEngine(cv_svm_local,
                                        num_processes=n_proc,
                                        resource_id="jl237561@gabriel",
                                        login="jl237561",
                                        remove_finished_wf=False,
                                        remove_local_tree=False,
                                        queue="Global_long")
        swf_engine.run(**Xy)
        print "engine_info ================"
        for job_info in swf_engine.engine_info:
            print "  job_info================="
            print "  mem_cost= ", job_info.mem_cost
            print "  vmem_cost= ", job_info.vmem_cost
            print "  time_cost= ", job_info.time_cost
            self.assertTrue(job_info.time_cost > 0)

if __name__ == "__main__":
    unittest.main()
