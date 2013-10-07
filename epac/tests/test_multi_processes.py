#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on 20 June 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr

"""

import unittest
import numpy as np
from epac.tests.wfexamples2test import get_wf_example_classes
from epac import LocalEngine
from epac import SomaWorkflowEngine

from sklearn import datasets
from epac.tests.utils import comp_2wf_reduce_res
from epac.tests.utils import compare_two_node


class EpacWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.n_cores = 3
        self.X = None
        self.y = None
        self.n_samples = 100
        self.n_features = int(1E03)
        self.X, self.y = datasets.make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=5)

    def tearDown(self):
        pass


    def test_examples_local_engine(self):
        list_all_examples = get_wf_example_classes()
        for example in list_all_examples:
#            if example().__class__.__name__ == "WFExample1" or\
#                example().__class__.__name__ == "WFExample2":
                # example = list_all_examples[0]
                wf = example().get_workflow()
                local_engine_wf = example().get_workflow()
                sfw_engine_wf = example().get_workflow()
                wf.run(X=self.X, y=self.y)
                local_engine = LocalEngine(tree_root=local_engine_wf,
                                           num_processes=self.n_cores)
                local_engine_wf = local_engine.run(X=self.X, y=self.y)
                sfw_engine = SomaWorkflowEngine(
                        tree_root=sfw_engine_wf,
                        num_processes=self.n_cores)
                sfw_engine_wf = sfw_engine.run(X=self.X, y=self.y)
                self.assertTrue(compare_two_node(wf, local_engine_wf))
                self.assertTrue(compare_two_node(wf, sfw_engine_wf))
                self.assertTrue(comp_2wf_reduce_res(wf, local_engine_wf))
                self.assertTrue(comp_2wf_reduce_res(wf, sfw_engine_wf))


if __name__ == '__main__':
    unittest.main()
