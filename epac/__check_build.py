# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:45:56 2013

@author: laure.hugo@cea.fr
"""

import imp
import sys
from distutils.version import LooseVersion as V

modules = ['dill', 'numpy', 'soma_workflow', 'sklearn']

for module in modules:
    try:
        imp.find_module(module)
    except ImportError:
        sys.stderr.write('Error, cannot import %s, '
                         'please check that you installed it properly' %
                         module)
    else:
        if module == 'dill':
            import dill
            if V(dill.__version__) < V("0.2a"):
                raise ValueError("dill version is too old to use, "
                                 "please use version on "
                                 "https://github.com/uqfoundation/dill")
        if module == 'joblib':
            import joblib
            if V(joblib.__version__) < V("0.7.1"):
                raise ValueError("joblib version is too old to use, "
                                 "please use version on "
                                 "https://github.com/joblib/joblib")
