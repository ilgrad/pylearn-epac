# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:45:56 2013

@author: laure.hugo@cea.fr
"""

import imp
import sys
from distutils.version import LooseVersion as V

modules = ['numpy', 'sklearn', 'joblib']

for module in modules:
    try:
        imp.find_module(module)
    except ImportError:
        sys.stderr.write('ERROR: cannot import %s, '
                         'please check that you installed it properly \n' %
                         module)
        raise
    else:
        if module == 'joblib':
            import joblib
            if V(joblib.__version__) < V("0.7.1"):
                raise ValueError("joblib version is too old to use, "
                                 "please use version on "
                                 "https://github.com/joblib/joblib")

try:
    imp.find_module("soma_workflow")
except ImportError:
    sys.stderr.write("WARNING: soma_workflow not installed, "
                     "you won't be able to use all the features of epac \n")
