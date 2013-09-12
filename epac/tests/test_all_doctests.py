# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:06:27 2013

@author: laure.hugo@cea.fr
"""

import unittest
import doctest
import epac


suite = unittest.TestSuite()

suite.addTest(doctest.DocTestSuite(epac.configuration))
suite.addTest(doctest.DocTestSuite(epac.errors))
suite.addTest(doctest.DocTestSuite(epac.stores))
suite.addTest(doctest.DocTestSuite(epac.utils))

# map_reduce package
suite.addTest(doctest.DocTestSuite(epac.map_reduce.engine))
suite.addTest(doctest.DocTestSuite(epac.map_reduce.exports))
suite.addTest(doctest.DocTestSuite(epac.map_reduce.inputs))
suite.addTest(doctest.DocTestSuite(epac.map_reduce.mappers))
suite.addTest(doctest.DocTestSuite(epac.map_reduce.reducers))
suite.addTest(doctest.DocTestSuite(epac.map_reduce.results))
suite.addTest(doctest.DocTestSuite(epac.map_reduce.split_input))

# sklearn_plugins package
suite.addTest(doctest.DocTestSuite(epac.sklearn_plugins.estimators))
suite.addTest(doctest.DocTestSuite(epac.sklearn_plugins.resampling))

# workflow package
suite.addTest(doctest.DocTestSuite(epac.workflow.base))
suite.addTest(doctest.DocTestSuite(epac.workflow.factory))
suite.addTest(doctest.DocTestSuite(epac.workflow.pipeline))
suite.addTest(doctest.DocTestSuite(epac.workflow.splitters))
suite.addTest(doctest.DocTestSuite(epac.workflow.wrappers))

unittest.TextTestRunner().run(suite)