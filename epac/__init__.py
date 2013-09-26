# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:58:21 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from epac.workflow.pipeline import Pipe
from epac.workflow.splitters import Perms, Methods, CV
from epac.workflow.splitters import CVBestSearchRefit
from epac.workflow.splitters import CVBestSearchRefitParallel
from epac.workflow.splitters import ColumnSplitter, RowSplitter, CRSplitter
from epac.workflow.base import BaseNode, key_pop, key_split
from epac.configuration import conf, debug
from epac.map_reduce.results import ResultSet, Result
from epac.utils import train_test_merge, train_test_split, dict_diff, range_log2, export_csv
from epac.stores import StoreFs, StoreMem
from epac.map_reduce.mappers import MapperSubtrees
from epac.map_reduce.engine import SomaWorkflowEngine, LocalEngine
from epac.map_reduce.reducers import ClassificationReport, PvalPerms

__version__ = '0.10-git'

import sklearn_plugins

import __check_build

__all__ = ['BaseNode',
           'Pipe',
           'Perms',
           'Methods',
           'CV',
           'CVBestSearchRefit',
           'CVBestSearchRefitParallel',
           'ColumnSplitter', 'RowSplitter', 'CRSplitter',
           'ClassificationReport', 'PvalPerms',
           'Result',
           'ResultSet',
           'sklearn_plugins',
           'conf',
           'debug',
           'train_test_split',
           'train_test_merge',
           'key_pop',
           'key_split',
           'dict_diff',
           'export_csv',
           'StoreFs',
           'StoreMem',
           'range_log2',
           'MapperSubtrees',
           'SomaWorkflowEngine',
           'LocalEngine'
           ]