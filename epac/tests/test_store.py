# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:55:05 2013

@author: jinpeng.li@cea.fr
"""


import numpy as np
from epac.stores import epac_joblib

npdata1 = np.random.random(size=(5, 5))
npdata2 = np.random.random(size=(5, 5))

dict_data = {"1": npdata1, "2": npdata2}
epac_joblib.dump(dict_data, "/tmp/123")

dict_data2 = epac_joblib.load("/tmp/123")
