# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:55:05 2013

@author: jinpeng.li@cea.fr
"""


#import numpy as np
#from epac.stores import epac_joblib
#
#npdata1 = np.random.random(size=(5, 5))
#npdata2 = np.random.random(size=(5, 5))
#
#dict_data = {"1": npdata1, "2": npdata2}
#epac_joblib.dump(dict_data, "/tmp/123")
#
#dict_data2 = epac_joblib.load("/tmp/123")

from epac.stores import extract_values
from epac.stores import replace_values


class TestC:
    def __init__(self):
        self.A = "C1"
        self.B = "C2"


class TestD:
    def __init__(self):
        self.A = "D1"
        self.B = "D2"
        self.C = TestC()

obj = TestD()
extracted_values, obj = extract_values(obj, str)
obj = replace_values(obj, extracted_values)
print obj.A
print obj.B
print obj.C.A
print obj.C.B
