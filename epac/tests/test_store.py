# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:55:05 2013

@author: jinpeng.li@cea.fr
"""


import numpy as np
from epac.configuration import conf
from epac.stores import epac_joblib

conf.MEMM_THRESHOLD = 100
npdata1 = np.random.random(size=(5, 5))
npdata2 = np.random.random(size=(100, 5))

dict_data = {"1": npdata1, "2": npdata2}
epac_joblib.dump(dict_data, "/tmp/123")
dict_data2 = epac_joblib.load("/tmp/123")

np.all(dict_data2["1"] == npdata1)
np.all(dict_data2["2"] == npdata2)


#from epac.stores import extract_values
#from epac.stores import replace_values
#from epac.stores import TagObject
#
#
#def func_is_need_extract(obj):
#    if type(obj) is str:
#        if(len(obj) >= 3):
#            return True
#    return False
#
#
#def func_can_deeper(obj):
#    return type(obj) is not str
#
#
#class TestC:
#    def __init__(self):
#        self.A = "C1A"
#        self.B = "C2"
#
#
#class TestD:
#    def __init__(self):
#        self.A = "D1"
#        self.B = "D2"
#        self.C = TestC()
#
#
#obj = TestD()
#extracted_values, obj = extract_values(obj,
#                                       func_is_need_extract,
#                                       func_can_deeper)
#print obj.A
#print obj.B
#print isinstance(obj.C.A, TagObject) 
#print obj.C.B
#obj = replace_values(obj, extracted_values)
#print obj.A
#print obj.B
#print obj.C.A
#print obj.C.B

#
#
#import numpy as np
#
#testarray = np.asarray([[1.0, 1.0, 1],
#                       [1.0, 1.0, 1]])
#
#testarray = np.asarray([1.0, 1.0, 1])
#
#testarray = np.random.random((10, 10))
#
#
#def func_is_big_nparray(obj):
#    if isinstance(obj, np.ndarray):
#        num = 1
#        for ishape in obj.shape:
#            num = num * ishape
#        print "num=", num
#        if num > 100:
#            return True
#    return False
#
#print func_is_big_nparray(testarray)
