# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:55:05 2013

@author: jinpeng.li@cea.fr
"""

import unittest
import numpy as np
from epac.configuration import conf
from epac.stores import epac_joblib
from epac.stores import TagObject


class TestStore(unittest.TestCase):
    def test_store(self):
        conf.MEMM_THRESHOLD = 100
        npdata1 = np.random.random(size=(2, 2))
        npdata2 = np.random.random(size=(100, 5))

        dict_data = {"1": npdata1, "2": npdata2}
        epac_joblib.dump(dict_data, "/tmp/123")
        isinstance(dict_data["2"], TagObject)
        dict_data2 = epac_joblib.load("/tmp/123")
        self.assertTrue(np.all(dict_data2["1"] == npdata1))
        self.assertTrue(np.all(dict_data2["2"] == npdata2))

if __name__ == '__main__':
    unittest.main()
