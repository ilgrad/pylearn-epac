# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:02:00 2013

@author: laure
"""

import os

n_samples = 50
test_list = [500, 1000]

filename = 'result_test.txt'

# Clearing file from other tries
open(filename, 'w').close()

#n_samples = 500
#test_list = [50000 * 2**n for n in range(0, 10)]
for n_features in test_list:
    os.system("unbuffer python test_memmapping.py %i %i \
               >> %s" %(n_samples, n_features, filename))
    with open(filename, "a") as myfile:
        myfile.write("\n \n")
