# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:02:00 2013

@author: laure
"""

import os
import time
import psutil
import subprocess
import commands


def get_pid(process_name):
    cmd = 'ps -ef | '\
          'grep "%s" | '\
          'grep -v unbuffer | '\
          'grep -v grep | '\
          "awk '{print $2}'" % process_name
    ret_id = commands.getoutput(cmd)
    if len(ret_id) == 0:
        ret_id = None
    return ret_id


def get_mem_cost(process_name):
    pid = get_pid(process_name)
    if not pid:
        return None
    cmd = 'ps -o rss %s' % pid
    mem_cost = commands.getoutput(cmd)
    mem_cost = mem_cost.split("\n")[1]
    return int(mem_cost)

n_samples = 50
test_list = [500, 1000]

filename = 'result_test.txt'

# Clearing file from other tries
open(filename, 'w').close()

#n_samples = 500
#test_list = [50000 * 2**n for n in range(0, 10)]
for n_features in test_list:
    process_name = "test_memmapping.py"
    os.system("(unbuffer python %s %i %i \
               >> %s &)" %(process_name, n_samples, n_features, filename))
    time.sleep(2)
    
    print "=========================="
    print "n_features = ", n_features
    while True:
        pid = get_pid(process_name)
        if not pid:
            break
        mem_cost = get_mem_cost(process_name)
        time.sleep(30)
        print "memory cost = ", mem_cost

    with open(filename, "a") as myfile:
        myfile.write("\n \n")
