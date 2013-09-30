# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:02:00 2013

@author: laure
"""

import os
import time
import commands
import datetime


def get_pid(process_name):
    cmd = 'ps -ef | '\
          'grep "%s" | '\
          'grep -v unbuffer | '\
          'grep -v grep | '\
          "awk '{print $2}'" % process_name
    ret_id = commands.getoutput(cmd)
    if len(ret_id) == 0:
        ret_id = None
    else:
        ret_id = ret_id.split()
    return ret_id


def get_mem_cost(process_name):
    list_pid = get_pid(process_name)
    if not list_pid:
        return None
    total_mem_cost = 0
    for pid in list_pid:
        cmd = 'ps -o rss %s' % pid
        mem_cost = commands.getoutput(cmd)
        mem_cost = mem_cost.split("\n")[1]
        total_mem_cost += int(mem_cost)
    return total_mem_cost

#n_samples = 500
#test_list = [50000, 70000]
memmap = 'False'

filename = 'result_test.txt'

# Clearing file from other tries
open(filename, 'w').close()

n_samples = 500
test_list = [50000 * (2 ** n) for n in range(0, 7)]
for n_proc in range(1, 9):
    for n_features in test_list:
        process_name = "test_memmapping.py"
        os.system("(unbuffer python %s %i %i %s %i \
                   >> %s &)" % (process_name, n_samples, \
                   n_features, memmap, n_proc, filename))
        time.sleep(5)

        print "=========================="
        print "n_features = ", n_features
        print "memmap enabled = ", memmap
        print "number of processes =", n_proc
        start_time = datetime.datetime.now()
        print "Starting time = ", repr(start_time)
        max_mem_cost = 0
        while True:
            pid = get_pid(process_name)
            if not pid:
                break
            mem_cost = get_mem_cost(process_name)
            if max_mem_cost < mem_cost:
                max_mem_cost = mem_cost
            time.sleep(10)
            print "memory cost = ", mem_cost
        print "max memory cost = ", max_mem_cost
        finished_time = datetime.datetime.now()
        print "Finished time = ", repr(finished_time)
        print "Time cost=", repr((finished_time - start_time).seconds)

        with open(filename, "a") as myfile:
            myfile.write("\n \n")
