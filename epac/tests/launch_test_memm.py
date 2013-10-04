# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:02:00 2013

@author: laure
"""

import os
import time
import commands
import datetime


###########################
## Memory cost functions ##
###########################

def get_pid(process_name):
    ''' Return the pid of the processes whose name contains process_name.
    '''
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
    ''' Return the sum of the memory costs of all processes whose name
        contains process_name.
    '''
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


def print_process_mem_cost(process_name, delay=10):
    '''  Print the memory cost of all processes whose name contains
        process_name.

    Print every 'delay' secondes the memory cost of all the processes
    whose name contains process_name, while at least one exists.
    Print the max memory cost at the end of the processes.
    '''
    max_mem_cost = 0
    while True:
        pid = get_pid(process_name)
        if not pid:
            break
        mem_cost = get_mem_cost(process_name)
        if max_mem_cost < mem_cost:
            max_mem_cost = mem_cost
        time.sleep(delay)
        print "memory cost = ", mem_cost
    print "max memory cost = ", max_mem_cost


#################################
## Definition of the variables ##
#################################

n_samples = 500
n_features_list = [50000 * (2 ** n) for n in range(0, 6)]
memmap = True
n_proc_list = range(1, 9)
is_on_cluster = False
directory = '/volatile'

# Path of the file to write results
filename = 'result_test.txt'

# Clearing file from other tries
open(filename, 'w').close()


####################
## Memory testing ##
####################

for n_proc in n_proc_list:
    print "==========================="
    print "====== %i process(es) ======" % n_proc
    for n_features in n_features_list:
        print "---------------------------"
        print "n_features = ", n_features
        print "memmap enabled = ", memmap
        print "number of processes =", n_proc
        process_name = "test_memmapping.py"
        start_time = datetime.datetime.now()
        print "Starting time = ", repr(start_time)

        cmd = ""
        if not is_on_cluster:
            cmd = "(unbuffer python %s %i %i %s %i %s %s >> %s &)" % \
                                                (process_name,
                                                 n_samples,
                                                 n_features,
                                                 repr(memmap),
                                                 n_proc,
                                                 repr(is_on_cluster),
                                                 directory,
                                                 filename)
            os.system(cmd)
            time.sleep(5)
            print_process_mem_cost(process_name)
        else:
            cmd = "unbuffer python %s %i %i %s %i %s >> %s" % \
                                                (process_name,
                                                 n_samples,
                                                 n_features,
                                                 memmap,
                                                 n_proc,
                                                 repr(is_on_cluster),
                                                 filename)
            os.system(cmd)
        finished_time = datetime.datetime.now()
        print "Finished time = ", repr(finished_time)
        print "Time cost=", repr((finished_time - start_time).seconds)

        # Remove the previously created temp files
        os.system('rm -rf /tmp/tmp*')
        if directory:
            os.system('rm -rf %s/tmp*' % directory)
