#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on 2 May 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
@author: jinpeng.li@cea.fr

"""

import sys
import os
import socket
import subprocess

from epac.errors import NoSomaWFError, NoEpacTreeRootError
from epac.configuration import conf
from epac.utils import which
from epac.workflow.splitters import CVBestSearchRefit

# _classes_cannot_be_splicted = [CVBestSearchRefit.__class__.__name__]
_classes_cannot_be_splicted = [CVBestSearchRefit.__class__.__name__]


def _is_cannot_be_splicted(signature):
    for _class_cannot_be_splicted in _classes_cannot_be_splicted:
        if _class_cannot_be_splicted in signature:
            return True
    return False


def _push_node_in_list(node, nodes_per_process_list):
    '''Push node in the list which contains the minimun number of nodes
    '''
    min_len = -1
    min_key = -1
    for key in nodes_per_process_list.keys():
        if (min_len == -1 or min_len > len(nodes_per_process_list[key])):
            min_key = key
            min_len = len(nodes_per_process_list[key])
    if min_key != -1:
        nodes_per_process_list[min_key].append(node.get_key())
    return nodes_per_process_list


def _export_nodes_recursively(
    node,
    num_processes,
    nodes_per_process_list,
    max_depth=conf.MAX_DEPTH_SPLIT_TREE,
    cur_depth=0
):
    '''Allocate recursively the nodes for list
    '''
    num_processes = len(nodes_per_process_list)
    children_nodes = node.children
    len_children = len(children_nodes)
    cur_depth = cur_depth + 1
    if len_children == 0 or cur_depth >= max_depth:
        nodes_per_process_list = _push_node_in_list(
            node,
            nodes_per_process_list)
        return nodes_per_process_list
    left = len_children % num_processes
    if len_children >= num_processes:
        for i in range(len_children - left):
            nodes_per_process_list = _push_node_in_list(
                node=children_nodes[i],
                nodes_per_process_list=nodes_per_process_list)
    if left > 0:
        for i in range(len_children - left, len_children):
            if (not _is_cannot_be_splicted(children_nodes[i].get_signature()))\
               and (not children_nodes[i].stop_top_down):
                nodes_per_process_list = _export_nodes_recursively(
                    children_nodes[i],
                    num_processes,
                    nodes_per_process_list,
                    max_depth=max_depth,
                    cur_depth=cur_depth)
            else:
                nodes_per_process_list = _push_node_in_list(
                    node=children_nodes[i],
                    nodes_per_process_list=nodes_per_process_list)
    return nodes_per_process_list


def export_nodes2num_processes(node, num_processes):
    '''export nodes
    Try to build "num_processes" queues which contains almost equally number
    of Epac nodes for computing.

    Parameters
    ----------
    node:epac.base.WFNode
        Epac tree root where you want to start to parallelly compute
        using "in_num_processes" cores.

    num_processes:integer
        The number of processes you have.
    '''
    nodes_per_process_list = dict()
    for i in range(num_processes):
        nodes_per_process_list[i] = list()
    nodes_per_process_list = _export_nodes_recursively(
        node=node,
        num_processes=num_processes,
        nodes_per_process_list=nodes_per_process_list)
    return nodes_per_process_list


def _gen_keysfile_list_from_nodes_list(
    working_directory,
    nodes_per_process_list
):
    '''Generating a list of files where each file contains a set of keys.
    Generating a list of files where each file contains a set of keys.
    A key means a node which can be considered as a job. All node's leaves
    should be computed in a job.
    '''
    keysfile_list = list()
    jobi = 0
    for npp_key in nodes_per_process_list.keys():
        keysfile = "." + os.path.sep + repr(jobi) + "." + conf.SUFFIX_JOB
        keysfile_list.append(keysfile)
        # print "in_working_directory="+in_working_directory
        # print "keysfile="+keysfile
        abs_keysfile = os.path.join(working_directory, keysfile)
        f = open(abs_keysfile, 'w')
        for key_signature in nodes_per_process_list[npp_key]:
            f.write("%s\n" % key_signature)
        f.close()
        jobi = jobi + 1
    return keysfile_list


def save_job_list(working_directory,
                   nodesinput_list):
    '''Write job list into working_directory as 0.job, 1.job, etc.

    Parameters
    ----------
    working_directory: string
        directory to write job list

    nodesinput_list: list of NodesInput
        This is for parallel computing for each element in the list.
        All of them are saved separately in working_directory.

    Example
    -------
    >>> from epac.map_reduce.exports import save_job_list
    >>> nodesinput_list = [{'Perms/Perm(nb=0)': 'Perms/Perm(nb=0)'},
    ...                    {'Perms/Perm(nb=1)': 'Perms/Perm(nb=1)'},
    ...                    {'Perms/Perm(nb=2)': 'Perms/Perm(nb=2)'}]
    >>> working_directory =  "/tmp"
    >>> save_job_list(working_directory, nodesinput_list)
    ['./0.job', './1.job', './2.job']
    '''
    keysfile_list = list()
    jobi = 0
    for nodesinput in nodesinput_list:
        keysfile = "." + os.path.sep + repr(jobi) + "." + conf.SUFFIX_JOB
        keysfile_list.append(keysfile)
        # print "in_working_directory="+in_working_directory
        # print "keysfile="+keysfile
        abs_keysfile = os.path.join(working_directory, keysfile)
        f = open(abs_keysfile, 'w')
        for key_signature in nodesinput:
            f.write("%s\n" % key_signature)
        f.close()
        jobi = jobi + 1
    return keysfile_list


if __name__ == "__main__":
    import doctest
    doctest.testmod()
