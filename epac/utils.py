# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:13:26 2013

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import os
import csv
import dill as pickle
from epac.configuration import conf
from epac.workflow.base import key_push, key_pop


def get_next_number(str_value):
    start_pos = -1
    end_pos = -1
    rm_space = False
    left_part = ""
    for i in range(len(str_value)):
        char = str_value[i]
        if char != " " and start_pos == -1:
            start_pos = i
        elif char == " " and start_pos != -1 and end_pos == -1:
            end_pos = i
            rm_space = True
            break
    if end_pos == -1:
        end_pos = len(str_value)
    if rm_space:
        left_part = str_value[end_pos + 1: len(str_value)]
    else:
        left_part = str_value[end_pos: len(str_value)]
    num_str = str_value[start_pos: end_pos]
    return (int(num_str), left_part)


def range_log2(n, add_n=True):
    """Return log2 range starting from 1"""
    rang = (2**np.arange(int(np.floor(np.log2(n))) + 1)).tolist()
    if add_n:
        rang.append(int(n))
    return rang


## =========== ##
## == Utils == ##
## =========== ##

def _list_diff(l1, l2):
    return [item for item in l1 if not item in l2]


def _list_contains(l1, l2):
    return all([item in l1 for item in l2])


def _list_union_inter_diff(*lists):
    """Return 3 lists: intersection, union and differences of lists
    """
    union = set(lists[0])
    inter = set(lists[0])
    for l in lists[1:]:
        s = set(l)
        union = union | s
        inter = inter & s
    diff = union - inter
    return list(union), list(inter), list(diff)


def _list_indices(l, val):
    return [i for i in xrange(len(l)) if l[i] == val]


def dict_diff(*dicts):
    """Find the differences in a dictionaries

    Returns
    -------
    diff_keys: a list of keys that differ amongs dicts
    diff_vals: a dict with keys values differences between dictonaries.
        If some dict differ bay keys (some keys are missing), return
        the key associated with None value

    Examples
    --------
    >>> dict_diff(dict(a=1, b=2, c=3), dict(b=0, c=3))
    {'a': None, 'b': [0, 2]}
    >>> dict_diff(dict(a=1, b=[1, 2]), dict(a=1, b=[1, 3]))
    {'b': [[1, 2], [1, 3]]}
    >>> dict_diff(dict(a=1, b=np.array([1, 2])), dict(a=1, b=np.array([1, 3])))
    {'b': [array([1, 2]), array([1, 3])]}
    """
    # Find diff in keys
    union_keys, inter_keys, diff_keys = _list_union_inter_diff(*[d.keys()
                                            for d in dicts])
    diff_vals = dict()
    for k in diff_keys:
        diff_vals[k] = None
    # Find diff in shared keys
    for k in inter_keys:
        if isinstance(dicts[0][k], (np.ndarray, list, tuple)):
            if not np.all([np.all(d[k] == dicts[0][k]) for d in dicts]):
                diff_vals[k] = [d[k] for d in dicts]
        elif isinstance(dicts[0][k], dict):
            if not np.all([d[k] == dicts[0][k] for d in dicts]):
                diff_vals[k] = [d[k] for d in dicts]
        else:
            s = set([d[k] for d in dicts])
            if len(s) > 1:
                diff_vals[k] = list(s)
    return diff_vals


def _sub_dict(d, subkeys):
    return {k: d[k] for k in subkeys}


def _as_dict(v, keys):
    """
    Ensure that v is a dict, if not create one using keys.

    Example
    -------
    >>> _as_dict(([1, 2], [3, 1]), ["x", "y"])
    {'y': [3, 1], 'x': [1, 2]}
    """
    if isinstance(v, dict):
        return v
    if len(keys) == 1:
        return {keys[0]: v}
    if len(keys) != len(v):
        raise ValueError("Do not know how to build a dictionnary with keys %s"
            % keys)
    return {keys[i]: v[i] for i in xrange(len(keys))}


def _dict_prefix_keys(d, prefix):
    return {prefix + str(k): d[k] for k in d}


def _dict_suffix_keys(d, suffix):
    return {str(k) + suffix: d[k] for k in d}


def _func_get_args_names(f):
    """Return non defaults function args names
    """
    import inspect
    a = inspect.getargspec(f)
    if a.defaults:
        args_names = a.args[:(len(a.args) - len(a.defaults))]
    else:
        args_names = a.args[:len(a.args)]
    if "self" in args_names:
        args_names.remove("self")
    return args_names


def which(program):
    """Same with "which" command in linux
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def export_csv(result_set, filename):
    '''Export the results to a CSV file

    Parameters
    ----------
    result_set: ResultSet to export

    filename: Relative or absolute path to the CSV file
        If the file doesn't exist, create it

    Example
    -------
    >>> from epac import Result, ResultSet
    >>> from epac.utils import export_csv
    >>> r1 = Result('SVC(C=1)', **dict(a=1, b=2))
    >>> r2 = Result('SVC(C=2)', **dict(a=1, b=2))
    >>> set = ResultSet(r1, r2)
    >>> export_csv(set, 'results.csv')
    >>> with open('results.csv', 'rb') as csvfile:  # doctest: +NORMALIZE_WHITESPACE
    ...     print csvfile.read()
    a,b,key
    1,2,SVC(C=1)
    1,2,SVC(C=2)
    <BLANKLINE>
    '''
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_keys = result_set.values()[0].keys()
        result_keys.sort()
        spamwriter.writerow(result_keys)
        for result in result_set.values():
            temp_list = []
            for key in result_keys:
                temp_list.append(result[key])
            spamwriter.writerow(temp_list)


## ============================================== ##
## == down-stream data-flow manipulation utils == ##
## ============================================== ##

def train_test_split(Xy):
    """Split Xy into two dictonaries. If input dictonnary whas not build
    with train_test_merge(Xy1, Xy2) then return twice the input
    dictonnary.

    Parameters
    ----------
    Xy: dict

    Returns
    -------
    dict1, dict2 : splited dictionaries

    Example
    -------
    >>> train_test_merged = train_test_merge(dict(a=1, b=2), dict(a=33, b=44, c=55))
    >>> print train_test_merged
    {'c/test': 55, 'a/test': 33, 'b/test': 44, 'a/train': 1, 'b/train': 2}
    >>> print train_test_split(train_test_merged)
    ({'a': 1, 'b': 2}, {'a': 33, 'c': 55, 'b': 44})
    >>> print train_test_split(dict(a=1, b=2))
    ({'a': 1, 'b': 2}, {'a': 1, 'b': 2})
    """
    keys_train = [k for k in Xy if (key_pop(k)[1] == conf.TRAIN)]
    keys_test = [k for k in Xy if (key_pop(k)[1] == conf.TEST)]
    if not keys_train and not keys_test:
        return Xy, Xy
    if keys_train and keys_test:
        Xy_train = {key_pop(k)[0]: Xy[k] for k in keys_train}
        Xy_test = {key_pop(k)[0]: Xy[k] for k in keys_test}
        return Xy_train, Xy_test
    raise KeyError("data-flow could not be splitted")


def train_test_merge(Xy_train, Xy_test):
    """Merge two dict avoiding keys collision.

    Parameters
    ----------
    Xy_train: dict
    Xy_test: dict

    Returns
    -------
    dict : merged dictionary

    Example
    -------
    >>> train_test_merge(dict(a=1, b=2), dict(a=33, b=44, c=55)) == {'a/test': 33, 'a/train': 1, 'b/test': 44, 'b/train': 2, 'c/test': 55}
    True
    """
    Xy_train = {key_push(k, conf.TRAIN): Xy_train[k] for k in Xy_train}
    Xy_test = {key_push(k, conf.TEST): Xy_test[k] for k in Xy_test}
    Xy_train.update(Xy_test)
    return Xy_train


def save_dictionary(dataset_dir, **Xy):
    '''Save a dictionary to a directory
    Save a dictionary to a directory. This dictionary may contain
    numpy array, numpy.memmap

    Example
    -------
    from sklearn import datasets
    from epac.utils import save_dictionary
    X, y = datasets.make_classification(n_samples=500,
                                        n_features=200000,
                                        n_informative=2,
                                        random_state=1)
    Xy = dict(X=X, y=y)
    save_dictionary("/tmp/save_datasets_data", **Xy)
    '''
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    index_filepath = os.path.join(dataset_dir, conf.DICT_INDEX_FILE)
    file_dict_index = open(index_filepath, "w+")
    file_dict_index.write(repr(len(Xy)) + "\n")
    for key in Xy:
        filepath = os.path.join(dataset_dir, key + ".npy")
        file_dict_index.write(key)
        file_dict_index.write("\n")
        file_dict_index.write(filepath)
        file_dict_index.write("\n")
    file_dict_index.close()
    for key in Xy:
        filepath = os.path.join(dataset_dir, key + ".npy")
        np.save(filepath, Xy[key])


def is_need_mem(filepath):
    filesize = os.path.getsize(filepath)
    if filesize > conf.MEMM_THRESHOLD:
        return True
    return False


def load_dictionary(dataset_dir):
    '''Load a dictionary
    Load a dictionary from save_dictionary

    Example
    -------
    from epac.utils import load_dictionary
    Xy = load_dictionary("/tmp/save_datasets_data")
    '''
    if not os.path.exists(dataset_dir):
        return None
    index_filepath = os.path.join(dataset_dir, conf.DICT_INDEX_FILE)
    if not os.path.isfile(index_filepath):
        return None
    file_dict_index = open(index_filepath, "r")
    len_dict = file_dict_index.readline()
    res = {}
    for i in range(int(len_dict)):
        key = file_dict_index.readline()
        key = key.strip("\n")
        filepath = file_dict_index.readline()
        filepath = filepath.strip("\n")
        if is_need_mem(filepath):
            data = np.load(filepath, "r+")
        else:
            data = np.load(filepath)
        res[key] = data
    return res


def clean_tree_stores(tree_root):
    for each_node in tree_root.walk_true_nodes():
        if each_node.store:
            each_node.store = None


if __name__ == "__main__":
    import doctest
    doctest.testmod()