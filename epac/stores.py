# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:54:35 2013

Stores for EPAC

@author: edouard.duchesnay@cea.fr
"""

import os
import shutil
# import pickle
import dill as pickle
import joblib
import json
import inspect
import numpy as np
from abc import abstractmethod
from epac.configuration import conf
from epac.map_reduce.results import ResultSet


class TagObject:
    def __init__(self):
        self.hash_id = os.urandom(32)


def func_is_big_nparray(obj):
    if type(obj) in (np.ndarray, np.matrix, np.memmap):
    # if isinstance(obj, np.ndarray):
        num = 8
        for ishape in obj.shape:
            num = num * ishape
        if num > conf.MEMM_THRESHOLD:
            return True
    return False


def replace_values(obj, extracted_values, max_depth=10):
    """
    See example in extract_values
    """
    is_modified = False
    max_depth = max_depth - 1
    if max_depth < 0:
        return (obj, is_modified)
    # object is TagObject
    if isinstance(obj, TagObject):
        tag_value = obj
        set_value = extracted_values[tag_value.hash_id]
        obj = set_value
        is_modified = True
        return (obj, is_modified)

    # StoreMem
    if isinstance(obj, StoreMem):
        obj2set, tmp_is_modified = replace_values(obj.dict,
                                                      extracted_values,
                                                      max_depth)
        if tmp_is_modified:
            is_modified = True
            try:
                obj.dict = obj2set
            except:
                pass

    # ResultSet
    if isinstance(obj, ResultSet):
        obj2set, tmp_is_modified = replace_values(obj.results,
                                                      extracted_values,
                                                      max_depth)
        if tmp_is_modified:
            is_modified = True
            try:
                obj.results = obj2set
            except:
                pass

    # dictionary case
    if isinstance(obj, dict):
        for key in obj:
            obj2set, tmp_is_modified = replace_values(obj[key],
                                                      extracted_values,
                                                      max_depth)
            if tmp_is_modified:
                is_modified = True
                try:
                    obj[key] = obj2set
                except:
                    pass

    # list case
    if isinstance(obj, list):
        for iobj in xrange(len(obj)):
            obj2set, tmp_is_modified = replace_values(obj[iobj],
                                                      extracted_values,
                                                      max_depth)
            if tmp_is_modified:
                is_modified = True
                try:
                    obj[iobj] = obj2set
                except:
                    pass
    return (obj, is_modified)


def extract_values(obj,
                   func_is_need_extract,
                   max_depth=10):
    """
    Example
    -------
    >>> import numpy as np
    >>> from epac.configuration import conf
    >>> from epac import Result, ResultSet
    >>> from epac.stores import StoreMem
    >>> from epac.stores import extract_values
    >>> from epac.stores import replace_values
    >>> from epac.stores import func_is_big_nparray
    >>> from epac.stores import TagObject
    >>>
    >>> conf.MEMM_THRESHOLD = 100
    >>> npdata1 = np.random.random(size=(2, 2))
    >>> npdata2 = np.random.random(size=(100, 5))
    >>>
    >>> store = StoreMem()
    >>>
    >>> r1 = Result('SVC(C=1)', a=npdata1, b=npdata2)
    >>> r2 = Result('SVC(C=2)', a=npdata2, b=npdata1)
    >>> set1 = ResultSet(r1, r2)
    >>> store.save('SVC', set1)
    >>> replaced_array, ext_obj, is_modified = extract_values(store, func_is_big_nparray)
    >>> isinstance(ext_obj.dict['SVC']['SVC(C=2)']['a'], TagObject)
    True
    >>> new_obj, _ = replace_values(ext_obj, replaced_array)
    >>> isinstance(new_obj.dict['SVC']['SVC(C=2)']['a'], TagObject)
    False
    >>> isinstance(new_obj.dict['SVC']['SVC(C=2)']['a'], np.ndarray)
    True

    """
    replaced_array = {}
    is_modified = False
    # When obj is replace-able
    if func_is_need_extract(obj):
        replaced_object = TagObject()
        tag_value = obj
        obj = replaced_object
        replaced_array[replaced_object.hash_id] = tag_value
        is_modified = True
        return (replaced_array, obj, is_modified)

    max_depth = max_depth - 1
    # print "max_depth=", max_depth
    if max_depth < 0:
        return (replaced_array, obj, is_modified)

    # StoreMem
    if isinstance(obj, StoreMem):
        pros_replaced_array, obj2set, tmp_is_modified \
                            = extract_values(obj.dict,
                                             func_is_need_extract,
                                             max_depth)
        if tmp_is_modified:
            is_modified = True
            StoreMem.dict = obj2set
            replaced_array.update(pros_replaced_array)

    # ResultSet
    if isinstance(obj, ResultSet):
        pros_replaced_array, obj2set, tmp_is_modified \
                            = extract_values(obj.results,
                                             func_is_need_extract,
                                             max_depth)
        if tmp_is_modified:
            is_modified = True
            obj.results = obj2set
            replaced_array.update(pros_replaced_array)
    # Dictionary case
    if isinstance(obj, dict):
        for key in obj:
            pros_replaced_array, obj2set, tmp_is_modified \
                                = extract_values(obj[key],
                                                 func_is_need_extract,
                                                 max_depth)
            if tmp_is_modified:
                is_modified = True
                obj[key] = obj2set
                replaced_array.update(pros_replaced_array)
    # List case
    if isinstance(obj, list):
        for iobj in xrange(len(obj)):
            pros_replaced_array, obj2set, tmp_is_modified \
                                    = extract_values(obj[iobj],
                                                     func_is_need_extract,
                                                     max_depth)
            if tmp_is_modified:
                is_modified = True
                obj[iobj] = obj2set
                replaced_array.update(pros_replaced_array)

    return (replaced_array, obj, is_modified)


class epac_joblib:
    """
    It is optimized for dictionary dump and load
    Since joblib produces too many small files for mamory mapping,
    we try to limit the produced files for dictionary.

    Example
    -------
    >>> import numpy as np
    >>> from epac.configuration import conf
    >>> from epac.stores import epac_joblib
    >>>
    >>> conf.MEMM_THRESHOLD = 100
    >>> npdata1 = np.random.random(size=(2, 2))
    >>> npdata2 = np.random.random(size=(100, 5))
    >>>
    >>> dict_data = {"1": npdata1, "2": npdata2}
    >>> epac_joblib.dump(dict_data, "/tmp/123")
    >>> dict_data2 = epac_joblib.load("/tmp/123")
    >>>
    >>> np.all(dict_data2["1"] == npdata1)
    True
    >>> np.all(dict_data2["2"] == npdata2)
    memmap(True, dtype=bool)
    >>> from epac.stores import StoreMem
    >>> from epac import Result, ResultSet
    >>>
    >>> store = StoreMem()
    >>>
    >>> r1 = Result('SVC(C=1)', a=npdata1, b=npdata2)
    >>> r2 = Result('SVC(C=2)', a=npdata2, b=npdata1)
    >>> set1 = ResultSet(r1, r2)
    >>> store.save('SVC', set1)
    >>> epac_joblib.dump(store, "/tmp/store")
    >>> store = epac_joblib.load("/tmp/store")
    >>>
    >>> r3 = Result('SVC(C=3)', **dict(a=1, b=2))
    >>> r4 = Result('SVC(C=4)', **dict(a=1, b=2))
    """
    @staticmethod
    def _epac_is_need_memm(obj):
        if type(obj) is np.ndarray:
            num_float = 1
            for ishape in obj.shape:
                num_float = num_float * ishape
            # equal to 100 * 1024 * 1024 which means 100 MB
            if num_float * 8 > 104857600:
                return True
        return False

    @staticmethod
    def _pickle_dump(obj, filename):
        output = open(filename, 'w+')
        pickle.dump(obj, output)
        output.close()

    @staticmethod
    def _pickle_load(filename):
        infile = open(filename, 'rb')
        obj = pickle.load(infile)
        infile.close()
        return obj

    @staticmethod
    def dump(obj, filename):
        filename_memobj = filename + conf.MEMOBJ_SUFFIX
        filename_norobj = filename + conf.NOROBJ_SUFFIX
        mem_obj, normal_obj, _ = extract_values(obj,
                                             func_is_big_nparray)
        joblib.dump(mem_obj, filename_memobj)
        epac_joblib._pickle_dump(normal_obj, filename_norobj)

        outfile = open(filename, "w+")
        outfile.write(conf.MEMOBJ_SUFFIX)
        outfile.write("\n")
        outfile.write(conf.NOROBJ_SUFFIX)
        outfile.write("\n")
        outfile.close()

    @staticmethod
    def load(filename, mmap_mode="r+"):
        filename_memobj = filename + conf.MEMOBJ_SUFFIX
        filename_norobj = filename + conf.NOROBJ_SUFFIX
        mem_obj = None
        normal_obj = None
        # Read index file
        infile = open(filename, "rb")
        lines = infile.readlines()
        infile.close()
        for i in xrange(len(lines)):
            lines[i] = lines[i].strip("\n")
        filename_memobj = filename + lines[0]
        filename_norobj = filename + lines[1]
        # Load Memory obj and Normal obj
        mem_obj = joblib.load(filename_memobj, mmap_mode)
        normal_obj = epac_joblib._pickle_load(filename_norobj)
        # Replace mem_obj (extracted values)
        normal_obj, _ = replace_values(normal_obj, mem_obj)
        return normal_obj


class Store(object):
    """Abstract Store"""

    @abstractmethod
    def save(self, key, obj, merge=False):
        """Store abstract method"""

    @abstractmethod
    def load(self, key):
        """Store abstract method"""


class StoreMem(Store):
    """ Store based on memory"""

    def __init__(self):
        self.dict = dict()

    def save(self, key, obj, merge=False):
        if not merge or not (key in self.dict):
            self.dict[key] = obj
        else:
            v = self.dict[key]
            if isinstance(v, dict):
                v.update(obj)
            elif isinstance(v, list):
                v.append(obj)

    def load(self, key):
        try:
            return self.dict[key]
        except KeyError:
            return None


class StoreFs(Store):
    """ Store based of file system

    Parameters
    ----------
    dirpath: str
        Root directory within file system

    clear: boolean
        If True clear (delete) everything under the root directory.

    """

    def __init__(self, dirpath, clear=False):

        self.dirpath = dirpath
        if clear:
            shutil.rmtree(self.dirpath)
        if not os.path.isdir(self.dirpath):
            os.mkdir(self.dirpath)

    def save(self, key, obj, protocol="txt", merge=False):
        """ Save object

        Parameters
        ----------

        key: str
            The primary key

        obj:
            object to be saved

        protocol: str
            "txt": try with JSON if fail use "bin": (pickle)
        """
        #path = self.key2path(key)
        path = os.path.join(self.dirpath, key)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        # JSON
        from epac.configuration import conf
        if protocol is "txt":
            file_path = path + conf.STORE_FS_JSON_SUFFIX
            json_failed = self.save_json(file_path, obj)
        if protocol is "bin" or json_failed:
            # saving in json failed => pickle
            file_path = path + conf.STORE_FS_PICKLE_SUFFIX
            self.save_pickle(file_path, obj)

    def load(self, key=""):
        """Load everything that is prefixed with key.

        Parmaters
        ---------
        key: str
            if key point to a file (without the extension), return the file
            if key point to a directory, return a dictionary where
            values are objects corresponding to all files found in all
            sub-directories. Values are indexed with their keys.
            if key is an empty string, assume dirpath is a tree root.

        See Also
        --------
        BaseNode.save()
        """
        from epac.configuration import conf
        from epac.workflow.base import key_pop
        path = os.path.join(self.dirpath, key)
        # prefix = os.path.join(path, conf.STORE_FS_NODE_PREFIX)
        if os.path.isfile(path + conf.STORE_FS_PICKLE_SUFFIX):
            loaded_node = self.load_pickle(path + conf.STORE_FS_PICKLE_SUFFIX)
            return loaded_node
        if os.path.isfile(path + conf.STORE_FS_JSON_SUFFIX):
            loaded_node = self.load_pickle(path + conf.STORE_FS_JSON_SUFFIX)
            return loaded_node
        if os.path.isdir(path):
            filepaths = []
            for base, dirs, files in os.walk(self.dirpath):
                #print base, dirs, files
                for filepath in [os.path.join(base, basename) for
                                 basename in files]:
                    _, ext = os.path.splitext(filepath)
                    if not ext == ".npy" and not ext == ".enpy":
                        filepaths.append(filepath)
            loaded = dict()
            dirpath = os.path.join(self.dirpath, "")
            for filepath in filepaths:
                _, ext = os.path.splitext(filepath)
                if ext == conf.STORE_FS_JSON_SUFFIX:
                    key1 = filepath.replace(dirpath, "").\
                        replace(conf.STORE_FS_JSON_SUFFIX, "")
                    obj = self.load_json(filepath)
                    loaded[key1] = obj
                elif ext == conf.STORE_FS_PICKLE_SUFFIX:
                    key1 = filepath.replace(dirpath, "").\
                        replace(conf.STORE_FS_PICKLE_SUFFIX, "")
                    loaded[key1] = self.load_pickle(filepath)
                elif ext == ".npy" or ext == ".enpy":
                    # joblib files
                    pass
                else:
                    raise IOError('File %s has an unkown extension: %s' %
                                  (filepath, ext))
            if key == "":  # No key provided assume a whole tree to load
                tree = loaded.pop(conf.STORE_EXECUTION_TREE_PREFIX)
                for key1 in loaded:
                    key, attrname = key_pop(key1)
                    #attrname, ext = os.path.splitext(basename)
                    if attrname != conf.STORE_STORE_PREFIX:
                        raise ValueError('Do not know what to do with %s') \
                            % key1
                    node = tree.get_node(key)
                    if not node.store:
                        node.store = loaded[key1]
                    else:
                        keys_local = node.store.dict.keys()
                        keys_disk = loaded[key1].dict.keys()
                        if set(keys_local).intersection(set(keys_disk)):
                            raise KeyError("Merge store with same keys")
                        node.store.dict.update(loaded[key1].dict)
                loaded = tree
            return loaded

    def save_pickle(self, file_path, obj):
        epac_joblib.dump(obj, file_path)
#        output = open(file_path, 'wb')
#        pickle.dump(obj, output)
#        output.close()

    def load_pickle(self, file_path):
#        u'/tmp/store/KFold-0/SVC/__node__NodeEstimator.pkl'
#        inputf = open(file_path, 'rb')
#        obj = pickle.load(inputf)
#        inputf.close()
        from epac.utils import try_fun_num_trials
        kwarg = {"filename": file_path}
        obj = try_fun_num_trials(epac_joblib.load,
                                 ntrials=10,
                                 **kwarg)
        # obj = joblib.load(filename=file_path)
        return obj

    def save_json(self, file_path,  obj):
        obj_dict = obj_to_dict(obj)
        output = open(file_path, 'wb')
        try:
            json.dump(obj_dict, output)
        except TypeError:  # save in pickle
            output.close()
            os.remove(file_path)
            return 1
        output.close()
        return 0

    def load_json(self, file_path):
        inputf = open(file_path, 'rb')
        obj_dict = json.load(inputf)
        inputf.close()
        return dict_to_obj(obj_dict)


## ============================== ##
## == Conversion Object / dict == ##
## ============================== ##

# Convert object to dict and dict to object for Json Persistance
def obj_to_dict(obj):
    # Composite objects (object, dict, list): recursive call
    if hasattr(obj, "__dict__") and hasattr(obj, "__class__")\
        and hasattr(obj, "__module__") and not inspect.isfunction(obj):  # object: rec call
        obj_dict = {k: obj_to_dict(obj.__dict__[k]) for k in obj.__dict__}
        obj_dict["__class_name__"] = obj.__class__.__name__
        obj_dict["__class_module__"] = obj.__module__
        return obj_dict
    elif inspect.isfunction(obj):                     # function
        obj_dict = {"__func_name__": obj.func_name,
                    "__class_module__": obj.__module__}
        return obj_dict
    elif isinstance(obj, dict):                       # dict: rec call
        return {k: obj_to_dict(obj[k]) for k in obj}
    elif isinstance(obj, (list, tuple)):              # list: rec call
        return [obj_to_dict(item) for item in obj]
    elif isinstance(obj, np.ndarray):                 # array: to list
        return {"__array__": obj.tolist()}
    else:
        return obj


def dict_to_obj(obj_dict):
    if isinstance(obj_dict, dict) and '__class_name__' in obj_dict:  # object
        cls_name = obj_dict.pop('__class_name__')               # : rec call
        cls_module = obj_dict.pop('__class_module__')
        obj_dict = {k: dict_to_obj(obj_dict[k]) for k in obj_dict}
        mod = __import__(cls_module, fromlist=[cls_name])
        obj = object.__new__(eval("mod." + cls_name))
        obj.__dict__.update(obj_dict)
        return obj
    if isinstance(obj_dict, dict) and '__func_name__' in obj_dict:  # function
        func_name = obj_dict.pop('__func_name__')
        func_module = obj_dict.pop('__class_module__')
        mod = __import__(func_module, fromlist=[func_name])
        func = eval("mod." + func_name)
        return func
    if isinstance(obj_dict, dict) and '__array__' in obj_dict:
        return np.asarray(obj_dict.pop('__array__'))
    elif isinstance(obj_dict, dict):                         # dict: rec call
        return {k: dict_to_obj(obj_dict[k]) for k in obj_dict}
    elif isinstance(obj_dict, (list, tuple)):                # list: rec call
        return [dict_to_obj(item) for item in obj_dict]
#    elif isinstance(obj, np.ndarray):                       # array: to list
#        return obj.tolist()
    else:
        return obj_dict


def save_tree(tree_root, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    store = StoreFs(dir_path, clear=True)
    tree_root.save_tree(store=store)


def load_tree(dir_path):
    store_fs = StoreFs(dirpath=dir_path)
    return store_fs.load()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
