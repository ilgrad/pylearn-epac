# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:07:22 2013

@author: jinpeng.li@cea.fr
"""
from multiprocessing import Pool


class MemmapMatrix:
    """
    To save parameters of numpy.memmap so that numpy.memmap can be pickled.


    Example
    -------
    >>> import numpy as np
    >>> from epac.aux_memmap import MemmapMatrix
    >>> mem_mat = np.memmap("/tmp/test.data",
    ...                  dtype='float32',
    ...                  mode='w+',
    ...                  shape=(3, 4))
    >>> mem_mat[1,1] = 5
    >>> print mem_mat
    [[ 0.  0.  0.  0.]
     [ 0.  5.  0.  0.]
     [ 0.  0.  0.  0.]]
    >>> mmatrix = MemmapMatrix(mem_mat)
    >>> mem_mat_cp = mmatrix.get()
    >>> print mem_mat_cp
    [[ 0.  0.  0.  0.]
     [ 0.  5.  0.  0.]
     [ 0.  0.  0.  0.]]

    """
    def __init__(self, mem_mat):
        self.filename = mem_mat.filename
        self.dtype = mem_mat.dtype.name
        self.shape = mem_mat.shape

    def get(self, mode="r+"):
        """
        Parameter
        ---------
        mode : str
            see the definition of mode in numpy.memmap
        """
        import numpy as np
        return np.memmap(self.filename,\
                        dtype=self.dtype,\
                        mode=mode,\
                        shape=self.shape)


class MPSafeData:
    """
    Convert all the data to multiprocessing safe data

    Example
    -------

    >>> import numpy as np
    >>> from epac.aux_memmap import MPSafeData
    >>> mem_mat = np.memmap("/tmp/test.data",
    ...                     dtype='float32',
    ...                     mode='w+',
    ...                     shape=(3, 4))
    >>> mem_mat[1, 1] = 5
    >>> dict_data = {}
    >>> dict_data["mem_mat"] = mem_mat
    >>> dict_data["test_int"] = int(5)
    >>> safe_dict = MPSafeData()
    >>> safe_dict.encode(dict_data)
    >>> dict_data_cp = safe_dict.decode()
    >>> for key in dict_data_cp:
    ...     print key, "=", dict_data_cp[key]
    ...
    test_int = 5
    mem_mat = [[ 0.  0.  0.  0.]
     [ 0.  5.  0.  0.]
     [ 0.  0.  0.  0.]]

    """
    def __init__(self):
        self.data = None
        self.type = None

    def encode(self, data):
        if type(data) is dict:
            self.type = dict
            self._encode_dict(data)

    def decode(self):
        if self.type is dict:
            return self._decode_dict()
        return None

    def _encode_dict(self, data):
        import numpy as np
        self.data = {}
        for key in data:
            if type(data[key]) is np.memmap:
                self.data[key] = MemmapMatrix(data[key])
            else:
                self.data[key] = data[key]

    def _decode_dict(self):
        if not self.data:
            return None
        data = {}
        for key in self.data:
            if isinstance(self.data[key], MemmapMatrix):
                data[key] = self.data[key].get()
            else:
                data[key] = self.data[key]
        return data


def _safe_map(func, safe_element):
    element = safe_element.decode()
    if not element:
        raise ValueError("cannot find correct decoder for _safe_map")
    ret = func(safe_element)
    safe_ret = MPSafeData(ret)
    return safe_ret

class SafePool:
    def __init__(self, processes):
        self.processes = processes

    def map(func, iterable):
        pool = Pool(processes=self.processes)
        safe_elements = []
        for dict_elem in iterable:
            safe_element = MPSafeData(dict_elem)
            safe_elements.append(safe_element)
        
        pool.map()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
