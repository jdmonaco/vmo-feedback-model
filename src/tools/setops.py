# encoding: utf-8
"""
setops.py -- Making built-in set operations available for numpy arrays

Created by Joe Monaco on 2008-06-19.
Copyright (c) 2008 Columbia University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""


import numpy as np


# Generic set-to-array translation function

def _do_set_op(u, v, set_op):
    assert type(u) is np.ndarray and type(v) is np.ndarray, 'need arrays'
    u_func = getattr(set(u), set_op)
    return np.array(list(u_func(set(v))))


# Create set operation functions

def intersection(u, v):
    """Get array intersection of input arrays u and v"""
    return _do_set_op(u, v, 'intersection')
    
def union(u, v):
    """Get array union of input arrays u and v"""
    return _do_set_op(u, v, 'union')
    
def difference(u, v):
    """Get array difference of input arrays u and v"""
    return _do_set_op(u, v, 'difference')
    
def symmetric_difference(u, v):
    """Get array symmetric_difference of input arrays u and v"""
    return _do_set_op(u, v, 'symmetric_difference')

    
# _ops = ('intersection', 'union', 'difference', 'symmetric_difference')
# for _op in _ops:
#     # tmp = lambda u, v: _do_set_op(u, v, _op)
#     def tmp(u, v): return _do_set_op(u, v, _op)
#     tmp.__doc__ = "Get array %s of input arrays u and v"%_op
#     exec '%s = tmp'%_op
#     

