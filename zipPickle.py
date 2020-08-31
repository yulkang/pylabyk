#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
zipPickle

From http://code.activestate.com/recipes/189972-zip-and-pickle/#c3

Created on Sun Oct 16 12:38:07 2016

@author: Zach Dwiel
"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

import pickle
import gzip

def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(object, file, protocol)
    # torch.save(object, file, pickle_protocol=protocol)
    file.close()

def load(filename, map_location='cpu'):
    """Loads a compressed object from disk
    """
    try:
        with gzip.GzipFile(filename, 'rb') as file:
            object = pickle.load(file)
    except RuntimeError:
        import torch
        with gzip.GzipFile(filename, 'rb') as file:
            object = torch.load(file, map_location=map_location)
    return object