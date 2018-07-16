#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
zipPickle

From http://code.activestate.com/recipes/189972-zip-and-pickle/#c3

Created on Sun Oct 16 12:38:07 2016

@author: Zach Dwiel
"""

import cPickle
import gzip

def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()

def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()

    return object