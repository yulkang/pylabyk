#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
zipPickle

From http://code.activestate.com/recipes/189972-zip-and-pickle/#c3

Created on Sun Oct 16 12:38:07 2016

@author: Zach Dwiel
"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

import os
import pickle
import gzip
import zlib

def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.

       Written to a temporary file beside the target and renamed into place,
       so a concurrent reader (another worker of the same parallel sweep)
       sees either no file or a complete one, never a truncated gzip.
       Added 2026-09-09 after a cold parallel run hit exactly that.
    """
    tmp = f'{filename}.tmp{os.getpid()}'
    file = gzip.GzipFile(tmp, 'wb')
    try:
        try:
            import torch
            torch.save(object, file, pickle_protocol=protocol)
        except RuntimeError:
            print('Failed to save with torch.save(); trying pickle.dump()')
            pickle.dump(object, file, protocol)
    finally:
        file.close()
    os.replace(tmp, filename)


def load(filename, map_location='cpu', use_torch=True):
    """Loads a compressed object from disk
    """
    if use_torch:
        try:
            import torch
            try:
                with gzip.GzipFile(filename, 'rb') as file:
                    object = torch.load(file, map_location=map_location)
            except (EOFError, zlib.error):
                from send2trash import send2trash
                send2trash(filename)
                print(f'Trashed the corrupted file: {filename}')
                raise
        except RuntimeError:
            print('Failed to load with torch.load(); trying pickle.load()')
            with gzip.GzipFile(filename, 'rb') as file:
                object = pickle.load(file)
    else:
        with gzip.GzipFile(filename, 'rb') as file:
            object = pickle.load(file)
    return object