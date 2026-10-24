#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
zipPickle

From http://code.activestate.com/recipes/189972-zip-and-pickle/#c3

Created on Sun Oct 16 12:38:07 2016

@author: Zach Dwiel
"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

import hashlib
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

       The data is flushed to disk before the rename, so a crash or power
       cut leaves either the previous state or the complete file under
       `filename`, never an empty one: XFS and ext4 can commit a rename
       before the data it points to. Added 2026-10-21.

       The temporary file is named by a hash of the target rather than the
       target plus a suffix: cache names run up to 253 bytes, and the
       suffix took a 245-byte one past the 255-byte limit on a name
       (coco1, 7-digit pids; 2026-10-24).
    """
    folder, name = os.path.split(filename)
    tag = hashlib.md5(name.encode()).hexdigest()[:16]
    tmp = os.path.join(folder, f'.{tag}.tmp{os.getpid()}')
    with open(tmp, 'wb') as raw:
        file = gzip.GzipFile(fileobj=raw, mode='wb')
        try:
            try:
                import torch
                torch.save(object, file, pickle_protocol=protocol)
            except RuntimeError:
                print('Failed to save with torch.save(); trying pickle.dump()')
                pickle.dump(object, file, protocol)
        finally:
            file.close()
        raw.flush()
        os.fsync(raw.fileno())
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