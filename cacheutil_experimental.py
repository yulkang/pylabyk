#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

import os
from . import zipPickle
from collections import OrderedDict
from .argsutil import dict2fname

"""
Deprecated subkeys separate from keys. 
Now all input/output must be done using key to Cache._dict[key].
Use Cache.get_dict() to get the whole _dict.
"""

def dict_except(d, keys_to_excl):
    return {k:d[k] for k in d if k not in keys_to_excl}

def obj2dict(obj, keys_to_excl=[], exclude_hidden=True):
    d = obj.__dict__
    if exclude_hidden:
        d = {k:d[k] for k in d if k[0] != '_'}
    if len(keys_to_excl) > 0:
        d = {k:d[k] for k in d if k not in keys_to_excl}
    return d

def dict2obj(d, obj):
    for k in d:
        obj.__dict__[k] = d[k]
    return obj

class Cache(object):
    """
    Caches to file for fast retrieval.

    EXAMPLE:
    import cacheutil
    def fun(test_param=1, to_recompute=False):
        cache = cacheutil.Cache()
        if cache.exists(['a', 'b']) and not to_recompute:
            a, b = cache.gets([
                'a', 'b'
            ])
        else:
            a = 10 * test_param
            b = 100 * test_param

            cache.update_dict({
                'a': a,
                'b': b
            })
        print('test_param: %d, a: %d, b:%d' % (test_param, a, b))

    for test_param_main in range(5):
        fun(test_param_main)
    """
    def __init__(self, fullpath='cache.zip.pkl', verbose=True, key=None):
        self.fullpath = fullpath
        self.verbose = verbose
        self._dict = {}
        self.to_save = False
        self.key = key
        if os.path.exists(self.fullpath):
            self._dict = zipPickle.load(self.fullpath)

    def format_key(self, key):
        """
        :param key: non-None object that converts into a string, e.g., locals()
        :rtype: str
        """
        return '%s' % key

    def exist(self, key=None):
        """
        :param key: non-None object that converts into a string, e.g., locals()
        :rtype: bool
        """
        if key is None:
            key = self.key
        return self.format_key(key) in self._dict

    def exists(self, keys):
        return (self.exist(key) for key in keys)

    def get(self, key):
        """
        :param key: non-None object that converts into a string, e.g., locals()
        :rtype: Any
        """
        if key is None:
            key = self.key
        if self.verbose and self.exists(key):
            print('Loaded cache from %s' % self.fullpath)
        v = self._dict[self.format_key(key)]
        return v

    def gets(self, keys):
        return (self.get(key) for key in keys)

    def get_dict(self):
        return self._dict

    def set(self, data, key=None):
        """
        Store the data in the cache.
        :param data: any data
        :param key: non-None object that converts into a string, e.g., locals()
            if None, set to self.key
        :rtype: None
        """
        if key is None:
            key = self.key
        self._dict[self.format_key(key)] = data
        self.to_save = True

    def update_dict(self, d):
        """
        Updates self.dict using keys in d
        :type d: dict
        """
        for key in d.keys():
            self._dict[key] = d[key]
        self.to_save = True

    def save(self):
        pth = os.path.dirname(self.fullpath)
        if not os.path.exists(pth) and pth != '':
            os.mkdir(pth)
        zipPickle.save(self._dict, self.fullpath)
        if self.verbose:
            print('Saved cache to %s' % self.fullpath)
        # with open(self.fullpath, 'w+b') as cache_file:
        #     pickle.dump(self.dict, cache_file)
        #     if self.verbose:
        #         print('Saved cache to %s' % self.fullpath)

    def __del__(self):
        if self.to_save:
            self.save()

# if __name__ == '__main__':
#     def fun(test_param=1, to_recompute=False):
#         cache = Cache(
#             os.path.join('cache.pkl'),
#             locals()
#         )
#         if cache.exists() and not to_recompute:
#             a, b = cache.getdict([
#                 'a', 'b'
#             ])
#         else:
#             a = 10 * test_param
#             b = 100 * test_param
#
#             cache.set({
#                 'a': a,
#                 'b': b
#             })
#         print('test_param: %d, a: %d, b:%d' % (test_param, a, b))
#
#     for test_param_main in range(5):
#         fun(test_param_main)