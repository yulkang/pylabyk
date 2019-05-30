import os
from . import zipPickle
from collections import OrderedDict
from .argsutil import dict2fname

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
    def fun(test_param=1, to_recompute=False):
        cache = Cache(
            os.path.join('cache.pkl'),
            locals()
        )
        if cache.exists() and not to_recompute:
            a, b = cache.getdict([
                'a', 'b'
            ])
        else:
            a = 10 * test_param
            b = 100 * test_param

            cache.set({
                'a': a,
                'b': b
            })
        print('test_param: %d, a: %d, b:%d' % (test_param, a, b))

    for test_param_main in range(5):
        fun(test_param_main)
    """
    def __init__(self, fullpath='cache.zpkl', key=None, verbose=True,
                 ignore_key=False):
        self.fullpath = fullpath
        self.verbose = verbose
        self.dict = {}
        self.to_save = False
        if key is None:
            self.key = None
        else:
            self.key = self.format_key(key)
        self.ignore_key = ignore_key
        if os.path.exists(self.fullpath):
            self.dict = zipPickle.load(self.fullpath)

    def format_key(self, key):
        """
        :param key: non-None object that converts into a string, e.g., locals()
        :rtype: str
        """
        return '%s' % key

    def exists(self, key=None):
        """
        :param key: non-None object that converts into a string, e.g., locals()
        :rtype: bool
        """
        if self.ignore_key:
            return self.dict.__len__() > 0
        if key is None:
            key = self.key
        return self.format_key(key) in self.dict

    def get(self, key=None, subkeys=None):
        """
        :param key: non-None object that converts into a string, e.g., locals()
        :param subkeys: if list, return a tuple of values for the subkeys
        :rtype: Any
        """
        if self.ignore_key:
            key = list(self.dict.keys())[0]
        elif key is None:
            key = self.key
        if self.verbose and self.exists(key):
            print('Loaded cache from %s' % self.fullpath)
        v = self.dict[self.format_key(key)]

        if subkeys is None:
            return v
        else:
            if type(subkeys) is str:
                return v[subkeys]
            else:
                return (v[k] for k in subkeys)

    def getdict(self, subkeys=None):
        """
        Return a tuple of values corresponding to subkeys from default key.
        Assumes that self.dict[key] is itself a dict.
        :type subkeys: list, str
        :param subkeys: list of keys to the cached dict (self.dict[key]).
        :return: a tuple of values corresponding to subkeys from default key.
        """
        return self.get(key=None, subkeys=subkeys)

    def set(self, data, key=None):
        """
        Store the data in the cache.
        :param data: Use dict to allow get() and getdict() to use subkeys.
        :param key: non-None object that converts into a string, e.g., locals()
        :rtype: None
        """
        if key is None:
            # assert self.key is not None, 'default key is not specified!'
            key = self.key
        self.dict[self.format_key(key)] = data
        self.to_save = True

    def save(self):
        pth = os.path.dirname(self.fullpath)
        if not os.path.exists(pth) and pth != '':
            os.mkdir(pth)
        zipPickle.save(self.dict, self.fullpath)
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