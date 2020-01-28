from lib.pylabyk import cacheutil
from lib.pylabyk import argsutil
import os
from collections import OrderedDict as odict

class LocalFile(object):
    def __init__(
            self,
            pth_root='../Data',
            subdir_default=''
    ):
        self.pth_root = pth_root
        self.subdir_default = subdir_default

    def get_pth_out(self, subdir=None):
        if subdir is None:
            subdir = self.subdir_default
        pth_out = os.path.join(self.pth_root, subdir)
        if not os.path.exists(pth_out):
            os.mkdir(pth_out)
        return pth_out

    def get_pth_cache(self, subdir=None):
        pth_cache = os.path.join(
            self.get_pth_out(subdir), 'cache')
        if not os.path.exists(pth_cache):
            os.mkdir(pth_cache)
        return pth_cache

    def get_file_cache(self, d, subdir=None):
        """
        :type d: Union[list, dict, None]
        :rtype: str
        """
        return os.path.join(
            self.get_pth_cache(subdir),
            cacheutil.dict2fname(d) + '.pkl.zip'
        )

    def get_cache(self, cache_kind, d=None, subdir=None):
        """
        :type cache_kind: str
        :type d: Union[list, dict, None]
        :rtype: cacheutil.Cache
        """
        if d is None:
            d = [{}]
        elif not (type(d) is list):
            d = [d]
        return cacheutil.Cache(
            self.get_file_cache(argsutil.kwdef(
                argsutil.merge_fileargs(d),
                [('cache', cache_kind)]
            ), subdir=subdir)
        )

    def get_file_fig(self, fig_kind, d=None, ext='.png', subdir=None):
        """
        :type fig_kind: str
        :type d: Union[list, dict, None]
        :type ext: str
        :rtype: str
        """
        if d is None:
            d = {}
        return os.path.join(
            self.get_pth_out(subdir), cacheutil.dict2fname(
                argsutil.kwdef(argsutil.merge_fileargs(d), [
                    ('plt', fig_kind)
                ])
            ) + ext
        )