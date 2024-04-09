#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.
from datetime import datetime, timedelta
import time

from . import cacheutil, np2
from . import argsutil
from .cacheutil import Cache
import os, shutil, sys
from collections import OrderedDict as odict
from typing import Union, Iterable, Tuple
from .cacheutil import mkdir4file
from .argsutil import (dict2fname, fname2dict,
    kwdef, merge_fileargs, fullpath2hash)


def replace_ext(fullpath, ext_new):
    """

    :param fullpath:
    :param ext_new: Should start with a '.'
    :return: fullpath with the old extension replaced with ext_new
    """
    fp = os.path.splitext(fullpath)[0]
    return fp + ext_new


def add_subdir(fullpath: str, subdir: str) -> str:
    pth, nam = os.path.split(fullpath)
    return os.path.join(pth, subdir, nam)


def copy2subdir(fullpath: str, subdir: str = None, verbose=True) -> str:
    """

    :param fullpath:
    :param subdir: if None or '', skip copying
    :param verbose:
    :return: full path to the destination
    """
    if subdir is None or subdir == '':
        return fullpath

    dst = add_subdir(fullpath, subdir)
    pth = os.path.split(dst)[0]
    if not os.path.exists(pth):
        os.mkdir(pth)
    shutil.copy2(fullpath, dst)
    if verbose:
        print('Copied to %s' % dst)
    return dst


def fname_only(fullpath: str) -> str:
    return os.path.split(fullpath)[1]


class LocalFile(object):
    replace_ext = replace_ext

    def __init__(
        self,
        pth_root='../Data',
        subdir_default='',
        cache_dir='cache',
        ext_fig='.png',  # .png is much faster than .pdf (~5x)
        # ext_fig='.pdf',
        kind2subdir=False,
        shorten_dict=True,
    ):
        self.pth_root = pth_root
        self.subdir_default = subdir_default
        self.cache_dir = cache_dir
        self.ext_fig = ext_fig
        self.kind2subdir=kind2subdir
        self.shorten_dict = shorten_dict

    def dict2fname(self, d: dict) -> str:
        if self.shorten_dict:
            d = np2.shorten_dict(d)
        return dict2fname(d)

    def fname2dict(self, fname: str, lengthen=False) -> dict:
        if lengthen:
            raise NotImplementedError()
        return fname2dict(fname)

    def get_pth_out(self, subdir=None):
        if subdir is None:
            subdir = self.subdir_default
        if isinstance(subdir, dict):
            subdir = self.dict2fname(subdir)
        pth_out = os.path.join(self.pth_root, subdir)
        return pth_out

    def get_pth_cache(self, subdir=None, cache_dir=None):
        if cache_dir is None:
            cache_dir = self.cache_dir
        pth_cache = os.path.join(
            self.get_pth_out(subdir), cache_dir)
        if not os.path.exists(pth_cache):
            mkdir4file(pth_cache)
        return pth_cache

    def get_file_cache(
            self,
            cache_kind: str,
            d: [Iterable[tuple], dict, odict, None],
            subdir=None,
            # cache_dir=None
    ) -> str:
        """
        """
        return self.get_file(
            filekind='cache', kind=cache_kind,
            d=d, ext='.zpkl', subdir=subdir
        )
        # return os.path.join(
        #     self.get_pth_cache(subdir, cache_dir=cache_dir),
        #     self.dict2fname(d) + '.zpkl'
        # )

    def get_file0(self, file: str, subdir=''):
        return os.path.join(
            self.get_pth_out(subdir), file
        )

    def get_file(
        self, filekind: str = '', kind: str = '',
        d: Union[Iterable[tuple], dict, odict, str, None] = None,
        ext=None, subdir=None,
        max_len=250,
        return_exists=False,
        return_fname0=False,
    ) -> Union[str, Tuple[str, bool]]:
        """
        :type filekind: str
        :type kind: str
        :type d: Union[Iterable[tuple], dict, odict, None]
        :type ext: str
        :rtype: str
        """
        if ext is None:
            ext = '.' + filekind
        if d is None:
            d = {}
        elif isinstance(d, str):
            pass
        elif not (type(d) is list):
            d = [d]

        if isinstance(d, str):
            fname = d
        else:
            kw_fname = kwdef(
                    merge_fileargs(d),
                    {},
                    sort_merged=False, sort_given=True, def_bef_given=True
                )
            fname = self.dict2fname(merge_fileargs(kw_fname))

        if len(filekind) > 0 or len(kind) > 0:
            fname = '%s=%s+%s' % (filekind, kind, fname)

        if subdir is None:
            if self.kind2subdir:
                subdir = filekind + '=' + kind
            else:
                subdir = ''
        elif isinstance(subdir, dict):
            subdir = self.dict2fname(d)
            if self.kind2subdir:
                subdir = os.path.join(subdir, filekind + '=' + kind)
        else:
            assert isinstance(subdir, str)

        fullpath = os.path.join(
            self.get_pth_out(subdir), fname + ext
        )
        fullpath = fullpath.replace('\\\\', '\\')  # remove duplicate backslashes in Windows

        fname0 = fname
        if len(fname) > max_len:
            fname = fullpath2hash(fname0)
            fullpath_short = os.path.join(
                self.get_pth_out(subdir), fname + ext
            )
            fullpath_short_txt = fullpath_short + '.hash.txt'
            exists = os.path.exists(fullpath_short + '.hash.txt')

            fullpath_short_txt = fullpath_short_txt.replace('\\\\', '\\')  # remove duplicate backslashes in Windows
            mkdir4file(fullpath_short_txt)
            with open(fullpath_short_txt, 'w') as f:
                f.write(fname0)
            print(f'File name too long: {fname0}\n'
                  f'    Writing instead to {fname}\n'
                  f'    and recording the original name in '
                  f'{fullpath_short_txt}')
            fullpath = fullpath_short
        else:
            exists = os.path.exists(fullpath)

        assert not (return_exists and return_fname0)

        if return_exists:
            return fullpath, exists
        elif return_fname0:
            return fullpath, fname0
        else:
            return fullpath

    def get_cache(
            self, cache_kind: str,
            d: Union[str, dict] = None,
            subdir: Union[str, dict] = None,
            ignore_key=True,
            **kwargs) -> Cache:
        """
        :type cache_kind: str
        :type d: Union[Iterable[tuple], dict, odict, None]
        """
        if subdir is None and self.kind2subdir:
            subdir = 'cache=%s' % cache_kind

        fname = self.get_file_cache(
            cache_kind=cache_kind, d=d, subdir=subdir
        )
        return Cache(fname, **{
            'ignore_key': ignore_key,
            **kwargs
        })

    def get_file_fig(self, fig_kind,
                     d: Union[Iterable[tuple], dict, odict, None] = None,
                     ext=None, subdir=None) -> str:
        """
        """
        if ext is None:
            ext = self.ext_fig
        if self.kind2subdir and subdir is None:
            subdir = 'plt=' + fig_kind
        return self.get_file('plt', fig_kind, d=d, ext=ext, subdir=subdir)


    def get_file_csv(self, kind,
                     d: Union[Iterable[tuple], dict, odict, None] = None,
                     ext='.csv', subdir=None) -> str:
        """
        """
        if self.kind2subdir and subdir is None:
            subdir = 'tab=' + kind
        return self.get_file('tab', kind, d, ext='.csv', subdir=subdir)


def get_utc_offset():
    # Getting the current local time
    local_time = datetime.now()

    # Getting the current UTC time
    utc_time = datetime.utcnow()

    # Calculating the offset
    offset = local_time - utc_time

    # Adjusting for daylight saving time
    if time.localtime().tm_isdst:
        offset += timedelta(hours=1)

    # Formatting the offset
    hours_offset = int(offset.total_seconds() / 3600)
    offset_str = f"UTC{'+' if hours_offset >= 0 else ''}{hours_offset}"
    return offset_str


class DualOutput(object):
    """
    make it such that stdout is written to both the terminal and a file

    Usage:
    with DualOutput('out.txt'):
        print('hello')
    """
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename, 'a')
        self.stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.file.flush()
        os.fsync(self.file.fileno())
        # self.file.close()
        # self.file = open(self.filename, 'a')
        self.stdout.write(text)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.file.close()
        self.file.flush()
        os.fsync(self.file.fileno())

