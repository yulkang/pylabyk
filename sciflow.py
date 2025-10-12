"""
Scientific workflow with Command, Results, and Render

Command keeps parameters for analysis without results,
so that the arguments to reproduce the analysis are clear
without time or space cost.

Results keeps the results of analyses that might take time to run,
so that the results can be cached and reused.

Render keeps human-readable plot, table, or text output,
and keeps it dissociated from the results,
which can be rendered in multiple ways.
"""

#  Copyright (c) 2025  Yul HR Kang. yulkang at kaist dot ac dot kr
import dataclasses
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from numpy.lib.npyio import NpzFile

from pylabyk import np2, plt2
from pylabyk.argsutil import dict2fname
from pylabyk.cacheutil import Cache, mkdir4file
from pylabyk.localfile import LocalFile


@dataclass
class Cacheable:
    """
    Mixin for Command, Results, and Render
    """
    file_kind='cache'
    """
    Used in LocalFile.get_file() and LocalFile.get_cache(),
    as file_kind=label
    file_kind is for distinguishing between Command, Results, and Render.
    such as cmd, res, rdr, plt, tbl, txt, etc.
    
    Keep it as a class attribute in subclasses, 
    so that __init__() can have required arguments. 
    """

    file_ext='.npz'

    label = None
    """
    Used in LocalFile.get_file() and LocalFile.get_cache(),
    as file_kind=label
    label is for distinguishing within Command, Results, and Render.
    The same label should be used across Command and its corresponding Results,
    while Render can have the same label (when there is only a single kind of
    rendering) or different labels (when there are multiple kinds of rendering).

    Keep it as a class attribute in subclasses, 
    so that __init__() can have required arguments. 
    """

    def get_dict_file(self) -> Dict[str, str]:
        return self.asdict()

    def get_fname(self, localfile: LocalFile = None) -> str:
        if localfile is None:
            return dict2fname(self.get_dict_file())
        else:
            return localfile.get_file(
                filekind=self.file_kind, kind=self.label,
                d=self.get_dict_file(),
                ext=self.file_ext,
            )

    def get_cache(self, localfile: LocalFile, allow_pickle=False) -> NpzFile:
        npz = np.load(self.get_fname(localfile), allow_pickle=allow_pickle)
        return npz

    def asdict(self) -> Dict[str, Any]:
        # noinspection PyTypeChecker
        return dataclasses.asdict(self)

    def save(self, localfile: LocalFile, compress=True, allow_pickle=False):
        fname = self.get_fname(localfile)
        mkdir4file(fname)

        if compress:
            np.savez_compressed(
                fname,
                allow_pickle=allow_pickle,
                **self.asdict(),
            )
        else:
            np.savez(
                fname,
                allow_pickle=allow_pickle,
                **self.asdict(),
            )
        # with self.get_cache(localfile) as cache:
        #     cache.set(self.asdict())

    def load(self, localfile: LocalFile):
        npz = self.get_cache(localfile)
        for k, v in npz.items():
            setattr(
                self,
                k,
                np2.scalar2item(v)
            )

        # with self.get_cache(localfile) as cache:
        #     kw = cache.get()
        #     for k, v in kw.items():
        #         setattr(self, k, v)


@dataclass
class Command(Cacheable):
    """
    Command without results.
    Must be instantiated instantaneously.
    """
    file_kind='cmd'


class Results(Cacheable):
    """
    Results of analyses.
    Might take time to run.
    Construct only with a single Command instance.
    """
    file_kind='res'

    command: Command

    def asdict(self) -> Dict[str, Any]:
        d = super().asdict()
        d['command'] = self.command.asdict()
        return d

    def load(self, localfile):
        npz = self.get_cache(localfile)
        for k, v in npz.items():
            if k == 'command':
                v = np2.scalar2item(v)
                for k1, v1 in v.items():
                    setattr(self.command, k1, v1)
            else:
                setattr(
                    self,
                    k,
                    np2.scalar2item(v)
                )

    def get_dict_file(self) -> Dict[str, str]:
        return self.command.get_dict_file()

    @property
    def label(self):
        return self.command.label

    def main(self):
        """
        Run analysis and fill in results.
        Must not have any input arguments.
        """
        raise NotImplementedError()


class Render(Results):
    """Human-readable plot, table, or text output"""
    file_kind='rdr'  # 'plt', 'tbl', 'txt', etc.

    results: Results
    """
    Render must depend on exactly one Command (inherited from Results)
    and one Results.
    If you need multiple Results, make another Results class that contains them.
    """

    def render(self):
        raise NotImplementedError()


class Plot(Render):
    file_kind='plt'
    axs: plt2.GridAxes

    def savefig(self, localfile: LocalFile, **kwargs):
        plt2.savefig(
            fname=self.get_fname(localfile),
            fig=self.axs.figure,
            **kwargs
        )


class Table(Render):
    file_kind='tbl'
    tbl: pd.DataFrame


class Text(Render):
    file_kind='txt'
    txt: str
