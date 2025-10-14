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
from pylabyk.cacheutil import mkdir4file
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

    exclude_from_cache=()
    exclude_from_fname=()
    allow_pickle=False

    def get_dict_file(self) -> Dict[str, str]:
        return np2.rmkeys(self.asdict(), self.exclude_from_fname)

    def get_fname(
        self,
        localfile: LocalFile = None,
        dict_file: Dict[str, Any] = ()
    ) -> str:
        dict_file = {**self.get_dict_file(), **dict(dict_file)}
        if localfile is None:
            return dict2fname(dict_file)
        else:
            return localfile.get_file(
                filekind=self.file_kind,
                kind=self.label,
                d=dict_file,
                ext=self.file_ext,
            )

    def get_cache(self, localfile: LocalFile) -> NpzFile:
        npz = np.load(self.get_fname(localfile), allow_pickle=self.allow_pickle)
        return npz

    def asdict(self) -> Dict[str, Any]:
        # noinspection PyTypeChecker
        return dataclasses.asdict(self)

    def asdict_for_cache(self) -> Dict[str, Any]:
        return np2.rmkeys(self.asdict(), self.exclude_from_cache)

    def save(self, localfile: LocalFile, compress=True):
        fname = self.get_fname(localfile)
        mkdir4file(fname)

        d = np2.rmkeys(self.asdict(), self.exclude_from_cache)

        if compress:
            np.savez_compressed(fname, allow_pickle=self.allow_pickle, **d)
        else:
            np.savez(fname, allow_pickle=self.allow_pickle, **d)
        # with self.get_cache(localfile) as cache:
        #     cache.set(self.asdict())

    def load(self, localfile: LocalFile):
        npz = self.get_cache(localfile)
        for k, v in npz.items():
            if k not in self.exclude_from_cache:
                setattr(self, k, np2.scalar2item(v))

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
    exclude_from_cache=('command',)

    command: Command

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
    exclude_from_cache=('command', 'results',)
    exclude_from_fname=('results',)

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
    file_ext = '.pdf'

    axs: plt2.GridAxes = None

    def savefig(self, localfile: LocalFile, **kwargs):
        plt2.savefig(
            fname=self.get_fname(localfile),
            fig=self.axs.figure,
            **kwargs
        )


class Table(Render):
    file_kind='tbl'
    file_ext = '.csv'

    tbl: pd.DataFrame = None


class Text(Render):
    file_kind='txt'
    file_ext = '.txt'

    txt: str = None
