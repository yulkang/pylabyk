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
from typing import Dict
import pandas as pd

from pylabyk import np2, plt2
from pylabyk.argsutil import dict2fname
from pylabyk.cacheutil import Cache
from pylabyk.localfile import LocalFile


@dataclass
class Cacheable:
    """
    Mixin for Command, Results, and Render
    """
    label: str = None

    def get_dict_file(self) -> Dict[str, str]:
        return np2.rmkeys(self.asdict(), ['label'])

    def get_fname(self, localfile: LocalFile = None) -> str:
        if localfile is None:
            return dict2fname(self.get_dict_file())
        else:
            return localfile.get_file(
                filekind='cache', kind=self.label,
                d=self.get_dict_file()
            )

    def get_cache(self, localfile: LocalFile) -> Cache:
        return localfile.get_cache(
            self.label,
            self.get_dict_file()
        )

    def asdict(self) -> dict:
        # noinspection PyTypeChecker
        return dataclasses.asdict(self)

    def save(self, localfile: LocalFile):
        with self.get_cache(localfile) as cache:
            cache.set(self.asdict())

    def load(self, localfile: LocalFile):
        with self.get_cache(localfile) as cache:
            kw = cache.get()
            for k, v in kw.items():
                setattr(self, k, v)


@dataclass
class Command(Cacheable):
    """
    Command without results.
    Must be instantiated instantaneously.
    """
    pass


class Results(Cacheable):
    """
    Results of analyses.
    Might take time to run.
    """
    command: Command

    def get_dict_file(self) -> Dict[str, str]:
        return self.command.get_dict_file()

    @property
    def label(self):
        return self.command.label

    def get_cache(self, localfile: LocalFile) -> Cache:
        return localfile.get_cache(
            self.label,
            self.get_dict_file()
        )

    def main(self):
        """
        Run analysis and fill in results.
        """
        raise NotImplementedError()


class Render(Cacheable):
    """Human-readable plot, table, or text output"""
    command_render: Command
    results: Results
    render_kind: str = 'plt'  # 'plt', 'tbl', 'txt', etc.

    @property
    def label(self):
        return self.results.label

    @property
    def command_results(self):
        return self.results.command

    def get_dict_file(self) -> Dict[str, str]:
        return self.command_render.get_dict_file()

    def get_fname(self, localfile: LocalFile = None) -> str:
        return dict2fname({
            **{self.render_kind: self.label},
            **self.get_dict_file()
        })

    def render(self):
        raise NotImplementedError()


class Plot(Render):
    render_kind: str = 'plt'
    axs: plt2.GridAxes = None


class Table(Render):
    render_kind: str = 'tbl'
    tbl: pd.DataFrame = None


class Text(Render):
    render_kind: str = 'txt'
    txt: str = None