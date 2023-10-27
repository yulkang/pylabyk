#  Copyright (c) 2022  Yul HR Kang. hk2699 at caa dot columbia dot edu.


from typing import Dict, Callable, Any, List
from types import ModuleType
from matplotlib import pyplot as plt
from pylabyk import plt2


class PlotRecord:
    """
    UNUSED: perhaps not better than pickling matplotlib object directly
    Use plt2.savefig() and loadfig() instead
    """
    _module: Dict[str, ModuleType]
    _fun: Dict[str, Callable]
    _rec: List[(str, Dict)]

    def __init__(self):
        self.add_module('plt', plt)
        self.add_module('plt2', plt2)

    def add_module(self, module_name: str, module: ModuleType):
        self._module[module_name] = module

    def add_fun(self, fun_name: str, fun: Callable):
        self._fun[fun_name] = fun

    def call_fun(self, name: str, kwargs: Dict[str, Any]) -> Any:
        """

        :param name: fun_name or module_name.fun_name
        :param kwargs: used to call the function
        :return: return value from the function call
        """
        if name in self._fun:
            fun = self._fun[name]
        else:
            split_name = name.split(sep='.')
            module_name = '.'.join(split_name[:-1])
            fun_name = split_name[-1]
            fun = self._module[module_name].__getattribute__(fun_name)
        return fun(**kwargs)

    def record(self, name: str, kwargs: Dict[str, Any] = (), to_call=False):
        kwargs = dict(kwargs)
        self._rec.append((name, kwargs))
        if to_call:
            return self.call_fun(name, kwargs)