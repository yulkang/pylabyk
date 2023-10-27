#! /usr/bin/env python
# coding: utf-8

# Author: João S. O. Bueno
# Copyright (c) 2009 - Fundação CPqD
# License: LGPL V3.0

# adapted for Python 3.9; copy_dependencies() and find_module_files()
# added by Yul Kang


from types import ModuleType, FunctionType
from typing import Sequence
import sys, os, shutil
from importlib import reload

from traceback import print_exc
# from builtins import ModuleNotFoundError

from pylabyk.cacheutil import mkdir4file


def copy_dependencies(
    module: ModuleType,
    out_dir: str,
    incl: Sequence[str] = (),
    excl: Sequence[str] = (),
) -> (Sequence[str], Sequence[str]):
    """

    :param module:
    :param out_dir:
    :param incl:
    :param excl:
    :return: files_src, files_dst
    """
    files_src = find_module_files(module, incl=incl, excl=excl)
    files_dst = []

    os.mkdir(out_dir)
    cwd = os.getcwd()

    for file in files_src:
        dst = file.replace(cwd, os.path.join(cwd, out_dir))
        mkdir4file(dst)
        files_dst.append(dst)

        print(f'Copying {file.replace(cwd, "")}\nto {dst.replace(cwd, out_dir)}')
        shutil.copy(file, dst)
    print(f'Copied {len(files_src)} files to {out_dir}')
    return files_src, files_dst


def find_module_files(
    module: ModuleType, incl: Sequence[str] = (), excl: Sequence[str] = ()
) -> Sequence[str]:
    """

    :param module:
    :param incl:
    :param excl:
    :return:
    """

    tree = find_all_loaded_modules(module)
    tree_w_file = []
    files = []
    for module in tree:
        try:
            if module.__file__ is not None:
                files.append(module.__file__)
                tree_w_file.append(module)
        except (AttributeError, ImportError):
            print_exc()

    files = [
        f for f in files if
        (any([s in f for s in incl])
         and not any([s in f for s in excl]))
    ]
    return files


def find_all_loaded_modules(module, all_mods = None):
    """
    from https://stackoverflow.com/a/1828014/2565317
    :param module:
    :param all_mods:
    :return:
    """
    if all_mods is None:
        all_mods = {module}
    try:
        for item_name in dir(module):
            try:
                item = getattr(module, item_name)
                if type(item) in [type, FunctionType]:
                    print(f'Found {item.__name__} in {item.__module__}')
                    if item.__module__ is None:
                        continue
                    item = __import__(item.__module__)
                    print('--')
                    # raise RuntimeError(
                    #     f'{item_name} is imported directly in '
                    #     f'{item.__module__} - '
                    #     f'Import module instead to enable finding '
                    #     f'all dependencies!')
            except (AttributeError, ModuleNotFoundError, ImportError):
                # print('--')
                print_exc()
                continue

            if isinstance(item, ModuleType) and not item in all_mods:
                all_mods.add(item)
                find_all_loaded_modules(item, all_mods)
    except ImportError:
        print_exc()
        all_mods = all_mods - {module}

    return all_mods


def find_dependent_modules():
    """gets a one level inversed module dependence tree"""
    tree = {}
    for module in sys.modules.values():
        if module is None:
            continue
        tree[module] = set()
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, ModuleType):
                tree[module].add(attr)
            elif type(attr) in (FunctionType, type):
                tree[module].add(attr.__module__)
    return tree


def get_reversed_first_level_tree(tree):
    """Creates a one level deep straight dependence tree"""
    new_tree = {}
    for module, dependencies in tree.items():
        for dep_module in dependencies:
            if dep_module is module:
                continue
            if not dep_module in new_tree:
                new_tree[dep_module] = {module}  # set([module])
            else:
                new_tree[dep_module].add(module)
    return new_tree


def find_dependants_recurse(key, rev_tree, previous=None):
    """Given a one-level dependance tree dictionary,
       recursively builds a non-repeating list of all dependant
       modules
    """
    if previous is None:
        previous = set()
    if not key in rev_tree:
        return []
    this_level_dependants = set(rev_tree[key])
    next_level_dependants = set()
    for dependant in this_level_dependants:
        if dependant in previous:
            continue
        tmp_previous = previous.copy()
        tmp_previous.add(dependant)
        next_level_dependants.update(
             find_dependants_recurse(dependant, rev_tree,
                                     previous=tmp_previous,
                                    ))
    # ensures reloading order on the final list
    # by postponing the reload of modules in this level
    # that also appear later on the tree
    dependants = (list(this_level_dependants.difference(
                        next_level_dependants)) +
                  list(next_level_dependants))
    return dependants


def get_reversed_tree():
    """
        Yields a dictionary mapping all loaded modules to
        lists of the tree of modules that depend on it, in an order
        that can be used fore reloading
    """
    tree = find_dependent_modules()
    rev_tree = get_reversed_first_level_tree(tree)
    compl_tree = {}
    for module, dependant_modules in rev_tree.items():
        compl_tree[module] = find_dependants_recurse(module, rev_tree)
    return compl_tree


def reload_dependences(module):
    """
        reloads given module and all modules that
        depend on it, directly and otherwise.
    """
    tree = get_reversed_tree()
    reload(module)
    for dependant in tree[module]:
        reload(dependant)