#  Copyright (c) 2024  Yul HR Kang. hk2699 at caa dot columbia dot edu.

from multiprocessing.pool import Pool
from typing import Any, Callable, Sequence

from pylabyk import np2


# task executed in a worker process
def task(value):
    # check for failure case
    if value == 2:
        raise Exception('Something bad happened!')
    # report a value
    return value


def demo_fun(a, b):
    if a == 0 and b == 2:
        raise Exception(f'Error: {a} and {b}')
    else:
        return a * 10 + b


# protect the entry point
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    args = [[0, 0, 0], [0, 1, 2]]

    res = np2.vectorize_par(
        demo_fun, args, meshgrid_input=False)

    # res = np2.vectorize_par(np2.TaskResult.run_task, [
    #     np2.arrayobj1d([demo_fun, v]) for v in args
    # ], meshgrid_input=False)  # type: Sequence[np2.TaskResult]
    #
    # res_list = [r.result for r in res]
    # for r in res:  # type: np2.TaskResult
    #     if r.exception:
    #         print(f'Exception: {r.exception}')
    #         try:
    #             r.rerun()
    #         except Exception as e:
    #             print(f'Failed to rerun: {e}')
    #     else:
    #         print(f'Result: {r.result}')
    print('--')
