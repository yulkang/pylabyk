#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import torch
import matplotlib.pyplot as plt
from lib.pylabyk import numpytorch as npt
from lib.pylabyk.numpytorch import npy, npys


def print_demo(p, fun):

    out = fun(p)

    print('-----')
    print('fun: %s' % fun.__name__)
    print('p:')
    print(p)
    print('out[0]:')
    print(out[0])

    print('out[1]:')
    print(out[1])

    print('out[0].sum(), out[1].sum()')
    print(out[0].sum(), out[1].sum())


if __name__ == '__main__':
    for p, fun in [
        (torch.tensor([
            [0., 1.],
            [0.5, 0.5]
        ]) * 1., npt.min_distrib),
        (torch.tensor([
            [1., 0.],
            [0.5, 0.5]
        ]) * 1., npt.min_distrib),
        (torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5]
        ]) * 0.1, npt.min_distrib),
        (torch.tensor([
            [0., 1.],
            [0.5, 0.5]
        ]) * 1., npt.max_distrib),
        (torch.tensor([
            [1., 0.],
            [0.5, 0.5]
        ]) * 1., npt.max_distrib),
        (torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5]
        ]) * 0.1, npt.max_distrib),
    ]:
        print_demo(p, fun)
