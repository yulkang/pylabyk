import pytest
import torch, numpy as np
from .. import numpytorch as npt


def test_sum_log_prob():
    print('')
    for src, dim, keepdim in [
        (torch.tensor([1e-10, 1.]).log(), None, False),
        (torch.arange(-200, -100), None, False),
        (torch.arange(10, 20), None, False),
        (torch.ones([2, 3, 4]), 1, False),
        (torch.ones([2, 3, 4]), 1, True),
        (torch.ones([2, 3, 4]), -1, False),
        (torch.ones([2, 3, 4]), -1, True),
        (torch.ones([2, 3, 4]), [1, -1], False),
        (torch.ones([2, 3, 4]), [1, -1], True),
    ]:
        robust = npt.sum_log_prob(
            src, dim=dim, keepdim=keepdim, robust=True)
        nonrobust = npt.sum_log_prob(
            src, dim=dim, keepdim=keepdim, robust=False)

        diff = float(torch.amax(torch.abs(robust - nonrobust)))
        thres = float(torch.amax(torch.abs(src)) / 100.)
        print('---')
        print('src.shape: ', end='')
        print(src.shape)
        print('dim: ', end='')
        print(dim)
        print(f'keepdim: {keepdim}')
        print('robust.shape: ', end='')
        print(robust.shape)
        print('nonrobust.shape: ', end='')
        print(nonrobust.shape)
        print(f'diff: {diff}, thres: {thres}')
        assert robust.shape == nonrobust.shape
        assert diff < thres
