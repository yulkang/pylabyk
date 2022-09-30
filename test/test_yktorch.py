#  Copyright (c) 2022  Yul HR Kang. hk2699 at caa dot columbia dot edu.

import pytest

import numpy as np
import torch
import pylabyk.yktorch as ykt

EPSILON = 1e-6

class TestBoundedParameter:
    def check_given_lb_ub(self, lb, ub, v0=None):
        if v0 is None:
            v0 = torch.rand_like(lb)
        model = ykt.BoundedParameter(v0, lb, ub)
        for v00, v1 in zip(v0.flatten(), model.v[:].flatten()):
            assert ((v00 - v1).abs() < EPSILON).all(), f'{v00} != {v1}'

    def test_unbounded(self, shape=(10,)):
        self.check_given_lb_ub(
            torch.zeros(shape) - np.inf,
            torch.ones(shape) + np.inf
        )

    def test_lb_only(self, shape=(10,)):
        self.check_given_lb_ub(
            torch.zeros(shape),
            torch.ones(shape) + np.inf
        )

    def test_ub_only(self, shape=(10,)):
        self.check_given_lb_ub(
            torch.zeros(shape) - np.inf,
            torch.ones(shape)
        )

    def test_free_lb_ub(self, shape=(10,)):
        self.check_given_lb_ub(
            torch.zeros(shape),
            torch.ones(shape)
        )

    def test_fixed(self, shape=(10,)):
        v0 = torch.rand(shape)
        self.check_given_lb_ub(
            v0, v0, v0
        )


if __name__ == '__main__':
    tests = TestBoundedParameter()
    methods = [
        getattr(tests, v) for v in dir(tests) if v.startswith('test')]

    for method in methods:
        method()

    print('--')
