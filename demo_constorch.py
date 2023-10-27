#  Copyright (c) 2021  Yul HR Kang. hk2699 at caa dot columbia dot edu.

import torch
import numpy as np
from . import yktorch as ykt
from matplotlib import pyplot as plt


class DemoModule(ykt.BoundedModule):
    def __init__(self, param0):
        super().__init__()
        self.lbub = ykt.BoundedParameter(param0, lb=-1., ub=1.)
        self.lb = ykt.BoundedParameter(param0, lb=-1., ub=np.inf)
        self.ub = ykt.BoundedParameter(param0, lb=-np.inf, ub=1.)
        self.no_bound = ykt.BoundedParameter(param0, lb=-np.inf, ub=np.inf)

    def forward(self, target):
        """
        Note: currently need to use indexing, e.g., param[:] or param[1:],
        to invoke __getitem__() and __setitem__()
        """
        return (
                (self.lbub[:] - target) ** 2
                + (self.lb[:] - target) ** 2
                + (self.ub[:] - target) ** 2
                + (self.no_bound[:] - target) ** 2
        ).sum()


# Usage is almost the same as torch.nn.Module,
# except that currently a parameter's value needs to be
# referred to as, e.g., model.param[:],
# to invoke __getitem__() and __setitem__().
model = DemoModule(torch.tensor([0.5, -0.5, 0.]))
print(model.lbub[:] + 2.) # works like regular parameters

loss = model(torch.tensor([-2., 2., 0.]))
model.zero_grad()
loss.backward()

# Loss and gradient vector, useful for optimization
# with scipy.optimize.minimize()
print(loss)
print(model.grad_vec())

# Bar plot of parameters with bounds, useful for monitoring training
model.plot_params()
plt.show()