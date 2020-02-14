"""
Similar to demo_obj but with ykt.BoundedModule and softmax
For gradient-less methods, see:
http://scipy-lectures.org/advanced/mathematical_optimization/#gradient-less-methods
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import distributions as distrib

import numpy as np
from scipy import optimize

from lib.pylabyk.torch2scipy.obj import PyTorchObjective

from tqdm import tqdm

if __name__ == '__main__':
    # whatever this initialises to is our "true" W
    linear = nn.Linear(32, 32)
    linear = linear.eval()

    # input X
    N = 10000
    X = torch.Tensor(N, 32)
    X.uniform_(0., 1.)  # fill with uniform
    eps = torch.Tensor(N, 32)
    eps.normal_(0., 1e-4)

    # output Y
    with torch.no_grad():
        # Y = linear(X)  # + eps
        Y = distrib.Categorical(
            logits=linear(X)
        ).sample()

    # make module executing the experiment
    class Objective(nn.Module):
        def __init__(self):
            super(Objective, self).__init__()
            self.linear = nn.Linear(32, 32)
            self.linear = self.linear.train()
            self.X, self.Y = X, Y

        def pred(self):
            output = self.linear(self.X)


        def forward(self):
            output = self.linear(self.X)
            return F.cross_entropy(output, self.Y)
            # return F.mse_loss(output, self.Y).mean()


    objective = Objective()

    maxiter = 100
    with tqdm(total=maxiter) as pbar:
        def verbose(xk):
            pbar.update(1)


        # try to optimize that function with scipy
        obj = PyTorchObjective(objective)
        xL = optimize.minimize(obj.fun, obj.x0, method='BFGS', jac=obj.jac,
                               callback=verbose,
                               options={'gtol': 1e-6, 'disp': True,
                                        'maxiter': maxiter})
        # xL = optimize.minimize(obj.f

    xL.x