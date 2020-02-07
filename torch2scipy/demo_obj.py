import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from scipy import optimize

from lib.pylabyk.torch2scipy.obj import PyTorchObjective

from tqdm import tqdm

"""
From https://gist.github.com/gngdb/a9f912df362a85b37c730154ef3c294b
"""

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
        Y = linear(X)  # + eps


    # make module executing the experiment
    class Objective(nn.Module):
        def __init__(self):
            super(Objective, self).__init__()
            self.linear = nn.Linear(32, 32)
            self.linear = self.linear.train()
            self.X, self.Y = X, Y

        def forward(self):
            output = self.linear(self.X)
            return F.mse_loss(output, self.Y).mean()


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
        # xL = optimize.minimize(obj.fun, obj.x0, method='CG', jac=obj.jac)# , options={'gtol': 1e-2})