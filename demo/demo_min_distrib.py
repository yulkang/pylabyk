#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import torch
import matplotlib.pyplot as plt
from lib.pylabyk import numpytorch as npt
from lib.pylabyk.numpytorch import npy, npys


if __name__ == '__main__':
    p = torch.tensor([
        [0., 1.],
        [0.5, 0.5]
    ]) * 1.
    p_min, p_1st = npt.min_distrib(p)
    print((p_min, p_1st))
    print((p_min.sum(), p_1st.sum()))

    p = torch.tensor([
        [1., 0.],
        [0.5, 0.5]
    ]) * 1.
    p_min, p_1st = npt.min_distrib(p)
    print((p_min, p_1st))
    print((p_min.sum(), p_1st.sum()))

    p = torch.tensor([
        [0.5, 0.5],
        [0.5, 0.5]
    ]) * 0.1
    p_min, p_1st = npt.min_distrib(p)
    print((p_min, p_1st))
    print((p_min.sum(), p_1st.sum()))
