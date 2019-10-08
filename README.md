# pylabyk
Pylab-style utilities

This library is meant for my personal use, but everyone is welcome to use it and contribute to it.

# Highlights
## numpytorch.py
### Matrix operations  
`kron(a,b)`: [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product)    
`block_diag(m)`: [Block diagonal](https://en.wikipedia.org/wiki/Block_matrix)  
### NaN-related
See also
 [discussion on PyTorch repo](https://github.com/pytorch/pytorch/issues/21987]) 
 
`nanmean(v, dim=None, keepdims=False)`: similar to [`np.nanmean`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmean.html)  
`nansum(v, dim=None, keepdims=False)`: similar to [`np.nansum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmean.html)

Example:  
```python
import torch  
import numpytorch as npt
import numpy as np

v = torch.tensor([[1., 2., np.nan, 3.],[10., 30., np.nan, np.nan]])
npt.nanmean(v)
# Out: tensor(9.2000)

np.mean([1, 2, 3, 10, 30])
# Out: 9.2

npt.nanmean(v, dim=0)
# Out: tensor([ 5.5000, 16.0000,     nan,  3.0000])

npt.nanmean(v, dim=1)
# Out: tensor([ 2., 20.])

npt.nanmean(v, dim=1, keepdim=True)
# Out: tensor([[ 2.],
#              [20.]])
```

### Shape manipulation
`attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0)`: prepend/append size-1 dim.  
`vec2matmul(vec)`: adds a dim at the end. Equivalent to unsqueeze(-1).  
`matmul2vec(mat)`: removes a dim at the end. Equivalent to squeeze(-1). 