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

### Shape manipulation
`attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0)`: prepend/append size-1 dim.  
`vec2matmul(vec)`: adds a dim at the end. Equivalent to unsqueeze(-1).  
`matmul2vec(mat)`: removes a dim at the end. Equivalent to squeeze(-1). 