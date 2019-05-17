import numpy as np
import torch
import numpy_groupies as npg

#%% Wrapper that allows numpy-style syntax for torch
def kw_np2torch(kw):
    keys = kw.keys()
    for subs in [('axis', 'dim'),
                 ('keepdims', 'keepdim')]:
        # if subs[0] in keys:
        try:
            kw[subs[1]] = kw.pop(subs[0])
        except:
            pass
    return kw

class WrapTorch(object):
    """Uses torch as the backend; allows numpy-style sytax"""

    backend = torch

    arange = torch.arange
    array = torch.tensor
    cat = torch.cat
    exp = torch.exp
    log = torch.log
    newaxis = None

    def sumto1(self, v, dim=None, axis=None):
        if axis is not None:
            dim = axis
        if dim is None:
            return v / torch.sum(v)
        else:
            return v / torch.sum(v, dim, keepdim=True)

    def zeros(self, *args, **kwargs):
        if type(args[0]) is list:
            args = args[0]
        return torch.zeros(args, **kwargs)

    def ones(self, *args, **kwargs):
        if type(args[0]) is list:
            args = args[0]
        return torch.ones(args, **kwargs)

    def sum(self, v, *args, **kwargs):
        return torch.sum(v, *args, **kw_np2torch(kwargs))

    def max(self, v, *args, **kwargs):
        return torch.max(v, *args, **kw_np2torch(kwargs))

    def argmax(self, v, *args, **kwargs):
        return torch.argmax(v, *args, **kw_np2torch(kwargs))

    def min(self, v, *args, **kwargs):
        return torch.min(v, *args, **kw_np2torch(kwargs))

    def argmin(self, v, *args, **kwargs):
        return torch.argmin(v, *args, **kw_np2torch(kwargs))

    def abs(self, *args, **kwargs):
        return torch.abs(*args, **kwargs)

#%%
npt_torch = WrapTorch()
npt_numpy = np # Perhaps not fine if torch syntax is used

#%% Utility functions specifically for PyTorch
def enforce_tensor(v, min_ndim=1):
    if not torch.is_tensor(v):
        v = torch.tensor(v)
    if v.ndimension() < min_ndim:
        v = v.expand(v.shape + torch.Size([1] * (min_ndim - v.ndimension())))
    return v

def block_diag(matrices):
    ns = torch.LongTensor([m.shape[-1] for m in matrices])
    n = torch.sum(ns)
    batch_shape = matrices[0].shape[:-2]
    ndim_batch = len(batch_shape)
    # v = torch.zeros(list(matrices[0].shape[:-2]) + [n, n])
    cn0 = 0
    vs = []
    for n1, m1 in zip(ns, matrices):
        vs.append(torch.cat((
            torch.zeros(batch_shape + torch.Size([n1, cn0])),
            m1,
            torch.zeros(batch_shape + torch.Size([n1, n - cn0 - n1]))
        ), dim=ndim_batch + 1))
        # v[cn0:(cn0 + n1), cn0:(cn0 + n1)] = m1
        cn0 += n1
    v = torch.cat(vs, dim=ndim_batch)
    return v

#%% Shortcuts for torch
def float(v):
    return v.type(torch.get_default_dtype())

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))

def append_dim(v, n_dim_to_append=1):
    return attach_dim(v, n_dim_to_append=n_dim_to_append)

def prepend_dim(v, n_dim_to_prepend=1):
    return attach_dim(v, n_dim_to_prepend=n_dim_to_prepend)

def append_to_ndim(v, n_dim_desired):
    return attach_dim(v, n_dim_to_append=n_dim_desired - v.dim())

def prepend_to_ndim(v, n_dim_desired):
    return attach_dim(v, n_dim_to_prepend=n_dim_desired - v.dim())

def vec_on_dim(v, dim, ndim):
    shape = [1] * ndim
    shape[dim] = -1
    return v.view(shape)

def repeat_all(*args, shape=None):
    """
    Repeat tensors so that all tensors are of the same size.
    Tensors must have the same number of dimensions;
    otherwise, use repeat_batch() to prepend dimensions.
    :param shape: desired shape of the output. Give None to match max shape
    of each dim. Give -1 at dims where the max shape is desired.
    """

    ndim = args[0].ndimension()
    max_shape = torch.ones(ndim, dtype=torch.long)
    for arg in args:
        max_shape, _ = torch.max(torch.cat([
            torch.tensor(arg.shape)[None, :], max_shape[None, :]],
            dim=0), dim=0)
    if shape is None:
        shape = max_shape
    else:
        shape = torch.tensor(shape)
        is_free = shape == -1
        shape[is_free] = max_shape[is_free]

    out = []
    for arg in args:
        out.append(arg.repeat(
            *tuple((shape / torch.tensor(arg.shape)).long())))

    return tuple(out)

def repeat_to_shape(arg, shape):
    """
    :type arg: torch.Tensor
    :param shape: desired shape of the output
    :rtype: torch.Tensor
    """
    return repeat_all(arg, shape=shape)[0]

def repeat_batch(*args,
                 repeat_existing_dims=False, to_append_dims=False,
                 shape=None):
    """
    Repeat first dimensions, while keeping last dimensions the same
    :type args: (*torch.Tensor)
    :param args: tuple of tensors to repeat.
    :param repeat_existing_dims: whether to repeat singleton dims.
    :param to_append_dims: if True, append dims if needed; if False, prepend.
    :param shape: desired shape of the output. Give None to match max shape
    of each dim. Give -1 at dims where the max shape is desired.
    :return:
    """

    ndims = [arg.ndimension() for arg in args]
    max_ndim = np.amax(ndims)

    out = []
    for (ndim, arg) in zip(ndims, args):
        if to_append_dims:
            out.append(attach_dim(arg, 0, max_ndim - ndim))
        else:
            out.append(attach_dim(arg, max_ndim - ndim, 0))

    if repeat_existing_dims:
        return repeat_all(*tuple(out), shape=shape)
    else:
        return tuple(out)

def sumto1(v, dim=None, axis=None):
    """
    Make v sum to 1 across dim, i.e., make dim conditioned on the rest.
    dim can be a tuple.
    :param v: tensor.
    :param dim: dimensions to be conditioned upon the rest.
    :param axis: if given, overrides dim.
    :return: tensor of the same shape as v.
    """
    if axis is not None:
        dim = axis
    if dim is None:
        return v / torch.sum(v)
    else:
        return v / torch.sum(v, dim, keepdim=True)

def numpy(v):
    return v.detach().numpy()

npy = numpy

def npys(*args):
    return tuple([npy(v) for v in args])

def isnan(v):
    if v.dtype is torch.long:
        return torch.tensor(np.nan).long() == v
    else:
        return torch.isnan(v)

#%% Permute
def t(tensor):
    nd = tensor.ndimension()
    return tensor.permute(list(range(nd - 2)) + [nd - 1, nd - 2])

def permute2st(v, ndim_en=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_en: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(-ndim_en, 0)] + [*range(nd - ndim_en)])

def permute2en(v, ndim_st=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_st: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(ndim_st, nd)] + [*range(ndim_st)])

#%% Aggregate
def aggregate(subs, val=1., *args, **kwargs):
    """
    :param subs: [dim, element]
    :type subs: torch.LongTensor, (*torch.LongTensor)
    :type size: torch.LongTensor
    """

    if type(subs) is tuple or type(subs) is list:
        subs = np.concatenate(npys(*(sub.reshape(1,-1) for sub in subs)), 0)
    elif torch.is_tensor(subs):
        subs = npy(subs)
    return torch.tensor(npg.aggregate(subs, val, *args, **kwargs))

    # if size is None:
    #     size = torch.max(subs, 1)
    # elif not torch.is_tensor(size):
    #     size = torch.tensor(size)
    # #%%
    # cumsize = torch.cumprod(torch.cat((torch.tensor([1]), size.flip(0)),
    #                                   0)).flip(0)
    # #%%
    # ind = subs * cumsize[:,None]
    #
    # raise NotImplementedError(
    #     'Not finished implementation! Use npg.aggregate meanwhile!')

#%% Stats
def entropy(tensor, *args, **kwargs):
    """
    :type tensor: torch.Tensor
    :param tensor: probability. Optionally provide dim and keepdim for
    summation.
    :return: torch.Tensor
    """
    out = torch.log2(tensor) * tensor
    out[tensor == 0] = 0.
    return out.sum(*args, **kwargs)

