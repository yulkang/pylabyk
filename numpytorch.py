#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F
import numpy_groupies as npg
from matplotlib import pyplot as plt
from typing import Union, Iterable, Tuple, Dict

from torch.distributions import MultivariateNormal, Uniform, Normal, \
    Categorical, OneHotCategorical

#%% Wrapper that allows numpy-style syntax for torch
def ____NUMPY_COMPATIBILITY____():
    pass

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

#%% Constants
nan = torch.tensor(np.nan)
pi = torch.tensor(np.pi)
pi2 = torch.tensor(np.pi * 2)

#%% Utility functions specifically for PyTorch
def ____GRADIENT____():
    pass

def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

#%% Types
def ____TYPE____():
    pass


def tensor(v: Union[float, np.ndarray, torch.Tensor],
           min_ndim=1,
           **kwargs):
    """
    Construct a tensor if the input is not; otherwise return the input as is.
    Same as enforce_tensor
    :param v:
    :param min_ndim:
    :param kwargs:
    :return:
    """
    if not torch.is_tensor(v):
        v = torch.tensor(v, **kwargs)
    if v.ndimension() < min_ndim:
        v = v.expand(v.shape + torch.Size([1] * (min_ndim - v.ndimension())))
    return v


enforce_tensor = tensor


def float(v):
    return v.type(torch.get_default_dtype())

def numpy(v: Union[torch.Tensor, np.ndarray]):
    """
    Construct a np.ndarray from tensor; otherwise return the input as is
    :type v: torch.Tensor
    :rtype: np.ndarray
    """
    try:
        return v.clone().detach().numpy()
    except AttributeError:
        try:
            return v.clone().detach().cpu().numpy()
        except AttributeError:
            assert isinstance(v, np.ndarray) or sparse.isspmatrix(v) \
                   or np.isscalar(v)
            return v
            # except AssertionError:
            #     return np.array(v)


npy = numpy


def npys(*args):
    return tuple([npy(v) for v in args])


#%% NaN-related
def ____NAN____():
    pass

nanint = torch.tensor(np.nan).long()
def isnan(v):
    if v.dtype is torch.long:
        return v == nanint
    else:
        return torch.isnan(v)


def nan2v(v, fill=0):
    v[isnan(v)] = fill
    return v


def nansum(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs)

def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / float(~is_nan).sum(*args, **kwargs)


def softmax_mask(w: torch.Tensor,
                 dim=-1,
                 mask: torch.BoolTensor = None
                 ) -> torch.Tensor:
    """
    Allows having -np.inf in w to mask out, or give explicit bool mask
    :param w:
    :param dim:
    :param mask:
    :return:
    """
    if mask is None:
        mask = w != -np.inf
    minval = torch.min(w[~mask])  # to avoid affecting torch.max

    w1 = w.clone()
    w1[~mask] = minval

    # to prevent over/underflow
    w1 = w1 - torch.max(w1, dim=dim, keepdim=True)[0]

    w1 = torch.exp(w1)
    p = w1 / torch.sum(w1 * mask.float(), dim=dim, keepdim=True)
    p[~mask] = 0.
    return p


#%% Shape manipulation
def ____SHAPE____():
    pass

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
vec_on = vec_on_dim

def repeat_all(*args, shape=None, use_expand=False):
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
        if use_expand:
            out.append(arg.expand(
                *tuple(shape)))
                # *tuple((shape / torch.tensor(arg.shape)).long())))
        else:
            out.append(arg.repeat(
                *tuple((shape / torch.tensor(arg.shape)).long())))

    return tuple(out)

def expand_all(*args, shape=None):
    """
    Expand tensors so that all tensors are of the same size.
    Tensors must have the same number of dimensions;
    otherwise, use expand_batch() to prepend dimensions.
    :param args:
    :param shape:
    :return:
    """
    return repeat_all(*args, shape=shape, use_expand=True)

def repeat_to_shape(arg, shape):
    """
    :type arg: torch.Tensor
    :param shape: desired shape of the output
    :rtype: torch.Tensor
    """
    return repeat_all(arg, shape=shape)[0]

def max_shape(shapes):
    return torch.Size(
        torch.max(
            torch.stack([torch.tensor(v) for v in shapes]),
            dim=0
        )[0]
    )

def repeat_dim(tensor, repeat, dim):
    """
    :type tensor: torch.Tensor
    :type repeat: int
    :type dim: int
    """
    rep = torch.ones(tensor.dim(), dtype=torch.long)
    rep[dim] = torch.tensor(repeat)
    return tensor.repeat(torch.Size(rep))

def repeat_batch(*args,
                 repeat_existing_dims=False, to_append_dims=False,
                 shape=None,
                 use_expand=False):
    """
    Repeat first dimensions, while keeping last dimensions the same.
    :param args: tuple of tensors to repeat.
    :param repeat_existing_dims: whether to repeat singleton dims.
    :param to_append_dims: if True, append dims if needed; if False, prepend.
    :param shape: desired shape of the output. Give None to match max shape
    of each dim. Give -1 at dims where the max shape is desired.
    :param use_expand: True to use torch.expand instead of torch.repeat,
    to share the same memory across repeats.
    :return: tuple of repeated tensors.
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
        return repeat_all(*tuple(out), shape=shape, use_expand=use_expand)
    else:
        return tuple(out)

def expand_batch(*args, **kwargs):
    """
    Same as repeat_batch except forcing use_expand=True, to share memory
    across repeats, i.e., expand first dimensions, while keeping last
    dimensions the same
    :param args: tuple of tensors to repeat.
    :param repeat_existing_dims: whether to repeat singleton dims.
    :param to_append_dims: if True, append dims if needed; if False, prepend.
    :param shape: desired shape of the output. Give None to match max shape
    of each dim. Give -1 at dims where the max shape is desired.
    :return: tuple of repeated tensors.
    """
    return repeat_batch(*args, use_expand=True, **kwargs)

def expand_upto_dim(args, dim, to_expand_left=True):
    """
    Similar to expand_batch(), but keeps some dims unexpanded even if they
    don't match.
    :param args: iterable yielding torch.Tensor
    :param dim: if to_expand_left=True, then arg[:dim] is expanded,
        otherwise, arg[dim:] is expanded, for each arg in args.
        Note that dim=-1 leaves the last dim unexpanded.
        This is necessary to make dim=0 expand the first.
    :param to_expand_left: if True, left of dim is expanded while the rest of
    the dims are kept unchanged.
    :return: tuple of expanded args
    """
    ndims = [arg.ndimension() for arg in args]
    max_ndim = np.amax(ndims)

    out1 = []
    for (ndim, arg) in zip(ndims, args):
        if to_expand_left:
            # prepend dims
            out1.append(attach_dim(arg, max_ndim - ndim, 0))
        else:
            # append dims
            out1.append(attach_dim(arg, 0, max_ndim - ndim))

    if to_expand_left:
        if dim >= 0:
            ndim_expand = dim
        else:
            ndim_expand = max_ndim + dim
        if ndim_expand <= 0:
            # Nothing to expand - return
            return tuple(args)

        max_shape = torch.zeros(ndim_expand, dtype=torch.long)
        for o1 in out1:
            max_shape, _ = torch.max(torch.cat([
                max_shape[None,:],
                torch.tensor(o1.shape[:dim])[None,:]
            ], dim=0), dim=0)
        out2 = []
        ndim_kept = len(out1[0].shape[dim:])
        for o1 in out1:
            out2.append(o1.repeat([
                int(a) for a in torch.cat([
                    max_shape / torch.tensor(o1.shape[:dim],
                                              dtype=torch.long),
                    torch.ones(ndim_kept, dtype=torch.long)
                ], 0)
            ]))
    else:
        raise NotImplementedError(
            'to_expand_left=False not implemented/tested yet!')
        # if dim > 0:
        #     ndim_expand = max_ndim - dim
        # else:
        #     ndim_expand = -dim
        # max_shape = torch.zeros(ndim_expand)
        # for o1 in out1:
        #     max_shape = torch.max(torch.cat([
        #         max_shape[None,:],
        #         torch.tensor(arg.shape[dim:])[None,:]
        #     ], dim=0), dim=0)
        # out2 = []
        # ndim_kept = len(out1[0].shape[dim:])
        # for arg in args:
        #     out2.append(arg.repeat(
        #         [1] * ndim_kept
        #         + list(max_shape / torch.tensor(arg.shape[:dim]))))
    return tuple(out2)

#%% Permute
def ____PERMUTE____():
    pass

def t(tensor):
    nd = tensor.dim()
    return tensor.permute(list(range(nd - 2)) + [nd - 1, nd - 2])

def permute2st(v, ndim_en=1):
    """
    Permute last ndim_en of an array v to the first
    :type v: torch.Tensor
    :type ndim_en: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(-ndim_en, 0)] + [*range(nd - ndim_en)])
p2st = permute2st


def permute2en(v, ndim_st=1):
    """
    Permute first ndim_en of an array v to the last
    :type v: torch.Tensor
    :type ndim_st: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(ndim_st, nd)] + [*range(ndim_st)])
p2en = permute2en


#%% Indices
def ____INDICES____():
    pass

def unravel_index(v, shape, **kwargs):
    """
    For now, just use np.unravel_index()
    :type v: torch.LongTensor
    :type shape: torch.Size, tuple, list
    :type kwargs: dict
    :return: torch.LongTensor
    """
    return torch.tensor(np.unravel_index(v, shape, **kwargs))

def ravel_multi_index(v, shape, **kwargs):
    """
    For now, just use np.ravel_multi_index()
    :type v: torch.LongTensor
    :type shape: torch.Size, tuple, list
    :type kwargs: dict
    :return: torch.LongTensor
    """
    return torch.tensor(np.ravel_multi_index(v, shape, **kwargs))

#%% Algebra
def sumto1(v, dim=None, axis=None, keepdim=True):
    """
    Make v sum to 1 across dim, i.e., make dim conditioned on the rest.
    dim can be a tuple.

    :param v: tensor.
    :param dim: dimensions to be conditioned upon the rest.
    :param axis: if given, overrides dim.
    :return: tensor of the same shape as v.
    :rtype: torch.Tensor
    """
    if axis is not None:
        dim = axis
    if dim is None:
        return v / torch.sum(v)
    else:
        return v / torch.sum(v, dim, keepdim=keepdim)

#%% Aggregate
def ____AGGREGATE____():
    pass

def scatter_add(subs, val, dim=0, shape=None):
    """
    @param subs: ndim x n indices, suitable for np.ravel_multi_index
    @type subs: Union[np.ndarray, torch.LongTensor])
    @param val: n x ... values to add
    @type val: torch.Tensor
    @return:
    """
    if shape is None:
        shape = [(np.amax(sub) + 1) for sub in subs]
    idx = torch.LongTensor(np.ravel_multi_index(subs, shape))
    return torch.zeros(np.prod(shape).astype(np.long)).scatter_add(
        dim=dim, index=idx, src=val
    ).reshape(shape)

def aggregate(subs, val=1., *args, **kwargs):
    """
    :param subs: [dim, element]
    :type subs: torch.LongTensor, (*torch.LongTensor)
    :type size: torch.LongTensor
    """

    if type(subs) is tuple or type(subs) is list:
        subs = np.stack(subs)
        # subs = np.concatenate(npys(*(sub.reshape(1,-1) for sub in subs)), 0)
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
def ____STATS____():
    pass


def logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p) - torch.log(torch.tensor(1.) - p)


def logistic(x: torch.Tensor) -> torch.Tensor:
    p = torch.tensor(1.) / (torch.tensor(1.) + torch.exp(-x))
    # p_nan = isnan(p)
    # if p_nan.any():
    #     p[(x > 0) & p_nan] = 1.
    #     p[(x < 0) & p_nan] = 0.
    return p


def conv_t(p, kernel, **kwargs):
    """
    1D convolution with the starting time of the signal and kernel anchored.

    EXAMPLE:
    p_cond_rt = npt.conv_t(
        p_cond_td[None],  # [1, cond, fr]
        p_tnd[None, None, :].expand([n_cond, 1, nt]), # [cond, 1, fr]
        groups=n_cond
    )
    :param p: [batch, time] or [batch, channel_in, time]
    :param kernel: [time] or [channel_out, channel_in, time]
    :param kwargs: fed to F.conv1d
    :return: p[batch, time] or [batch, channel_out, time]
    """
    nt = p.shape[-1]
    if p.ndim == 1:
        p = p[None, None, :]
    elif p.ndim == 2:  # [batch, time]
        p = p[:, None, :]
    else:
        assert p.ndim == 3

    if kernel.ndim == 1:
        return F.conv1d(
            p,
            kernel.flip(-1)[None, None, :],
            padding=kernel.shape[-1],
            **kwargs
        ).squeeze(0).squeeze(0)[:nt]
    else:
        return F.conv1d(
            p,
            kernel.flip(-1),
            padding=kernel.shape[-1],
            **kwargs
        )[:, :, :nt].squeeze(0)


def shiftdim(v: torch.Tensor, shift: torch.Tensor, dim=0, pad='repeat'):
    if torch.is_floating_point(shift):
        lb = shift.floor().long()
        ub = lb + 1
        p = torch.tensor(1.) - torch.cat([shift.reshape([1]) - lb,
                                          ub - shift.reshape([1])], 0)
        return (
                shiftdim(v, shift=lb, dim=dim) * p[0]
                + shiftdim(v, shift=ub, dim=dim) * p[1]
        )

    if dim != 0:
        v = v.transpose(0, dim)

    if shift == 0:
        pass
    else:
        if shift > 0:
            if pad == 'repeat':
                vpad = v[0].expand((shift,) + v.shape[1:])
            else:
                vpad = torch.zeros((shift,) + v.shape[1:])

            v = torch.cat([
                vpad,
                v[:-shift],
            ])
        else:
            if pad == 'repeat':
                vpad = v[-1].expand((-shift,) + v.shape[1:])
            else:
                vpad = torch.zeros((-shift,) + v.shape[1:])

            v = torch.cat([
                v[-shift:],
                vpad,
            ])

    if dim != 0:
        v = v.transpose(0, dim)
    return v


def interp1d(query: torch.Tensor, value: torch.Tensor, dim=0) -> torch.Tensor:
    """

    :param query: index on dim. Should be a FloatTensor for gradient.
    :param value:
    :param dim:
    :return: interpolated to give value[query] (when dim=0)
    """
    v = value
    if dim != 0:
        v = v.transpose(0, dim)
    else:
        v = v

    if torch.is_floating_point(query):
        q0 = query.floor().long()
        q1 = q0 + 1
        p = query - q0
        v = (
            interp1d(q0, v, 0) * (torch.tensor(1.) - p.expand_as(v))
            + interp1d(q1, v, 0) * p.expand_as(v)
        )
    else:
        v = v[query]

    if dim != 0:
        v = v.transpose(0, dim)
    else:
        v = v
    return v


def mean_distrib(p, v, axis=None):
    if axis is None:
        kw = {}
    else:
        kw = {'axis': axis}
    return (p * v).sum(**kw) / p.sum(**kw)


def var_distrib(p, v, axis=None):
    return (
            mean_distrib(p, v ** 2, axis=axis)
            - mean_distrib(p, v, axis=axis) ** 2
    )


def std_distrib(p, v, axis=None):
    return var_distrib(p, v, axis=axis).sqrt()


def sem_distrib(p, v, axis=None, n=None):
    if n is None:
        if axis is None:
            n = p.sum()
        else:
            n = p.sum(axis)
    v = var_distrib(p, v, axis=axis)
    if torch.is_tensor(v):
        return (v / n).sqrt()
    else:
        return np.sqrt(v / n)


def min_distrib(p:torch.Tensor
                ) -> (torch.Tensor, torch.Tensor):
    """
    Distribution of the min of independent RVs R0 ~ p[0] and R1 ~ p[1].
    When ndims(p) > 2, each pair of p[0, r0, :] and p[1, r1, :] is processed
    separately. p.sum(1) is taken as the number of trials.

    p_min, p_1st = min_distrib(p)

    p_min(t,1,:): Probability distribution of min(t_1 ~ p(:,1), t_2 ~ p(:,2))
    p_1st(t,k,:): Probability of t_k happening first at t.
                  sums(p_1st, [1, 2]) gives all 1's.

    Formula from: http://math.stackexchange.com/questions/308230/expectation-of-the-min-of-two-independent-random-variables

    :param p: [id, value, [batch, ...]]
    :return: p_min[value, batch, ...], p_1st[id, value, batch, ...]
    """
    assert p.shape[0] == 2

    shape0 = p.shape
    p = p.reshape([shape0[0], shape0[1], -1])
    p0 = p.clone()

    # p[id, value, batch] = P(value | id, batch)
    p = sumto1(p, 1)

    # c[id, value, batch] = P(v <= value | id, batch)
    c = p.cumsum(1)

    # P(T <= either 0 or 1) = P(T <= 0 or 1) - P(T <= 0 and 1)
    # c_min[1, value, batch]
    c_min = c.sum(0, keepdim=True) - c.prod(0, keepdim=True)

    # P(T == either 0 or 1) = diff P(T <= either 0 or 1)
    # p_min[1, value, batch]
    p_min = F.pad(c_min, [0, 0, 1, 0])
    p_min = (p_min[:, 1:] - p_min[:, :-1]).clamp_min(0)

    # p_1st[id, value, batch] = P(v == v_min == v_id < v_other | batch)
    # = P(v_min == v | batch) * P(v_id == v,  v_min = v | batch)
    p_1st = p_min * nan2v(sumto1(p0, 0))

    # if indep == 'indep':
    #     # p_both[batch]
    #     p_both = sum_p[0] * sum_p[1]
    #
    #     # p_min[id, value, batch]
    #     p_min = nan2v(sumto1(p_min, 0).clone()) * sum_p
    # elif indep == 'disjoint':
    #     raise NotImplementedError()
    # else:
    #     raise ValueError()

    # Special case 1: No p_1st if neither p(:,1) nor p(:,2)
    any_p = (p != 0).any(0, keepdim=True)
    p_1st = p_1st * any_p

    # Special case 2: zeros across all values in either id
    # ch0[id, batch]
    ch0 = (p == 0).all(1).any(0)
    # ch00[batch]
    ch00 = ch0.any(0)
    p_min[0, :, ch00] = 0
    p_1st[:, :, ch00] = 0

    # # Keep the total count the product of the two sums
    # sum_p[]
    sum_p = p0.sum(1, keepdim=True).prod(0, keepdim=True)
    p_min = nan2v(sumto1(p_min, [0, 1])) * sum_p
    p_1st = nan2v(sumto1(p_1st, [0, 1])) * sum_p

    return p_min[0].reshape(shape0[1:]), p_1st.reshape(shape0)


def max_distrib(p: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Distribution of the max of independent RVs R0 ~ p[0] and R1 ~ p[1].
    When ndims(p) > 2, each pair of p[0, r0, :] and p[1, r1, :] is processed
    separately. p.sum(1) is taken as the number of trials.

    p_max, p_1st = min_distrib(p)

    p_max(t,1,:): Probability distribution of min(t_1 ~ p(:,1), t_2 ~ p(:,2))
    p_last(t,k,:): Probability of t_k happening first at t.
                  sums(p_1st, [1, 2]) gives all 1's.

    Formula from: http://math.stackexchange.com/questions/308230/expectation-of-the-min-of-two-independent-random-variables

    :param p: [id, value, [batch, ...]]
    :return: p_max[value, batch, ...], p_last[id, value, batch, ...]
    """
    p_min, p_1st = min_distrib(p.flip(1))
    return p_min.flip(0), p_1st.flip(1)


def sem(v, dim=0):
    return torch.std(v, dim=dim) / torch.sqrt(v.shape[dim])


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


def softmax_bias(p, slope, bias):
    """
    Symmetric softmax with bias. Only works for binary. Works elementwise.
    Cannot use too small or large bias (roughly < 1e-3 or > 1 - 1e-3)
    :param p: between 0 and 1.
    :param slope: arbitary real value. 1 gives identity mapping, 0 always 0.5.
    :param bias: between 1e-3 and 1 - 1e-3. Giving p=bias returns 0.5.
    :return: transformed probability.
    :type p: torch.FloatTensor
    :type slope: torch.FloatTensor
    :type bias: torch.FloatTensor
    :rtype: torch.FloatTensor
    """
    k = (1. - bias) ** slope
    k = k / (bias ** slope + k)
    q = k * p ** slope
    q = q / (q + (1. - k) * (1. - p) ** slope)
    return q

    # k = -torch.log(torch.tensor(2.)) / torch.log(torch.tensor(bias))
    # q = (p ** k ** slope)
    # return q / (q + (1. - p ** k) ** slope)


def test_softmax_bias():
    p = torch.linspace(1e-4, 1 - 1e-4, 100);
    q = softmax_bias(p, torch.tensor(1.), p)
    plt.subplot(2, 3, 1)
    plt.plot(*npys(p, q))
    plt.xlabel('bias \& p')

    plt.subplot(2, 3, 2)
    biases = torch.linspace(1e-6, 1 - 1e-6, 5)
    for bias in biases:
        q = softmax_bias(p, torch.tensor(1.), bias)
        plt.plot(*npys(p, q))
    plt.xticks(npy(biases))
    plt.yticks(npy(biases))
    plt.grid(True)
    plt.axis('square')

    for col, bias in enumerate(torch.tensor([0.25, 0.5, 0.75])):
        plt.subplot(2, 3, 4 + col)
        for slope in torch.tensor([0., 1., 2.]):
            q = softmax_bias(p, slope, bias)
            plt.plot(*npys(p, q))
        plt.xticks(npy(biases))
        plt.yticks(npy(biases))
        plt.grid(True)
        plt.axis('square')

    plt.show()
    print('--')


def ____DISTRIBUTIONS_SAMPLING____():
    pass


def lognormal_params2mean_stdev(loc, scale):
    return torch.exp(loc + scale ** 2 / 2.), \
           (torch.exp(scale ** 2) - 1.) * torch.exp(2 * loc + scale ** 2)


def inv_gaussian_pdf(x, mu, lam):
    """
    As in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    @param x: values to query. Must be positive.
    @param mu: the expectation
    @param lam: lambda in Wikipedia's notation
    @return: p(x; mu, lam)
    """
    p = torch.sqrt(
        lam / (2 * pi * x ** 3)
    ) * torch.exp(-lam * (x - mu) ** 2 / (2 * mu ** 2 * x))
    return p


def inv_gaussian_cdf(x, mu, lam):
    c0 = torch.distributions.Normal(loc=0., scale=1.).cdf(
        torch.sqrt(lam / x) * (x / mu - 1.)
    )
    c1 = torch.distributions.Normal(loc=0., scale=1.).cdf(
        -torch.sqrt(lam / x) * (x / mu + 1.)
    )

    if torch.any(torch.isnan(c0)):
        print('--')  # CHECKING
    if torch.any(torch.isnan(c1)):
        print('--')  # CHECKING

    c = c0 + torch.exp(2. * lam / mu) * c1
    if torch.any(torch.isnan(c)):
        print('--')  # CHECKING
    return c


def inv_gaussian_variance(mu, lam):
    """
    As in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    @param mu: the expectation
    @param lam: lambda in Wikipedia's notation
    @return: Var[X]
    """
    return mu ** 3 / lam


def inv_gaussian_variance2lam(mu, var):
    return 1 / var * mu ** 3


def inv_gaussian_mean_std2params(mu, std):
    """mu, std -> mu, lam"""
    return mu, inv_gaussian_variance2lam(mu, std ** 2)


def inv_gaussian_pmf_mean_stdev(
        x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, dx=None,
        algo='diff_cdf'
) -> torch.Tensor:
    """

    :param x: must be a 1-dim tensor along dim.
    :param mu:
    :param std:
    :param dx:
    :return:
    """
    if dx is None:
        dx = x[[1]] - x[[0]]

    # --- Most robust against NaN; least valid
    if algo == 'dx':
        x, mu, std = expand_all(x, mu, std)
        p = torch.zeros_like(x)
        incl = x > 0
        p[incl] = inv_gaussian_pdf(
            x[incl], mu[incl],
            inv_gaussian_variance2lam(mu[incl], std[incl] ** 2)
        )
        p[incl] = p[incl] * dx

    elif algo == 'diff_cdf':
        # --- Least robust against NaN; most valid
        x = torch.cat([x, x[[-1]] + dx], dim=0)
        x, mu, std = expand_all(x, mu, std)
        c = torch.zeros_like(x)
        incl = x > 0
        c[incl] = inv_gaussian_cdf(
            x[incl], mu[incl],
            inv_gaussian_variance2lam(mu[incl], std[incl] ** 2)
        )

        is_nan = torch.isnan(c)
        if torch.any(is_nan):
            t = ((x - mu) / std)
            print(t[is_nan])
            print(t[~is_nan])
            print('--')

        p = c[1:] - c[:-1]

    elif algo == 'norm_w_cdf':
        x, mu, std = expand_all(x, mu, std)
        p = torch.zeros_like(x)
        incl = x > 0
        p[incl] = inv_gaussian_pdf(
            x[incl], mu[incl],
            inv_gaussian_variance2lam(mu[incl], std[incl] ** 2)
        )
        c = inv_gaussian_cdf(
            x[[-1]] + dx, mu,
            inv_gaussian_variance2lam(mu, std ** 2)
        )
        p[incl] = p[incl] / c[incl]

    else:
        raise ValueError()

    # p[torch.isnan(p)] = 0.
    is_nan = torch.isnan(p)
    if torch.any(is_nan):
        print('--')
    return p


def lognorm_params_given_mean_stdev(mean: torch.Tensor, stdev: torch.Tensor
                                    ) -> (torch.Tensor, torch.Tensor):
    sigma = torch.sqrt(torch.log(1 + (stdev / mean) ** 2))
    mu = torch.log(mean) - sigma ** 2 / 2.
    return mu, sigma


def lognorm_pmf(x: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor,
                ) -> torch.Tensor:
    """

    :param x: must be monotonic increasing with equal increment on dim 0
    :param mean:
    :param stdev:
    :return: p[k] = P(x[k] < X < x[k + 1]; mean, stdev)
    """

    mu, sigma = lognorm_params_given_mean_stdev(mean, stdev)

    dx = x[[1]] - x[[0]]
    x = torch.cat([x, x[[-1]] + dx], dim=0)
    x, mu, sigma = expand_all(x, mu, sigma)
    c = torch.zeros_like(x)
    incl = x > 0
    c[incl] = torch.distributions.LogNormal(mu[incl], sigma[incl]).cdf(
        x[incl]
    )
    p = c[1:] - c[:-1]

    is_nan = torch.isnan(p)
    if torch.any(is_nan):
        print('--')
    return p


def delta(levels, v, dlevel=None):
    """

    @type levels: torch.Tensor
    @type v: torch.Tensor
    @type dlevel: torch.Tensor
    @rtype: torch.Tensor
    """
    if dlevel is None:
        dlevel = (levels[1] - levels[0]).unsqueeze(0)
    return 1. - ((levels - v) / dlevel).abs().clamp(0., 1.)


def rand(shape, low=0, high=1):
    d = Uniform(low=low, high=high)
    return d.rsample(shape)


def mvnrnd(mu, sigma, sample_shape=()):
    d = MultivariateNormal(loc=mu, covariance_matrix=sigma)
    return d.rsample(sample_shape)


def normrnd(mu=0., sigma=1., sample_shape=(), return_distrib=False):
    """

    @param mu:
    @param sigma:
    @param sample_shape:
    @type return_distrib: bool
    @rtype: Union[(torch.Tensor, torch.distributions.Distribution),
    torch.Tensor]
    """
    d = Normal(loc=tensor(mu), scale=tensor(sigma))
    s = d.rsample(sample_shape)
    if return_distrib:
        return s, d
    else:
        return s
    s = d.rsample(sample_shape)
    if return_distrib:
        return s, d
    else:
        return s


def log_normpdf(sample, mu=0., sigma=1.):
    return Normal(loc=mu, scale=sigma).log_prob(sample)


# def categrnd(probs):
#     return torch.multinomial(probs, 1)


def categrnd(probs=None, logits=None, sample_shape=()):
    return torch.distributions.Categorical(
        probs=probs, logits=logits
    ).sample(sample_shape=sample_shape)


def onehotrnd(probs=None, logits=None, sample_shape=()):
    return torch.distributions.OneHotCategorical(
        probs=probs, logits=logits
    ).sample(sample_shape=sample_shape)


def mvnpdf_log(x, mu=None, sigma=None):
    """
    :param x: [batch, ndim]
    :param mu: [batch, ndim]
    :param sigma: [batch, ndim, ndim]
    :return: log_prob [batch]
    :rtype: torch.FloatTensor
    """
    if mu is None:
        mu = torch.tensor([0.])
    if sigma is None:
        sigma = torch.eye(len(mu))
    d = MultivariateNormal(loc=mu,
                           covariance_matrix=sigma)
    return d.log_prob(x)


def bootstrap(fun, samp, n_boot=100):
    n_samp = len(samp)
    ix = torch.randint(n_samp, (n_boot, n_samp))
    res = []
    for i_boot in range(n_boot):
        samp1 = [samp[s] for s in ix[i_boot,:]]
        res.append(fun(samp1))
    return res, ix

#%% Linear algebra
def ____LINEAR_ALGEBRA____():
    pass


def vec2mat0(vec):
    """Vector dim comes first, unlike v2m"""
    return torch.unsqueeze(vec, 1)
v2m0 = vec2mat0


def mat2vec0(mat):
    """Matrix dims come first, unlike v2m"""
    return torch.squeeze(mat, 1)
m2v0 = mat2vec0


def matmul0(a, b):
    """Matrix dims come first, unlike torch.matmul"""
    return p2st(p2en(a, 2) @ p2en(b, 2), 2)
mm0 = matmul0


def matvecmul0(mat, vec):
    """Matrix and vec dims come first. Vec is expanded to mat first."""
    return m2v0(matmul0(mat, torch.unsqueeze(vec, 1)))
mvm0 = matvecmul0


def vec2matmul(vec):
    """
    :type vec: torch.Tensor
    :rtype: torch.Tensor
    """
    return vec.unsqueeze(-1)
v2m = vec2matmul

def matmul2vec(mm):
    """
    :type mm: torch.Tensor
    :rtype: torch.Tensor
    """
    return mm.squeeze(-1)
m2v = matmul2vec

def matsum(*tensors):
    """
    Apply expand_upto_dim(tensors, -2) before adding them together,
    consistent with torch.matmul()
    :param tensors: iterable of tensors
    :return: sum of tensors, expanded except for the last two dimensions.
    """
    tensors = expand_upto_dim(tensors, -2)
    res = 0.
    for tensor in tensors:
        res = res + tensor
    return res

def get_jacobian(net, x, noutputs):
    """
    From https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
    :type net: torch.nn.Module
    :type x: torch.Tensor
    :type noutputs: int
    :rtype: torch.Tensor
    """
    x = x.squeeze()
    # n = x.size()[0]
    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.eye(noutputs))
    return x.grad.data

def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

def test_kron():
    a = repeat_dim(torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ]).unsqueeze(0), 5, 0)
    b = torch.tensor([[1.,1.,0.],[0.,1.,1.]]).unsqueeze(0)
    res = kron(a, b)
    print(res)
    print(a.shape)
    print(b.shape)
    print(res.shape)

def block_diag_irregular(matrices):
    # Block diagonal from a list of matrices that have different shapes.
    # If they have identical shapes, use block_diag(), which is vectorized.

    matrices = [p2st(m, 2) for m in matrices]

    ns = torch.LongTensor([m.shape[0] for m in matrices])
    n = torch.sum(ns)
    batch_shape = matrices[0].shape[2:]

    v = torch.zeros(torch.Size([n, n]) + batch_shape)
    for ii, m1 in enumerate(matrices):
        st = torch.sum(ns[:ii])
        en = torch.sum(ns[:(ii + 1)])
        v[st:en, st:en] = m1
    return p2en(v, 2)

    # cn0 = 0
    # vs = []
    # for n1, m1 in zip(ns, matrices):
    #     vs.append(torch.cat((
    #         torch.zeros(batch_shape + torch.Size([n1, cn0])),
    #         m1,
    #         torch.zeros(batch_shape + torch.Size([n1, n - cn0 - n1]))
    #     ), dim=ndim_batch + 1))
    #     # v[cn0:(cn0 + n1), cn0:(cn0 + n1)] = m1
    #     cn0 += n1
    # v = torch.cat(vs, dim=ndim_batch)
    # return v

def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def unblock_diag(m, n=None, size_block=None):
    """
    The inverse of block_diag(). Not vectorized yet.
    :param m: block diagonal matrix
    :param n: int. Number of blocks
    :size_block: torch.Size. Size of a block.
    :return: tensor unblocked such that the last sizes are [n] + size_block
    """
    # not vectorized yet
    if size_block is None:
        size_block = torch.Size(torch.tensor(m.shape[-2:]) // n)
    elif n is None:
        n = m.shape[-2] // torch.tensor(size_block[0])
        assert n == m.shape[-1] // torch.tensor(size_block[1])
    else:
        raise ValueError('n or size_block must be given!')
    m = p2st(m, 2)

    res = torch.zeros(torch.Size([n]) + size_block + m.shape[2:])
    for i_block in range(n):
        st_row = size_block[0] * i_block
        en_row = size_block[0] * (i_block + 1)
        st_col = size_block[1] * i_block
        en_col = size_block[1] * (i_block + 1)
        res[i_block,:] = m[st_row:en_row, st_col:en_col, :]

    return p2en(res, 3)

#%% Cross-validation
def ____CROSS_VALIDATION____():
    pass

def crossvalincl(n_tr, i_fold, n_fold=10, mode='consec'):
    """
    :param n_tr: Number of trials
    :param i_fold: Index of fold
    :param n_fold: Number of folds. If 1, training set = test set.
    :param mode: 'consec': consecutive trials; 'mod': interleaved
    :return: boolean (Byte) tensor
    """
    if n_fold == 1:
        return torch.ones(n_tr, dtype=torch.bool)
    elif n_fold < 1:
        raise ValueError('n_fold must be >= 1')

    if mode == 'mod':
        return (torch.arange(n_tr) % n_fold) == i_fold
    elif mode == 'consec':
        ix = (torch.arange(n_tr, dtype=torch.double) / n_tr *
              n_fold).long()
        return ix == i_fold
    else:
        raise NotImplementedError('mode=%s is not implemented!' % mode)

def ____CIRCULAR_STATS____():
    pass

def circdiff(angle1, angle2, maxangle=pi2):
    """
    :param angle1: angle scaled to be between 0 and maxangle
    :param angle2: angle scaled to be between 0 and maxangle
    :param maxangle: max angle. defaults to 2 * pi.
    :return: angular difference, between -.5 and +.5 * maxangle
    """
    return (((angle1 / maxangle) - (angle2 / maxangle) + .5) % 1. - .5) * maxangle


def rad2deg(rad):
    return rad / pi * 180.


def deg2rad(deg):
    return deg / 180 * pi


def prad2unitvec(prad, dim=-1):
    rad = prad * 2. * np.pi
    return torch.stack([torch.cos(rad), torch.sin(rad)], dim=dim)


def pconc2conc(pconc):
    pconc = torch.clamp(pconc, min=1e-6, max=1-1e-6)
    return 1. / (1. - pconc) - 1.


def vmpdf_prad_pconc(prad, ploc, pconc, normalize=True):
    """
    :param prad: 0 to 1 maps to 0 to 2*pi radians
    :param pconc: 0 to 1 maps to 0 to inf concentration
    :rtype: torch.Tensor
    """
    return vmpdf(prad2unitvec(prad),
                 prad2unitvec(ploc),
                 pconc2conc(pconc),
                 normalize=normalize)


def vmpdf_a_given_b(a_prad, b_prad, pconc):
    """

    :param a_prad: between 0 and 1. Maps to 0 and 2*pi.
    :type a_prad: torch.Tensor
    :param b_prad: between 0 and 1. Maps to 0 and 2*pi.
    :type b_prad: torch.Tensor
    :param pconc: float
    :return: p_a_given_b[index_a, index_b]
    :rtype: torch.Tensor
    """

    dist = ((a_prad.reshape([-1, 1]) - b_prad.reshape([1, -1])) %
            1.).double()
    return sumto1(vmpdf_prad_pconc(
        dist.flatten(), torch.tensor([0.]),
        tensor(pconc)
    ).reshape([a_prad.numel(), b_prad.numel()]), 1)


def vmpdf(x, mu, scale=None, normalize=True):
    from .hyperspherical_vae.distributions import von_mises_fisher as vmf

    if scale is None:
        # raise NotImplementedError('Using gradient not tested yet! (Seems '
        #                           'to gives NaN gradient when scale = 0)')
        scale = torch.sqrt(torch.sum(mu ** 2, dim=1, keepdim=True))
        mu = mu / scale
        # mu[scale[:,0] == 0, :] = 0.

    vm = vmf.VonMisesFisher(mu, scale + torch.zeros([1,1]))
    p = torch.exp(vm.log_prob(x))
    # if scale == 0.:
    #     p = torch.ones_like(p) / p.shape[0]
    if normalize:
        p = sumto1(p)
    return p

