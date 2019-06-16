import numpy as np
import torch
import numpy_groupies as npg
from matplotlib import pyplot as plt

from torch.distributions import MultivariateNormal, Uniform
from hyperspherical_vae.distributions import von_mises_fisher as vmf

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

#%% Types
def ____TYPE____():
    pass

def float(v):
    return v.type(torch.get_default_dtype())

def numpy(v):
    """
    :type v: torch.Tensor
    :rtype: np.ndarray
    """
    return v.detach().numpy()
npy = numpy

def npys(*args):
    return tuple([npy(v) for v in args])

nanint = torch.tensor(np.nan).long()

def isnan(v):
    if v.dtype is torch.long:
        return v == nanint
    else:
        return torch.isnan(v)

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
    return repeat_all(*args, shape=shape, use_expand=True)

def repeat_to_shape(arg, shape):
    """
    :type arg: torch.Tensor
    :param shape: desired shape of the output
    :rtype: torch.Tensor
    """
    return repeat_all(arg, shape=shape)[0]

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
        if dim > 0:
            ndim_expand = dim
        else:
            ndim_expand = max_ndim + dim
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

#%% Permute
def ____PERMUTE____():
    pass

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
p2st = permute2st

def permute2en(v, ndim_st=1):
    """
    Permute last ndim_en of tensor v to the first
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

#%% Aggregate
def ____AGGREGATE____():
    pass

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
def ____STATS____():
    pass

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

#%% Distributions/Sampling
def ____DISTRIBUTIONS_SAMPLING____():
    pass

def rand(shape, low=0, high=1):
    d = Uniform(low=low, high=high)
    return d.rsample(shape)

def mvnrnd(mu, sigma, sample_shape=torch.Size([])):
    d = MultivariateNormal(loc=mu, covariance_matrix=sigma)
    return d.rsample(sample_shape)

def vmpdf(x, mu, scale, normalize=True):
    vm = vmf.VonMisesFisher(mu, scale + torch.zeros([1,1]))
    p = torch.exp(vm.log_prob(x))
    if normalize:
        p = sumto1(p)
    return p

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

def get_jacobian(net, x, noutputs):
    """
    From https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
    :type net: torch.nn.Module
    :type x: torch.Tensor
    :type noutputs: int
    :rtype: torch.Tensor
    """
    x = x.squeeze()
    n = x.size()[0]
    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.eye(noutputs))
    return x.grad.data

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
        return torch.ones(n_tr, dtype=torch.uint8)
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

