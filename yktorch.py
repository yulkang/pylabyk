"""
Example:

"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.
import numpy as np
from pprint import pprint
from typing import Union, Iterable, List, Tuple, Sequence, Callable, \
    Dict
from collections import OrderedDict as odict, namedtuple
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
from copy import deepcopy
import warnings

import torch
from scipy import optimize as scioptim
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from pytorchobjective.obj_torch import PyTorchObjective

from . import numpytorch as npt, plt2
from .numpytorch import npy
from .cacheutil import mkdir4file

default_device = torch.device('cpu')  # CHECKING
# default_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%% Bounded parameters (under construction)
# this is for better autocompletion, etc.
# it should run backward() on the variables hidden in it.
# See torch.tensor.Tensor for more.

Param0 = namedtuple('Param0', ['v0', 'lb', 'ub'])

epsilon0 = 1e-6

class OverriddenParameter(nn.Module):
    """
    In operations requiring the whole parameter, use param[:] instead of
    param itself. For example, use result = param[:] + 1. Use this
    instead of param.v, because param[:] allows the code to
    work when param is substituted with another tensor.
    """

    def __init__(self, epsilon=epsilon0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.skip_loading_lbub = None
        self._shape = None

    @property
    def v(self):
        return self.param2data(self._param)

    @v.setter
    def v(self, data):
        self._param = nn.Parameter(self.data2param(data))

    @property
    def requires_grad(self):
        return self._param.requires_grad

    @property
    def shape(self):
        if self._shape is None:
            return self._param.shape
        else:
            return self._shape

    @shape.setter
    def shape(self, v):
        self._shape = v

    @requires_grad.setter
    def requires_grad(self, v):
        self._param.requires_grad = v

    def __getitem__(self, key):
        return self.v[key]

    def __setitem__(self, key, data):
        self._param[key] = self.data2param(data)

    def param2data(self, param):
        raise NotImplementedError()

    def data2param(self, data):
        raise NotImplementedError()

    def __str__(self):
        indent = '  '
        from textwrap import TextWrapper as TW
        def f(s, **kwargs):
            return TW(**kwargs).fill(s)
        return str((self.param2data(self._param), type(self).__name__))

    def get_n_free_param(self) -> int:
        """Number of free parameters"""
        if self._param.requires_grad:
            return torch.numel(self._param)
        else:
            return 0

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict.update({
            prefix + '_lb': self.lb,
            prefix + '_ub': self.ub,
            prefix + '_data': self.v,
            prefix + '_shape': self.shape,
        })
        return state_dict

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        lb_name = prefix + '_lb'
        ub_name = prefix + '_ub'
        param_name = prefix + '_param'
        data_name = prefix + '_data'
        shape_name = prefix + '_shape'
        if self.skip_loading_lbub:
            if lb_name in state_dict:
                state_dict.pop(lb_name)
            if ub_name in state_dict:
                state_dict.pop(ub_name)
        else:
            if lb_name in state_dict:
                self.lb = npt.tensor(state_dict.pop(lb_name))
            if ub_name in state_dict:
                self.ub = npt.tensor(state_dict.pop(ub_name))

        if data_name in state_dict:
            state_dict[param_name] = self.data2param(
                state_dict.pop(data_name).detach().clone().to(npt.get_device()))
        elif param_name in state_dict:
            state_dict[param_name] = npt.tensor(state_dict[param_name])

        if shape_name in state_dict:
            self.shape = state_dict.pop(shape_name)

        self.update_is_fixed()

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def update_is_fixed(self):
        pass  # only implemented in some of the subclasses


class BoundedParameter(OverriddenParameter):
    def __init__(self, data, lb=0., ub=1.,
                 lb_random=None, ub_random=None,
                 skip_loading_lbub=False,
                 requires_grad=True, randomize=False, # name=None,
                 **kwargs):
        """

        :param data:
        :param lb: None or -np.inf to skip
        :param ub: None or np.inf to skip
        :param skip_loading_lbub:
        :param requires_grad:
        :param randomize: True to randomize within lb and ub.
            Doesn't work when lb or ub is not given.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.lb = None if lb is None else enforce_float_tensor(lb)  # lb == -np.inf doesn't allow for non-scalar lb
        self.ub = None if ub is None else enforce_float_tensor(ub)

        self.lb_random = self.lb if lb_random is None else lb_random
        self.ub_random = self.ub if ub_random is None else ub_random

        # self.lb = None if lb == -np.inf or lb is None else npt.tensor(lb)
        # self.ub = None if ub == np.inf or ub is None else npt.tensor(ub)
        self.skip_loading_lbub = skip_loading_lbub
        if randomize:
            data = self.get_random_data0(data)
        else:
            data = enforce_float_tensor(data)

        # To allow fixing a subset of the parameter
        self.shape = data.shape
        self.update_is_fixed()

        self._param = nn.Parameter(
            self.data2param(data),
            requires_grad=requires_grad,
            # name=name
        )
        if self._param.ndim == 0:
            raise Warning('Use ndim>0 to allow consistent use of [:]. '
                          'If ndim=0, use paramname.v to access the '
                          'value.')

    def get_random_data0(self, data=None):
        if data is None:
            data = self.param2data(self._param)
        data = (
                   torch.rand_like(enforce_float_tensor(data))
               ) * (self.ub_random - self.lb_random) + self.lb_random
        return data

    def update_is_fixed(self):
        if (self.lb is not None and self.ub is not None):
            is_fixed = self.lb == self.ub
            self._fix_subset = is_fixed.any() and not is_fixed.all()
            if self._fix_subset:
                assert self.lb.shape == self.shape
                assert self.ub.shape == self.shape
                self._is_fixed = is_fixed
                self._fixed_data = self.lb[self._is_fixed]
            else:
                self._is_fixed = None
                self._fixed_data = None

    def data2param(self, data) -> torch.Tensor:
        # TODO: allow for non-scalar lb and ub
        #  to have -np.inf and np.inf elements.
        lb = npt.tensor(self.lb)
        ub = npt.tensor(self.ub)
        data = npt.tensor(enforce_float_tensor(data))

        if lb is not None and ub is not None and (lb == ub).all():
            if (data != lb).any():
                print('Out of lb=ub assignment to %s!' % type(self))
            return npt.zeros_like(data)

        if lb is not None:
            too_small = data < lb + self.epsilon
            if too_small.any():
                print('Out of lb assignment to %s!' % type(self))
        else:
            too_small = None

        if ub is not None:
            too_big = data > ub - self.epsilon
            if too_big.any():
                print('Out of ub assignment to %s!' % type(self))
        else:
            too_big = None

        if (lb is None) and (ub is None):  # Unbounded
            return data
        elif lb is None:
            data[too_big] = ub - self.epsilon
            return torch.log(ub - data)
        elif ub is None:
            data[too_small] = lb + self.epsilon
            return torch.log(data - lb)
        else:
            try:
                data[too_small] = lb + self.epsilon
            except RuntimeError:
                data[too_small] = (lb + self.epsilon)[too_small]

            try:
                data[too_big] = ub - self.epsilon
            except RuntimeError:
                data[too_big] = (ub - self.epsilon)[too_big]

            if self._fix_subset:
                is_free = ~self._is_fixed
                p = (data[is_free] - lb[is_free]) / (ub[is_free] - lb[is_free])
            else:
                p = (data - lb) / (ub - lb)
            return torch.log(p) - torch.log(1. - p)

    def param2data(self, param):
        lb = self.lb
        ub = self.ub
        param = enforce_float_tensor(param)

        def tensor1(v):
            return npt.tensor(v).to(param.device)
        if lb is not None:
            lb = tensor1(lb)
        if ub is not None:
            ub = tensor1(ub)

        if lb is None and ub is None: # Unbounded
            return param
        elif lb is None:
            return ub - torch.exp(param)
        elif ub is None:
            return lb + torch.exp(param)
        elif self._fix_subset:
            data = npt.empty(self.shape)
            data[self._is_fixed] = self._fixed_data
            is_free = ~self._is_fixed
            data[is_free] = (
                (1 / (1 + torch.exp(-param)))
                * (ub[is_free] - lb[is_free]) + lb[is_free]  # noqa
            )
            return data
        elif (lb == ub).all():
            return npt.zeros_like(param) + lb
        else:
            return (1 / (1 + torch.exp(-param))) * (ub - lb) + lb  # noqa

    def get_n_free_param(self) -> int:
        """Number of free parameters"""
        if not self._param.requires_grad:
            return 0
        else:
            lb = npt.tensor(-np.inf) if self.lb is None else self.lb
            ub = npt.tensor(np.inf) if self.ub is None else self.ub
            if self._fix_subset:
                return np.sum(npy(~self._is_fixed)).astype(int)
            else:
                return np.sum(npy(
                    ub.expand(self.shape)
                    > lb.expand(self.shape)
                ).astype(int))

    # NOTE: overriding load_state_dict() doesn't work because it doesn't know
    #  what prefix to use. Overriding _load_from_state_dict() as below worked.
    # def load_state_dict(
    #         self, state_dict,
    #         strict: bool = ...):
    #
    #     state_dict = state_dict.copy()
    #     self.lb = state_dict.pop('_lb')
    #     self.ub = state_dict.pop('_ub')
    #
    #     return super().load_state_dict(state_dict=state_dict,
    #                                    strict=strict)


class ProbabilityParameter(OverriddenParameter):
    def __init__(self, prob, probdim=0, requires_grad=True, randomize=False,
                 **kwargs):
        """

        :param prob:
        :param probdim:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.probdim = probdim
        self.lb = npt.zeros(1)
        self.ub = npt.ones(1)

        if randomize:
            prob1 = np.swapaxes(npy(prob), probdim, -1)
            prob = np.swapaxes(npy(torch.distributions.Dirichlet(
                npt.tensor(np.ones_like(prob1))
            ).sample()), probdim, -1)
        prob = npt.tensor(enforce_float_tensor(prob))

        self._param = nn.Parameter(self.data2param(prob), requires_grad=requires_grad)
        if self._param.ndim == 0:
            raise Warning('Use ndim>0 to allow consistent use of [:]. '
                          'If ndim=0, use paramname.v to access the '
                          'value.')
        self.skip_loading_lbub = True

    def data2param(self, prob):
        probdim = self.probdim
        prob = enforce_float_tensor(prob)

        prob[prob < self.epsilon] = self.epsilon
        prob[prob > 1. - self.epsilon] = 1. - self.epsilon
        prob = prob / torch.sum(prob, dim=probdim, keepdim=True)

        return torch.log(prob)

    def param2data(self, conf):
        return F.softmax(enforce_float_tensor(conf), dim=self.probdim)

    def get_n_free_param(self) -> int:
        """Number of free parameters after considering that probdim sums to 1"""
        len_probdim = self._param.shape[self.probdim]
        return int(self._param.numel()) // len_probdim * (len_probdim - 1)


class CircularParameter(BoundedParameter):
    def __init__(self, data, lb=0., ub=1., **kwargs):
        super().__init__(data, lb=lb, ub=ub, **kwargs)

    def data2param(self, data):
        data = enforce_float_tensor(data)
        return (data - self.lb) / (self.ub - self.lb) % 1.

    def param2data(self, param):
        param = enforce_float_tensor(param)
        return param * (self.ub - self.lb) + self.lb


#%% Bounded fit class
class LookUp(object):
    pass

class BoundedModule(nn.Module):
    def __init__(self):
        super().__init__()

        # for backward compatibility
        # - unused after introduction of BoundedParameter
        self._params_bounded = {} # {name:(lb, ub)}
        self._params_probability = {} # {name:probdim}
        self._params_circular = {} # {name:(lb, ub)}
        self.params_bounded = LookUp()
        self.params_probability = LookUp()
        self.params_circular = LookUp()
        self.epsilon = 1e-6

    def randomize(self):
        if len(self._params_bounded) > 0:
            raise NotImplementedError()
        if len(self._params_probability) > 0:
            raise NotImplementedError()
        if len(self._params_circular) > 0:
            raise NotImplementedError()

        d = odict(self.named_modules())
        for k, param in d.items():  # type: str, BoundedParameter
            if isinstance(param, BoundedParameter):
                data0 = param.get_random_data0()
                self.load_state_dict_data({k: data0})

    def setslice(self, name, index, value):
        v = self.__getattr__(name)
        v[index] = value
        self.__setattr__(name, v)

    # Bounded parameters whose elements are individually bounded
    def register_bounded_parameter(self, name, data, lb=0., ub=1.):
        lb = npt.tensor(lb, min_ndim=0)
        ub = npt.tensor(ub, min_ndim=0)
        data = enforce_float_tensor(data)
        self._params_bounded[name] = {'lb':lb, 'ub':ub}
        param = self._bounded_data2param(data, lb, ub)
        self.register_parameter('_bounded_' + name,
                                nn.Parameter(param, requires_grad=True))
        self.params_bounded.__dict__[name] = None # just a reminder

    def _bounded_data2param(self, data, lb=0., ub=1.):
        data = enforce_float_tensor(data)
        lb = npt.tensor(lb, min_ndim=0)
        ub = npt.tensor(ub, min_ndim=0)
        if lb is None and ub is None:  # Unbounded
            return data
        else:
            if lb is None:
                lb = -np.inf
            if ub is None:
                ub = np.inf
            data = torch.clamp(
                data, min=float(lb + self.epsilon),
                max=float(ub - self.epsilon)
            )

            if lb == -np.inf:
                # data[data > ub - self.epsilon] = ub - self.epsilon
                return torch.log(ub - data)
            elif ub == np.inf:
                # data[data < lb + self.epsilon] = lb + self.epsilon
                return torch.log(data - lb)
            else:
                # data[data < lb + self.epsilon] = lb + self.epsilon
                # data[data > ub - self.epsilon] = ub - self.epsilon
                p = (data - lb) / (ub - lb)
                return torch.log(p) - torch.log(1. - p)

    def _bounded_param2data(self, param, lb=0., ub=1.):
        lb = npt.tensor(lb, min_ndim=0)
        ub = npt.tensor(ub, min_ndim=0)
        param = enforce_float_tensor(param)
        if lb is None and ub is None:  # Unbounded
            return param
        elif lb is None:
            return ub - torch.exp(param)
        elif ub is None:
            return lb + torch.exp(param)
        else:
            return (1. / (1. + torch.exp(-param))) * (ub - lb) + lb

    # Probability parameters that should sum to 1 on a dimension
    def register_probability_parameter(self,
                                       name: str,
                                       prob: torch.Tensor,
                                       probdim=0) -> None:
        self._params_probability[name] = {'probdim':probdim}
        self.register_parameter('_conf_' + name,
                                nn.Parameter(self._prob2conf(prob)))
        self.params_probability.__dict__[name] = None

    def _prob2conf(self, prob, probdim=0):
        prob = enforce_float_tensor(prob)
        incl = prob < self.epsilon

        prob[prob < self.epsilon] = self.epsilon
        prob[prob > 1. - self.epsilon] = 1. - self.epsilon
        prob = prob / torch.sum(prob, dim=probdim, keepdim=True)

        return torch.log(prob)

    def _conf2prob(self, conf, probdim=0):
        return F.softmax(enforce_float_tensor(conf), dim=probdim)

    # Circular parameters: limited to (param - lb) % (ub - lb)
    def register_circular_parameter(self, name, data, lb=0., ub=1.):
        data = enforce_float_tensor(data)
        self._params_circular[name] = {'lb':lb, 'ub':ub}
        param = self._circular_data2param(data, lb, ub)
        self.register_parameter('_circular_' + name, nn.Parameter(param))
        self.params_circular.__dict__[name] = None

    def _circular_data2param(self, data, lb=0., ub=1.):
        data = enforce_float_tensor(data)
        return (data - lb) / (ub - lb) % 1.

    def _circular_param2data(self, param, lb=0., ub=1.):
        param = enforce_float_tensor(param)
        return param * (ub - lb) + lb

    # Convenience function
    @property
    def _parameters_incl_bounded(self):
        p0 = self._parameters
        k_all = list(p0.keys())
        p = odict()
        for k in k_all:
            for prefix in ['_bounded_', '_conf_']:
                if k.startswith(prefix):
                    k1 = k[len(prefix):]
                    p[k1] = self.__getattr__(k1)
                    break
            else:
                p[k] = p0[k]
        return p

    def named_bounded_param_value(self):
        d = odict(self.named_modules())
        return odict([
            (k, param.v)
            for k, param in d.items()
            if isinstance(param, OverriddenParameter)
        ])

    def named_bounded_lb(self):
        d = odict(self.named_modules())
        return odict([
            (k, param.lb)
            for k, param in d.items()
            if isinstance(param, OverriddenParameter)
        ])

    def named_bounded_ub(self):
        d = odict(self.named_parameters())
        return odict([
            (k, param.ub)
            for k, param in d.items()
            if isinstance(param, OverriddenParameter)
        ])

    # Get/set
    def __getattr__(self, item):
        # if item[0] == '_':
        #     return super().__getattribute__(item)

        # DEBUGGED: being specific about escaped item allows using
        #  parameter names starting with '_'
        if item in ['_modules', '_params_bounded', '_params_probability',
                    '_params_circular']:
            try:
                return super(BoundedModule, self).__getattribute__(item)
            except:
                return {}

        if hasattr(self, '_params_bounded'):
            _params = self.__dict__['_params_bounded']
            if item in _params:
                info = _params[item]
                param = super().__getattr__(
                    '_bounded_' + item)
                return self._bounded_param2data(param, **info)

        if hasattr(self, '_params_probability'):
            _params = self.__dict__['_params_probability']
            if item in _params:
                info = _params[item]
                param = super().__getattr__(
                    '_conf_' + item)
                return self._conf2prob(param, **info)

        if hasattr(self, '_params_circular'):
            _params = self.__dict__['_params_circular']
            if item in _params:
                info = _params[item]
                param = super().__getattr__(
                    '_circular_' + item)
                return self._circular_param2data(param, **info)

        return super().__getattr__(item)
        # v = super().__getattr__(item)
        # if isinstance(v, OverriddenParameter):
        #     return v.v
        # else:
        #     return v

    def __setattr__(self, item, value):
        # if isinstance(value, OverriddenParameter):
        #     self.add_module(item, value)
        #     return
        #     super().__setattr__(item, value)
        #     return

        # if (hasattr(self, '_modules')
        #         and item in self._modules
        #         and isinstance(self._modules[item], OverriddenParameter)
        # ):
        #     self._modules[item].v = value
        #     return

        if item in ['_params_bounded', '_params_probability']:
            self.__dict__[item] = value
            return

        if hasattr(self, '_params_bounded'):
            _params = self._params_bounded
            if item in _params:
                info = _params[item]
                param = self._bounded_data2param(value, **info)
                super().__setattr__(
                    '_bounded_' + item, nn.Parameter(param)
                )
                return

        if hasattr(self, '_params_probability'):
            _params = self._params_probability
            if item in _params:
                info = _params[item]
                param = self._prob2conf(value, **info)
                super().__setattr__(
                    '_conf_' + item, nn.Parameter(param)
                )
                return

        if hasattr(self, '_params_circular'):
            _params = self._params_circular
            if item in _params:
                info = _params[item]
                param = self._circular_data2param(value, **info)
                super().__setattr__(
                    '_circular_' + item, nn.Parameter(param)
                )
                return

        # if isinstance(value, OverriddenParameter):
        #     if value.name is None:
        #         value.name = item

        return super().__setattr__(item, value)

    # def __delattr__(self, item):
    #     if item in self._params_overridden:
    #         self._params_overridden.remove(item)
    #
    #     super().__delattr__(item)

    def __str__(self):
        def indent(s):
            return ['  ' + s1 for s1 in s]
        l = [
            type(self).__name__
        ]
        for name, v in self._parameters_incl_bounded.items():
            l += indent(
                str((name, v)).split('\n')
            )
        for name, v in self._modules.items():
            if isinstance(v, OverriddenParameter):
                l += indent(
                    [str((name, v.param2data(v._param)))]
                )
            else:
                l += indent(['%s (%s)' % (name, type(v).__name__)]) + \
                    indent(v.__str__().split('\n'))

        return '\n'.join(l)

    def plot_params(
            self,
            named_bounded_params: Sequence[Tuple[str, BoundedParameter]] = None,
            exclude: Iterable[str] = (),
            cmap='coolwarm',
            ax: plt.Axes = None
    ) -> mpl.container.BarContainer:
        if ax is None:
            ax = plt.gca()

        ax = plt.gca()
        names, v, grad, lb, ub, requires_grad = self.get_named_bounded_params(
            named_bounded_params, exclude=exclude)
        max_grad = np.amax(np.abs(grad))
        if max_grad == 0:
            max_grad = 1.
        v01 = (v - lb) / (ub - lb)
        grad01 = (grad + max_grad) / (max_grad * 2)
        n = len(v)

        grad01[~requires_grad] = np.nan
        # (np.amin(grad01) + np.amax(grad01)) / 2

        # ax = plt.gca()  # CHECKED

        for i, (lb1, v1, ub1, g1, r1) in enumerate(zip(lb, v, ub, grad,
                                                       requires_grad)):
            color = 'k' if r1 else 'gray'
            plt.text(0, i, '%1.2g' % lb1, ha='left', va='center', color=color)
            plt.text(1, i, '%1.2g' % ub1, ha='right', va='center', color=color)
            plt.text(0.5, i, '%1.2g %s'
                     % (v1,
                        ('(e%1.0f)' % np.log10(np.abs(g1)))
                        if r1 else '(fixed)'),
                     ha='center',
                     va='center',
                     color=color
                     )
        lut = 256
        colors = plt.get_cmap(cmap, lut)(grad01)
        for i, r in enumerate(requires_grad):
            if not r:
                colors[i, :3] = np.array([0.95, 0.95, 0.95])
        h = ax.barh(np.arange(n), v01, left=0, color=colors)
        ax.set_xlim(-0.025, 1)
        ax.set_xticks([])
        ax.set_yticks(np.arange(n))

        names = [v.replace('_', ' ') for v in names]
        ax.set_yticklabels(names)
        ax.set_ylim(n - 0.5, -0.5)
        plt2.box_off(['top', 'right', 'bottom'])
        plt2.detach_axis('x', amin=0, amax=1)
        plt2.detach_axis('y', amin=0, amax=n - 1)

        # plt.show()  # CHECKED

        return h

    def get_named_bounded_params(
            self, named_bounded_params: Dict[str, BoundedParameter] = None,
            exclude: Iterable[str] = ()
    ) -> (Iterable[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """

        :param named_bounded_params:
        :param exclude:
        :return: names, v, grad, lb, ub
        """
        if named_bounded_params is None:
            d = odict([(k, v) for k, v in self.named_modules()
                       if (isinstance(v, OverriddenParameter) #
                           # BoundedParameter)
                           and k not in exclude)])
        else:
            d = named_bounded_params
        names = []
        v = []
        lb = []
        ub = []
        grad = []
        requires_grad = []
        for name, param in d.items():
            v0 = param.v.flatten()
            if param._param.grad is None:
                g0 = npt.zeros_like(v0)
            else:
                g0 = param._param.grad.flatten()
            if param.lb is None:
                l0 = npt.tensor([-np.inf]).expand_as(param.v).flatten()
            else:
                l0 = npt.tensor(param.lb).expand_as(param.v).flatten()
            if param.ub is None:
                u0 = npt.tensor([np.inf]).expand_as(param.v).flatten()
            else:
                u0 = npt.tensor(param.ub).expand_as(param.v).flatten()

            for i, (v1, g1, l1, u1) in enumerate(zip(v0, g0, l0, u0)):
                v.append(npy(v1))
                grad.append(npy(g1))
                lb.append(npy(l1))
                ub.append(npy(u1))
                requires_grad.append(npy(param._param.requires_grad))
                if v0.numel() > 1:
                    names.append(name + '%d' % i)
                else:
                    names.append(name)
        v = np.stack(v)
        lb = np.stack(lb)
        ub = np.stack(ub)
        grad = -np.stack(grad)  # minimizing; so take negative
        requires_grad = np.stack(requires_grad)
        return names, v, grad, lb, ub, requires_grad

    def get_n_free_param(self) -> int:
        """Number of free parameters."""
        n_free = 0
        for name, module in self.named_modules():
            if module is not self and (
                    isinstance(module, OverriddenParameter)
                    or isinstance(module, BoundedModule)
            ):
                n_free += module.get_n_free_param()
        return n_free

    def load_state_dict_data(
            self, state_dict_data: Dict[str, torch.Tensor], strict=False
    ) -> (List[str], List[str]):
        """
        EXAMPLE:
        from pprint import pprint
        import yktorch as ykt

        class Objective(ykt.BoundedModule):
            def __init__(self):
                self.a = ykt.BoundedParameter([0.], lb=-1., ub=1.)
        obj = Objective()
        obj.load_state_dict_data({'a': torch.tensor([0.5])})
        pprint(obj.named_bounded_param_value())
        print(obj0.__str__())

        :param state_dict_data: {param_name: value} where param_name omits
            '._data' used by BoundedModule for convenience
        :param strict: whether to strictly match the keys in state_dict()
        :return: missing_keys (list) and unexpected keys (list)
        """
        state_dict1 = {
            k + '._data': v for k, v in state_dict_data.items()
        }
        return self.load_state_dict(state_dict1, strict=strict)

    def state_dict_data(self) -> Dict[str, torch.Tensor]:
        """
        An alias for named_bounded_param_value()
            that rhymes with load_state_dict_data()
        :return: parameter values (transformed to respect bounds).
        """
        return self.named_bounded_param_value()

    def parameters_data_vec(self) -> torch.Tensor:
        p = self.state_dict_data()
        return torch.cat([v.flatten() for v in p.values()], 0)

    def param_vec2value_vec(
            self, param_vec: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        param_vec0 = self.parameters_data_vec()
        self.load_state_dict(self.param_vec2dict(param_vec), strict=False)
        value_vec = self.parameters_data_vec()
        self.load_state_dict(self.param_vec2dict(param_vec0), strict=False)
        return value_vec

    def param_vec2dict(self, param_vec, suffix='._param') -> odict:
        dict0 = self.state_dict()  # type: Dict[str, torch.Tensor]
        dict1 = odict()
        if not torch.is_tensor(param_vec):
            param_vec = npt.tensor(param_vec)
        for k, v0 in dict0.items():
            if not k.endswith(suffix):
                continue
            size = v0.size()
            n = v0.numel()

            v1 = param_vec[:n]
            param_vec = param_vec[n:]

            v1 = v1.reshape(size)
            dict1[k] = v1
        return dict1

    def grad_vec(self):
        ps = self.parameters()
        return torch.cat([
            (p.grad.flatten() if p.requires_grad and p.grad is not None
             else npt.zeros(p.numel()))
            for p in ps
        ])

    def freeze_(self):
        """Freeze all parameters (set requires_grad=False)"""
        raise DeprecationWarning('use requires_grad_(False) instead!')
        self.set_requires_grad_(False)
        return self

    def unfreeze_(self):
        """Unfreeze all parameters (set requires_grad=True)"""
        raise DeprecationWarning('use requires_grad_(True) instead!')
        self.set_requires_grad_(True)
        return self

    def set_requires_grad_(self, requires_grad: bool):
        raise DeprecationWarning('use requires_grad_() instead!')
        npt.set_requires_grad(self, requires_grad)


def enforce_float_tensor(v: Union[torch.Tensor, np.ndarray], device=None
                         ) -> torch.Tensor:
    """
    :rtype: torch.DoubleTensor, torch.FloatTensor
    """
    if device is None:
        device = default_device
    if not torch.is_tensor(v):
        return npt.tensor(v, dtype=torch.get_default_dtype(), device=device)
    elif not torch.is_floating_point(v):
        return v.float()
    else:
        return v

#%% Test bounded module
import unittest
class TestBoundedModule(unittest.TestCase):
    # TODO
    @unittest.skip('until test for BoundedParameter is written')
    def test_bounded_data2param2data(self):
        def none2str(v):
            if v is None:
                return 'None'
            else:
                return '%d' % v

        bound = BoundedModule()
        data = npt.zeros(2, 3)
        for lb in [None, 0, -10, 10]:
            for ub in [None, 0, -5, 20]:
                if lb is not None and ub is not None and lb >= ub:
                    continue

                param = bound._bounded_data2param(data, lb, ub)
                data1 = bound._bounded_param2data(param, lb, ub)
                min_err = torch.min(torch.abs(data1 - data))
                self.assertTrue(
                    min_err <= bound.epsilon * 10,
                    'error of %g with lb=%s, ub=%s' % (
                        min_err, none2str(lb), none2str(ub))
                )

    # TODO
    @unittest.skip('until test for BoundedParameter is written')
    def test_register_bounded_parameter(self):
        def none2str(v):
            if v is None:
                return 'None'
            else:
                return '%d' % v

        bound = BoundedModule()
        data = npt.zeros(2, 3)
        for lb in [None, 0, -10, 10]:
            for ub in [None, 0, -5, 20]:
                if lb is not None and ub is not None and lb >= ub:
                    continue

                bound.register_bounded_parameter('bounded', data, lb, ub)
                data1 = bound.bounded

                min_err = torch.min(torch.abs(data1 - data))
                self.assertTrue(
                    min_err <= bound.epsilon * 10,
                    'error of %g with lb=%s, ub=%s' % (
                        min_err, none2str(lb), none2str(ub))
                )

    def test_register_circular_parameter(self):
        def none2str(v):
            if v is None:
                return 'None'
            else:
                return '%d' % v

        bound = BoundedModule()
        data = npt.zeros(2, 3)
        for lb in [0, 5, 20]:
            for scale in [1, 5, 20]:
                for offset in [0, 0.01, 0.5, scale]:
                    bound.register_circular_parameter('circular0',
                                                      data + lb
                                                      + offset, # w/o + scale
                                                      lb, lb + scale)
                    bound.register_circular_parameter('circular1',
                                                      data + lb
                                                      + scale + offset,
                                                      lb, lb + scale)
                    data0 = bound.circular0
                    data1 = bound.circular1

                    min_err = torch.min(torch.abs(data1 - data0))
                    self.assertTrue(
                        min_err <= bound.epsilon * 10,
                        'error of %g with lb=%s, scale=%s, offset=%g' % (
                            min_err, lb, scale, offset)
                    )

    def test_prob_data2param2data(self):
        bound = BoundedModule()
        for p in [0, 0.5, 1]:
            data = torch.cat([
                npt.zeros(1, 3) + p,
                npt.ones(1, 3) - p
                ], dim=0)
            param = bound._prob2conf(data)
            data1 = bound._conf2prob(param)

            min_err = torch.min(torch.abs(data1 - data))
            self.assertTrue(
                min_err <= bound.epsilon * 10,
                'error of %g with p=%g' % (
                    min_err, p)
            )

    def test_register_probability_parameter(self):
        bound = BoundedModule()
        for p in [0, 0.5, 1]:
            data = torch.cat([
                npt.zeros(1, 3) + p,
                npt.ones(1, 3) - p
                ], dim=0)
            bound.register_probability_parameter('prob', data)
            data1 = bound.prob

            min_err = torch.min(torch.abs(data1 - data))
            self.assertTrue(
                min_err <= bound.epsilon * 10,
                'error of %g with p=%g' % (
                    min_err, p)
            )


def ____Optimizer____():
    pass


def print_grad(model):
    print('Gradient:')
    pprint({k: v.grad for k, v in model.named_parameters()})


def flatten_dict(
        d: Dict[str, Union[torch.Tensor, np.ndarray]]
) -> Dict[str, float]:
    d1 = {}
    for k, v in d.items():
        v = npy(v)
        for i1, v1 in enumerate(v.flatten()):
            k1 = ('%s[%s]'
               % (k, ','.join([('%d' % v11)
                               for v11 in np.unravel_index(i1, v.shape)])))
            d1[k1] = v1
    return d1


# def plot_params(
#         params: Union[torch.nn.Module,
#                       Iterable[Tuple[str, torch.nn.Parameter]]],
# ):
#     """
#     :param params: Module or param.named_parameters()
#     :return:
#     """
#     if isinstance(params, torch.nn.Module):
#         params = odict(params.named_parameters())
#     else:
#         params = odict(params)
#
#
#     names = [params.keys()]
#     if to_plot_grad:
#         v = []
#
#     # for name, tensor in params.items():


ModelType = Union[OverriddenParameter, BoundedModule, nn.Module]
FunDataType = Callable[
    [str, int, int, int],
    Tuple[Union[None, torch.Tensor, Tuple[torch.Tensor, ...]],
          Union[None, torch.Tensor, Tuple[torch.Tensor, ...]]]
    # (mode='all'|'train'|'valid'|'train_valid'|'test', fold_valid=0, epoch=0,
    #  n_fold_valid=1)
    # -> (data, target)
    # Multiple data and target outputs can be accommodated using tuples.
    # NOTE: added n_fold_valid to avoid mistakes having fun_data conditioned
    #  to n_fold_valid different from that of optimize()
]
FunLossType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
PlotFunType = Callable[
    [ModelType, Dict[str, torch.Tensor]],
    Tuple[plt.Figure, Dict[str, torch.Tensor]]
]
PlotFunsType = Iterable[Tuple[str, PlotFunType]]


def optimize(
        model: ModelType,
        fun_data: FunDataType = None,
        fun_loss: FunLossType = None,
        plotfuns: PlotFunsType = None,
        optimizer_kind='Adam',
        max_epoch=100,
        patience=20,  # How many epochs to wait before quitting
        thres_patience=0.001,  # How much should it improve wi patience
        learning_rate=.5,
        reduce_lr_by=0.5,
        reduced_lr_on_epoch=0,
        reduce_lr_after=50,
        reset_lr_after=100,
        to_plot_progress=True,
        show_progress_every=5,  # number of epochs
        to_print_grad=True,
        n_fold_valid=1,
        epoch_to_check=None,  # CHECKED
        comment='',
        best_grad_mode='train',
        **kwargs  # to ignore unnecessary kwargs
) -> (float, dict, dict, List[float], List[float]):
    """

    :param model:
    :param fun_data: (mode='all'|'train'|'valid'|'train_valid'|'test',
    fold_valid=0, epoch=0, n_fold_valid=1) -> (data, target)
    :param fun_loss: (out, target) -> loss
    :param plotfuns: [(name_plotfun: str, plotfun: PlotFunType)]
    where plotfun(model, d) -> (fgure, d)
    and d is as returned by optimize() itself
    :param optimizer_kind:
    :param max_epoch:
    :param patience:
    :param thres_patience:
    :param learning_rate:
    :param reduce_lr_by:
    :param reduced_lr_on_epoch:
    :param reduce_lr_after:
    :param to_plot_progress:
    :param show_progress_every:
    :param to_print_grad:
    :param n_fold_valid:
    :param best_grad_mode: take gradient after fitting with this data mode;
    'None' to skip
    :param kwargs:
    :return: loss_test, best_state, d, losses_train, losses_valid where d
    contains 'data_*', 'target_*', 'out_*', and 'loss_*', where * is
    'train_valid', 'test', and 'all'.
    """
    def get_optimizer(model, lr):
        if optimizer_kind == 'SGD':
            return optim.SGD(model.parameters(),
                             lr=lr)
        elif optimizer_kind == 'Adam':
            return optim.Adam(model.parameters(),
                              lr=lr)
        elif optimizer_kind == 'LBFGS':
            return optim.LBFGS(model.parameters(),
                               lr=lr)
        else:
            raise NotImplementedError()

    if fun_data is None:
        fun_data = lambda *args: (None, None)
    if fun_loss is None:
        fun_loss = lambda *args: model()
    if plotfuns is None:
        plotfuns = {}

    learning_rate0 = learning_rate
    optimizer = get_optimizer(model, learning_rate)

    if optimizer_kind == 'LBFGS':
        assert n_fold_valid == 1, 'n_fold_valid > 1 is not implemented ' \
                                  'for LBFGS yet - ' \
                                  'will need averaging loss within the closure!'

    best_loss_epoch = 0
    best_loss_valid = np.inf
    best_state = model.state_dict()
    best_losses = []

    # CHECKED storing and loading states
    state0 = None
    loss0 = None
    data0 = None
    target0 = None
    out0 = None
    outs0 = None

    def array2str(v):
        return ', '.join(['%1.2g' % v1 for v1 in v.flatten()[:10]])

    def print_targ_out(target0, out0, outs0, loss0):
        print('target:\n' + array2str(target0))
        print('outs:\n' + '\n'.join(
            ['[%s]' % array2str(v) for v in outs0]))
        print('out:\n' + array2str(out0))
        print('loss: ' + '%g' % loss0)

    def fun_outs(model, data):
        p_bef_lapse0 = model.dtb(*data)[0].detach().clone()
        p_aft_lapse0 = model.lapse(p_bef_lapse0).detach().clone()
        return [
            p_bef_lapse0, p_aft_lapse0
        ]

    def are_all_equal(outs, outs0):
        for i, (out1, out0) in enumerate(zip(outs, outs0)):
             if (out1 != out0).any():
                 warnings.warn(
                     'output %d different! max diff = %g' %
                     (i, (out1 - out0).abs().max()))
                 print('--')

    # losses_train[epoch] = average cross-validated loss for the epoch
    losses_train = []
    losses_valid = []

    if to_plot_progress and max_epoch > 0:
        writer = SummaryWriter(comment=comment)
    t_st = time.time()
    epoch = 0

    def loss_wo_nan(model1, data1, target1) -> torch.Tensor:
        is_nan = (
            torch.isnan(data1.reshape(data1.shape[0], -1)).any(-1)
            | torch.isnan(target1.reshape(target1.shape[0], -1)).any(-1)
        )
        out1 = model1(data1[~is_nan])
        return fun_loss(out1, target1[~is_nan]), out1

    try:
        for epoch in range(max([max_epoch, 1])):
            losses_fold_train = []
            losses_fold_valid = []
            for i_fold in range(n_fold_valid):
                # NOTE: Core part
                data_train, target_train = fun_data('train', i_fold, epoch,
                                                    n_fold_valid)
                model.train()
                optimizer.zero_grad()
                loss_train1, out_train = loss_wo_nan(
                    model, data_train, target_train)
                # DEBUGGED: optimizer.step() must not be taken before
                #  storing best_loss or best_state
                losses_fold_train.append(loss_train1)

                if n_fold_valid == 1:
                    out_valid = npt.tensor(npy(out_train))
                    loss_valid1 = npt.tensor(npy(loss_train1))
                    data_valid = data_train
                    target_valid = target_train

                    # DEBUGGED: Unless directly assigned, target_valid !=
                    #  target_train when n_fold_valid = 1, which doesn't make
                    #  sense. Suggests a bug in fun_data when n_fold = 1
                else:
                    model.eval()
                    data_valid, target_valid = fun_data('valid', i_fold, epoch,
                                                        n_fold_valid)
                    loss_valid1, out_valid = loss_wo_nan(
                        model, data_train, target_train)
                    model.train()
                losses_fold_valid.append(loss_valid1)

            loss_train = torch.mean(torch.stack(losses_fold_train))
            loss_valid = torch.mean(torch.stack(losses_fold_valid))
            losses_train.append(npy(loss_train))
            losses_valid.append(npy(loss_valid))

            # steps are not taken here,
            # since it's BEFORE storing the best state
            # Still, take the gradient for plotting
            loss_train.backward()

            if to_plot_progress and max_epoch > 0:
                writer.add_scalar(
                    'loss_train', loss_train,
                    global_step=epoch
                )
                writer.add_scalar(
                    'loss_valid', loss_valid,
                    global_step=epoch
                )

            # --- Store best loss
            # NOTE: storing losses/states must happen BEFORE taking a step!
            if loss_valid < best_loss_valid:
                # is_best = True
                best_loss_epoch = deepcopy(epoch)
                best_loss_valid = npt.tensor(npy(loss_valid))
                best_state = model.state_dict()
                # pprint(model.state_dict_data())  # CHECKED best state
                # best_state_data = model.state_dict_data()
                # if loss_valid < 436:
                #     pprint(best_state_data)
                #     print('--')

            best_losses.append(best_loss_valid)

            # CHECKED storing and loading state
            if epoch == epoch_to_check:
                loss0 = loss_valid.detach().clone()
                state0 = model.state_dict()
                data0 = deepcopy(data_valid)
                target0 = deepcopy(target_valid)
                out0 = out_valid.detach().clone()
                outs0 = fun_outs(model, data0)

                loss001, _ = loss_wo_nan(model, data0, target0)
                # CHECKED: loss001 must equal loss0
                print('loss001 - loss0: %g' % (loss001 - loss0))

                print_targ_out(target0, out0, outs0, loss0)
                print('--')

            def print_loss():
                t_el = time.time() - t_st
                print('%1.0f sec/%d epochs = %1.1f sec/epoch, Ltrain: %f, '
                      'Lvalid: %f, LR: %g, best: %f, epochB: %d'
                      % (t_el, epoch + 1, t_el / (epoch + 1),
                         loss_train, loss_valid, learning_rate,
                         best_loss_valid, best_loss_epoch))

            if torch.isnan(loss_train):
                print_loss()
                print('NaN loss!')
                print('--')

            if show_progress_every != 0 and epoch % show_progress_every == 0:
                model.train()
                data_train_valid, target_train_valid = fun_data(
                    'train_valid', i_fold, epoch, n_fold_valid
                )
                loss_train_valid, out_train_valid = loss_wo_nan(
                    model, data_train_valid, target_train_valid)
                print_loss()

                if to_plot_progress and max_epoch > 0:
                    d = {
                        'data_train': data_train,
                        'data_valid': data_valid,
                        'data_train_valid': data_train_valid,
                        'out_train': out_train,
                        'out_valid': out_valid,
                        'out_train_valid': out_train_valid,
                        'target_train': target_train,
                        'target_valid': target_valid,
                        'target_train_valid': target_train_valid,
                        'loss_train': loss_train,
                        'loss_valid': loss_valid,
                        'loss_train_valid': loss_train_valid
                    }
                    d = {k: v.detach() if torch.is_tensor(v) else v
                         for k, v in d.items()}
                    for k, f in odict(plotfuns).items():
                        fig, d = f(model, d)
                        if fig is not None:
                            writer.add_figure(k, fig, global_step=epoch)
                    d = {
                        k: (v.detach() if torch.is_tensor(v) else v)
                        for k, v in d.items()
                    }

            # --- Learning rate reduction and patience
            # if epoch == reduced_lr_on_epoch + reset_lr_after
            # if epoch == reduced_lr_on_epoch + reduce_lr_after and (
            #         best_loss_valid
            #         > best_losses[-reduce_lr_after] - thres_patience
            # ):
            if epoch > 0 and epoch % reset_lr_after == 0:
                learning_rate = learning_rate0
            elif epoch > 0 and epoch % reduce_lr_after == 0:
                learning_rate *= reduce_lr_by
                optimizer = get_optimizer(model, learning_rate)
                reduced_lr_on_epoch = epoch

            if epoch >= patience and (
                    best_loss_valid
                    > best_losses[-patience] - thres_patience
            ):
                print('Ran out of patience!')
                if to_print_grad:
                    print_grad(model)
                break
            if to_print_grad and epoch == 0:
                print_grad(model)

            # --- Take a step (do after plotting)
            if optimizer_kind == 'LBFGS':
                model.train()
                def closure():
                    optimizer.zero_grad()
                    loss, out_train = loss_wo_nan(
                        model, data_train, target_train)
                    loss.backward()
                    return loss
                if max_epoch > 0:
                    optimizer.step(closure)
            else:
                # steps are not taken above for n_fold_valid == 1, so take a
                # step here, AFTER storing the best state
                if max_epoch > 0:
                    optimizer.step()

    except Exception as ex:
        from .cacheutil import is_keyboard_interrupt
        if not is_keyboard_interrupt(ex):
            raise ex
        print('fit interrupted by user at epoch %d' % epoch)

        from .localfile import LocalFile, datetime4filename
        localfile = LocalFile()
        cache = localfile.get_cache('model_data_target')
        data_train_valid, target_train_valid = fun_data(
            'all', 0, 0, n_fold_valid)
        cache.set({
            'model': model,
            'data_train_valid': data_train_valid,
            'target_train_valid': target_train_valid
        })
        cache.save()

    print_loss()
    if to_plot_progress and max_epoch > 0:
        writer.close()

    if epoch_to_check is not None:
        # Must print the same output as previous call to print_targ_out
        print_targ_out(target0, out0, outs0, loss0)

        model.load_state_dict(state0)
        state1 = model.state_dict()
        for (key0, param0), (key1, param1) in zip(
                state0.items(), state1.items()
        ):  # type: ((str, torch.Tensor), (str, torch.Tensor))
            if (param0 != param1).any():
                with torch.no_grad():
                    warnings.warn(
                        'Strange! loaded %s = %s\n'
                        '!= stored %s = %s\n'
                        'loaded - stored = %s'
                        % (key1, param1, key0, param0, param1 - param0))
        data, target = fun_data('valid', 0, epoch_to_check, n_fold_valid)

        if not torch.is_tensor(data):
            p_unequal = npt.tensor([
                (v1 != v0).double().mean() for v1, v0
                in zip(data, data0)
            ])
            if (p_unequal > 0).any():
                print('Strange! loaded data != stored data0\n'
                      'Proportion: %s' % p_unequal)
            else:
                print('All loaded data == stored data')
        elif (data != data0).any():
            print('Strange! loaded data != stored data0')
        else:
            print('All loaded data == stored data')

        if (target != target0).any():
            print('Strange! loaded target != stored target0')
        else:
            print('All loaded target == stored target')

        print_targ_out(target0, out0, outs0, loss0)

        # with torch.no_grad():
        #     out01 = model(data0)
        #     loss01 = fun_loss(out01, target0)
        model.train()
        # with torch.no_grad():
        # CHECKED
        # outs1 = fun_outs(model, data)
        # are_all_equal(outs1, outs0)

        out1 = model(data)
        if (out0 != out1).any():
            warnings.warn(
                'Strange! out from loaded params != stored out\n'
                'Max abs(loaded - stored): %g' %
                (out1 - out0).abs().max())
            print('--')
        else:
            print('out from loaded params = stored out')

        loss01 = fun_loss(out0, target0)
        print_targ_out(target0, out0, outs0, loss01)

        if loss0 != loss01:
            warnings.warn(
                'Strange!  loss1 = %g simply computed again with out0, '
                'target0\n'
                '!= stored loss0 = %g\n'
                'loaded - stored:  %g\n'
                'Therefore, fun_loss, out0, or target0 has changed!' %
                (loss01, loss0, loss01 - loss0))
            print('--')
        else:
            print('loss0 == loss01, simply computed again with out0, target0')

        loss1 = fun_loss(out1, target)
        if loss0 != loss1:
            warnings.warn(
                'Strange!  loss1 = %g from loaded params\n'
                '!= stored loss0 = %g\n'
                'loaded - stored:  %g' %
                (loss1, loss0, loss1 - loss0))
            print('--')
        else:
            print('loss1 = %g = loss0 = %g' % (loss1, loss0))

        loss10 = fun_loss(out1, target0)
        if loss0 != loss1:
            warnings.warn(
                'Strange!  loss10 = %g from loaded params and stored '
                'target0\n'
                '!= stored loss0 = %g\n'
                'loaded - stored:  %g' %
                (loss10, loss0, loss10 - loss0))
            print('--')
        else:
            print('loss10 = %g = loss10 = %g' % (loss1, loss0))
        print('--')

    model.load_state_dict(best_state)

    d = {}
    for mode in ['train_valid', 'valid', 'test', 'all']:
        data, target = fun_data(mode, 0, 0, n_fold_valid)

        if mode == best_grad_mode:
            model.train()
            model.zero_grad()

        loss, out = loss_wo_nan(model, data, target)

        if mode == best_grad_mode:
            loss.backward()

        d.update({
            'data_' + mode: data,
            'target_' + mode: target,
            'out_' + mode: npt.tensor(npy(out)),
            'loss_' + mode: npt.tensor(npy(loss))
        })

    if d['loss_valid'] != best_loss_valid:
        print('d[loss_valid]      = %g from loaded best_state \n'
              '!= best_loss_valid = %g\n'
              'd[loss_valid] - best_loss_valid = %g' %
              (d['loss_valid'], best_loss_valid,
               d['loss_valid'] - best_loss_valid))
        # pprint(best_state_data)
        # pprint(model.state_dict_data())
        print('--')

    if isinstance(model, OverriddenParameter):
        print(model.__str__())
    elif isinstance(model, BoundedModule):
        pprint(model._parameters_incl_bounded)
    else:
        pprint(model.state_dict())

    return d['loss_test'], best_state, d, losses_train, losses_valid


def tensor2str(v: Union[torch.Tensor], sep='; ') -> str:
    """Make a string that is human-readable and csv-compatible"""
    if v is None:
        return '() None'
    else:
        return '(%s) %s' % (
            sep.join(['%d' % s for s in v.shape]),
            sep.join(
                ['%g' % v1 for v1 in v.flatten()]
            ) if v.ndimension() > 0 else '%g' % v
        )


def save_optim_results(
        model: ModelType = None,
        best_state: Dict[str, torch.Tensor] = None,
        d: Dict[str, torch.Tensor] = None,
        plotfuns: PlotFunsType = None,
        fun_tab_file: Callable[[str, str], str] = None,
        fun_fig_file: Callable[[str, str], str] = None,
        plot_exts=('.png',)
) -> Iterable[str]:
    """

    :param model:
    :param best_state: model.state_dict()
    :param d: as returned from optimize()
    :param plotfuns: [fun_plot(model, d), ...]
        where fun_plot(model, d) -> (figure, _)
    :param fun_tab_file: (file_kind, extension) -> fullpath
    :param fun_fig_file: (file_kind, extension) -> fullpath
    :param plot_exts:
    :return:
    """
    files = []

    if fun_tab_file is None:
        def fun_tab_file(name, ext):
            return name + ext

    if fun_fig_file is None:
        def fun_fig_file(name, ext):
            return 'plt=%s%s' % (name, ext)

    if model is not None:
        best_state = odict(model.named_parameters())
    if best_state is not None:
        file = fun_tab_file('best_state', '.csv')
        mkdir4file(file)
        with open(file, 'w') as f:
            if isinstance(model, BoundedModule):
                names, v, grad, lb, ub, requires_grad = \
                    model.get_named_bounded_params()

                f.write('name, value, gradient, lb, ub, requires_grad\n')
                for name, v1, grad1, lb1, ub1, requires_grad1 in zip(
                    names, v, grad, lb, ub, requires_grad
                ):
                    f.write('%s, %g, %g, %g, %g, %d\n' % (
                        name, v1, grad1, lb1, ub1, requires_grad1
                    ))
            else:
                f.write('name, value, gradient\n')
                for k, v in best_state.items():
                    f.write('%s, %s, %s\n'
                            % (k, tensor2str(v), tensor2str(v.grad)))
        print('Saved to %s' % file)
        files.append(file)

    if d is not None:
        files1 = fun_tab_file('best_loss', '.csv')
        if isinstance(files1, str):
            files1 = [files1]
        for file in files1:
            mkdir4file(file)
            with open(file, 'w') as f:
                f.write('name, value\n')
                for k, v in d.items():
                    if k.startswith('loss'):
                        f.write('%s, %s\n'
                                % (k, (tensor2str(v) if torch.is_tensor(v)
                                       else '%g' % v)))
            print('Saved to %s' % file)
            files.append(file)

    if plotfuns is not None and model is not None and d is not None:
        plotfuns = odict(plotfuns)
        for k, fun_plot in plotfuns.items():
            for plot_ext in plot_exts:
                files1 = fun_fig_file(k, plot_ext)
                if isinstance(files1, str):
                    files1 = [files1]
                for file in files1:
                    mkdir4file(file)
                    fig, _ = fun_plot(model, d)
                    fig.savefig(file, dpi=300)
                    print('Saved to %s' % file)
                    files.append(file)
    return files


def demo_BoundedModule():
    # Demo BoundedModule
    bound = BoundedModule()

    bound.register_probability_parameter('prob', [0, 1])
    print(bound.prob)

    bound.prob = [1, 0]
    print(bound.prob)

    bound.prob = [0.2, 0.8]
    print(bound.prob)

    bound.register_bounded_parameter('bounded', [0, 0.5, 1], 0, 1)
    print(bound.bounded)

    bound.bounded = [1, 0.7, 0]
    print(bound.bounded)

    bound.register_circular_parameter('circular', [0, 0.5, 1, 1.5, 2], 0, 1)
    print(bound.circular)

    bound.circular = [1, 0.7, 0, -0.5]
    print(bound.circular)

    test = TestBoundedModule()
    test.test_bounded_data2param2data()
    test.test_prob_data2param2data()
    test.test_register_bounded_parameter()
    test.test_register_probability_parameter()
    test.test_register_circular_parameter()

    # %%
    res = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestBoundedModule).run(
        unittest.TestResult())
    print('errors:')
    print(res.errors)
    print('failures:')
    print(res.failures)


class DemoBoundedModule(BoundedModule):
    def __init__(self):
        super().__init__()
        self.th_fixed_all = BoundedParameter(
            [1., 2.], [1., 2.], [1., 2.]
        )
        self.th_fixed_subset = BoundedParameter(
            [1., 2.], [-1., 2.], [2., 2.]
        )
        self.th_fixed_none = BoundedParameter(
            [1., 2.], [-1., -2.], [2., 4.]
        )

        params = [
            self.th_fixed_all,
            self.th_fixed_subset,
            self.th_fixed_none
        ]
        for param in params:
            print(param[:])
            print(param._param[:])

        state_dict = self.state_dict()
        self.load_state_dict(state_dict, False)
        pprint(state_dict)

        for param in params:
            print(param[:])
            print(param._param[:])

        print('--')

def ____Main____():
    pass


if __name__ == 'main':
    model = DemoBoundedModule()
    # demo_BoundedModule()


def optimize_scipy(
        model: BoundedModule,
        maxiter=400,
        verbose=True,
        kw_optim=(),
        kw_optim_option=(),
        kw_pyobj=(),
) -> (torch.Tensor, torch.Tensor, np.array):
    """

    :param model:
    :param maxiter:
    :param verbose:
    :param kw_optim: e.g., {'callback': lambda x, *args: print(x)}
    :param kw_optim_option:
    :param kw_pyobj: {'separate_loss_for_jac': True} to separately return
            loss_for_grad, loss = model()
        e.g., for REINFORCE
    :return: param_fit, loss, out
    """
    kw_optim = dict(kw_optim)
    kw_optim_option = dict(kw_optim_option)

    t_st = time.time()
    # with tqdm(total=maxiter) as pbar:
    #     def verbose(xk):
    #         pbar.update(1)

    model.train()
    obj = PyTorchObjective(model, **dict(kw_pyobj))
    out = scioptim.minimize(
        obj.fun, obj.x0,
        jac=obj.jac,
        **{
            'method': 'BFGS',
            # method='L-BFGS-B',
            # callback=verbose,
            **kw_optim
        }, options={
            # 'gtol': 1e-16, # 12,
            'disp': True, 'maxiter': maxiter,
            # 'finite_diff_rel_step': 1e-6,
            **kw_optim_option
        },
    )
    model.load_state_dict(obj.unpack_parameters(out['x']))
    model.eval()
    if obj.separate_loss_for_jac:
        _, loss_eval = model()
    else:
        loss_eval = model()
    loss_eval = npy(loss_eval)
    out['fun_train'] = out['fun']
    out['fun_eval'] = loss_eval
    out['fun'] = loss_eval

    t_en = time.time()
    t_el = t_en - t_st
    print('Time elapsed: %1.3g s\n' % t_el)

    # --- Results
    param = out['x']  # obj.parameters()
    se_param = np.sqrt(np.diag(out['hess_inv']))
    out['x_param_vec'] = out['x']
    out['se_param_vec'] = se_param

    for k in ['x', 'se']:
        out[k + '_param_dict'] = model.param_vec2dict(out[k + '_param_vec'])

    out['x_value_vec'] = npy(model.param_vec2value_vec(param))
    out['lb_value_vec'] = npy(model.param_vec2value_vec(param - se_param))
    out['ub_value_vec'] = npy(model.param_vec2value_vec(param + se_param))

    for k in ['x', 'lb', 'ub']:
        out[k + '_value_dict'] = model.param_vec2dict(out[k + '_value_vec'])

    # NOTE: somehow this is necessary to set params to the fitted state
    model.load_state_dict(obj.unpack_parameters(out['x']))
    state_dict0 = model.state_dict()
    state_dict = {}
    for k, v in state_dict0.items():
        if torch.is_tensor(v):
            state_dict[k] = npt.dclone(v)
        else:
            state_dict[k] = v
        # try:
        #     state_dict[k] = npt.dclone(v)
        # except AttributeError:
        #     print(k)
        #     print(v)
        #     raise RuntimeError()
    out['state_dict'] = state_dict

    # def vec2str(v) -> str:
    #     return ', '.join(['%g' % v1 for v1 in v])

    if verbose:
        for k in out['x_value_dict'].keys():
            x1, lb1, ub1 = (
                out['x_value_dict'][k],
                out['lb_value_dict'][k],
                out['ub_value_dict'][k]
            )
            for i, (x11, lb11, ub11) in enumerate(zip(
                    x1, lb1, ub1
                    # x1.flatten(), lb1.flatten(), ub1.flatten()
            )):
                if x11.ndim == 0:
                    print('%s[%d]: %g (%g - %g)\n' % (k, i, x11, lb11, ub11))
    return out['x_value_vec'], loss_eval, out