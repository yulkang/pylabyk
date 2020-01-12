from collections import OrderedDict as odict

import torch
from torch import nn
from torch.nn import functional as F


#%% Options
"""
1. Use OverriddenParameter
Pros
: can assign to slice
: can use autocompletion
Cons
: need to use [:] to refer to the whole tensor

2. Use BoundedModule.register_()
Pros
: ?
Cons
: cannot assign to slice without using setslice()
"""

#%% Bounded parameters (under construction)
# this is for better autocompletion, etc.
# it should run backward() on the variables hidden in it.
# See torch.tensor.Tensor for more.

class OverriddenParameter(nn.Module):
    """
    In operations requiring the whole parameter, use param[:] instead of
    param itself. For example, use result = param[:] + 1. Use this
    instead of param.v, because param[:] allows the code to
    work when param is substituted with another tensor.
    """

    def __init__(self, epsilon=1e-6, *args, **kwargs):
        super().__init__()
        self.epsilon = epsilon

    @property
    def v(self):
        return self._param2data(self._param)

    @v.setter
    def v(self, data):
        self._param = self._data2param(data)

    def __getitem__(self, key):
        return self.v[key]

    def __setitem__(self, key, data):
        self._param[key] = self._data2param(data)

    def _param2data(self, param):
        raise NotImplementedError()

    def _data2param(self, data):
        raise NotImplementedError()

    def __str__(self):
        indent = '  '
        from textwrap import TextWrapper as TW
        def f(s, **kwargs):
            return TW(**kwargs).fill(s)
        return str((self._param2data(self._param), type(self).__name__))


class BoundedParameter(OverriddenParameter):
    def __init__(self, data, lb=0., ub=1., **kwargs):
        super().__init__(**kwargs)
        self.lb = lb
        self.ub = ub
        self._param = nn.Parameter(self._data2param(data))
        assert self._param.ndim > 0, \
            'scalar unallowed to enforce use of [:]'

    def _data2param(self, data):
        lb = self.lb
        ub = self.ub
        data = enforce_float_tensor(data)
        if lb is None and ub is None:# Unbounded
            return data
        elif lb is None:
            data[data > ub - self.epsilon] = ub - self.epsilon
            return torch.log(ub - data)
        elif ub is None:
            data[data < lb + self.epsilon] = lb + self.epsilon
            return torch.log(data - lb)
        else:
            data[data < lb + self.epsilon] = lb + self.epsilon
            data[data > ub - self.epsilon] = ub - self.epsilon
            p = (data - lb) / (ub - lb)
            return torch.log(p) - torch.log(1. - p)

    def _param2data(self, param):
        lb = self.lb
        ub = self.ub
        param = enforce_float_tensor(param)
        if lb is None and ub is None: # Unbounded
            return param
        elif lb is None:
            return ub - torch.exp(param)
        elif ub is None:
            return lb + torch.exp(param)
        else:
            return (1 / (1 + torch.exp(-param))) * (ub - lb) + lb


class ProbabilityParameter(OverriddenParameter):
    def __init__(self, prob, probdim=0, **kwargs):
        super().__init__(**kwargs)
        self.probdim = probdim
        self._param = nn.Parameter(self._data2param(prob))
        assert self._param.ndim > 0, \
            'scalar unallowed to enforce use of [:]'

    def _data2param(self, prob):
        probdim = self.probdim
        prob = enforce_float_tensor(prob)

        prob[prob < self.epsilon] = self.epsilon
        prob[prob > 1. - self.epsilon] = 1. - self.epsilon
        prob = prob / torch.sum(prob, dim=probdim, keepdim=True)

        return torch.log(prob)

    def _param2data(self, conf):
        return F.softmax(enforce_float_tensor(conf), dim=self.probdim)


class CircularParameter(OverriddenParameter):
    def __init__(self, data, lb=0., ub=1., **kwargs):
        super().__init__(**kwargs)
        data = enforce_float_tensor(data)
        self.lb = lb
        self.ub = ub
        self._param = nn.Parameter(self._data2param(data))
        assert self._param.ndim > 0, \
            'scalar unallowed to enforce use of [:]'

    def _data2param(self, data):
        data = enforce_float_tensor(data)
        return (data - self.lb) / (self.ub - self.lb) % 1.

    def _param2data(self, param):
        param = enforce_float_tensor(param)
        return param * (self.ub - self.lb) + self.lb


#%% Bounded fit class
class LookUp(object):
    pass

class BoundedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._params_bounded = {} # {name:(lb, ub)}
        self._params_probability = {} # {name:probdim}
        self._params_circular = {} # {name:(lb, ub)}
        self.params_bounded = LookUp()
        self.params_probability = LookUp()
        self.params_circular = LookUp()
        self.epsilon = 1e-6

    def setslice(self, name, index, value):
        v = self.__getattr__(name)
        v[index] = value
        self.__setattr__(name, v)

    # Bounded parameters whose elements are individually bounded
    def register_bounded_parameter(self, name, data, lb=0., ub=1.):
        data = enforce_float_tensor(data)
        self._params_bounded[name] = {'lb':lb, 'ub':ub}
        param = self._bounded_data2param(data, lb, ub)
        self.register_parameter('_bounded_' + name, nn.Parameter(param))
        self.params_bounded.__dict__[name] = None # just a reminder

    def _bounded_data2param(self, data, lb=0., ub=1.):
        data = enforce_float_tensor(data)
        if lb is None and ub is None:# Unbounded
            return data
        elif lb is None:
            data[data > ub - self.epsilon] = ub - self.epsilon
            return torch.log(ub - data)
        elif ub is None:
            data[data < lb + self.epsilon] = lb + self.epsilon
            return torch.log(data - lb)
        else:
            data[data < lb + self.epsilon] = lb + self.epsilon
            data[data > ub - self.epsilon] = ub - self.epsilon
            p = (data - lb) / (ub - lb)
            return torch.log(p) - torch.log(1. - p)

    def _bounded_param2data(self, param, lb=0., ub=1.):
        param = enforce_float_tensor(param)
        if lb is None and ub is None: # Unbounded
            return param
        elif lb is None:
            return ub - torch.exp(param)
        elif ub is None:
            return lb + torch.exp(param)
        else:
            return (1 / (1 + torch.exp(-param))) * (ub - lb) + lb

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

    # Get/set
    def __getattr__(self, item):
        if item[0] == '_':
            return super(BoundedModule, self).__getattribute__(item)

        # if item in ['_modules', '_params_bounded', '_params_probability',
        #             '_params_circular']:
        #     try:
        #         return super(BoundedModule, self).__getattribute__(item)
        #     except:
        #         return {}

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
                    [str((name, v._param2data(v._param)))]
                )
            else:
                l += indent(['%s (%s)' % (name, type(v).__name__)]) + \
                    indent(v.__str__().split('\n'))

        return '\n'.join(l)


def enforce_float_tensor(v):
    """
    :type v: torch.Tensor, np.ndarray
    :rtype: torch.DoubleTensor, torch.FloatTensor
    """
    if not torch.is_tensor(v):
        return torch.tensor(v, dtype=torch.get_default_dtype())
    elif not torch.is_floating_point(v):
        return v.float()
    else:
        return v

#%% Test bounded module
import unittest
class TestBoundedModule(unittest.TestCase):
    def test_bounded_data2param2data(self):
        def none2str(v):
            if v is None:
                return 'None'
            else:
                return '%d' % v

        bound = BoundedModule()
        data = torch.zeros(2, 3)
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

    def test_register_bounded_parameter(self):
        def none2str(v):
            if v is None:
                return 'None'
            else:
                return '%d' % v

        bound = BoundedModule()
        data = torch.zeros(2, 3)
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
        data = torch.zeros(2, 3)
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
                torch.zeros(1, 3) + p,
                torch.ones(1, 3) - p
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
                torch.zeros(1, 3) + p,
                torch.ones(1, 3) - p
                ], dim=0)
            bound.register_probability_parameter('prob', data)
            data1 = bound.prob

            min_err = torch.min(torch.abs(data1 - data))
            self.assertTrue(
                min_err <= bound.epsilon * 10,
                'error of %g with p=%g' % (
                    min_err, p)
            )

#%% Demo bounded module
if __name__ == 'main':
    #%%
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

    #%%
    res = unittest.defaultTestLoader.loadTestsFromTestCase(TestBoundedModule).run(
        unittest.TestResult())
    print('errors:')
    print(res.errors)
    print('failures:')
    print(res.failures)

    #%%