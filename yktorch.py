from collections import OrderedDict as odict

import torch
from torch import nn
from torch.nn import functional as F

#%% Bounded fit class
class BoundedModule(nn.Module):
    def __init__(self):
        super(BoundedModule, self).__init__()
        self._params_bounded = {} # {name:(lb, ub)}
        self._params_probability = {} # {name:probdim}
        self.epsilon = 1e-6

    def setslice(self, name, index, value):
        v = self.__getattr__(name)
        v[index] = value
        self.__setattr__(name, v)

    # Bounded parameters whose elements are individually bounded
    def register_bounded_parameter(self, name, data, lb=0, ub=1):
        data = enforce_float_tensor(data)
        self._params_bounded[name] = {'lb':lb, 'ub':ub}
        param = self.bounded_data2param(data, lb, ub)
        self.register_parameter('_bounded_' + name, nn.Parameter(param))

    def bounded_data2param(self, data, lb=0, ub=1):
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
            return torch.log(p) - torch.log(1 - p)

    def bounded_param2data(self, param, lb=0, ub=1):
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
                                nn.Parameter(self.prob2conf(prob)))

    def prob2conf(self, prob, probdim=0):
        prob = enforce_float_tensor(prob)
        incl = prob < self.epsilon

        prob[prob < self.epsilon] = self.epsilon
        prob[prob > 1 - self.epsilon] = 1 - self.epsilon
        prob = prob / torch.sum(prob, dim=probdim, keepdim=True)

        # ndim = prob.ndimension()
        # ix_conf = []
        # for dim in range(ndim):
        #     if dim == probdim:
        #         ix_conf.append(torch.arange(prob.shape[dim] - 1))
        #     else:
        #         ix_conf.append(torch.arange(prob.shape[dim]))
        # ix_conf = tuple(ix_conf)
        #
        # conf = torch.log(prob[ix_conf])
        # return conf

        return torch.log(prob)

    def conf2prob(self, conf, probdim=0):
        # ndim = conf.ndimension()
        # ix_conf = []
        # for dim in range(ndim):
        #     if dim == probdim:
        #         ix_conf.append(torch.arange(conf.shape[dim] + 1))
        #     else:
        #         ix_conf.append(torch.arange(conf.shape[dim]))
        # ix_conf = tuple(ix_conf)
        #
        # return prob

        return F.softmax(enforce_float_tensor(conf), dim=probdim)

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
        if item in ['_params_bounded', '_params_probability']:
            try:
                return super(BoundedModule, self).__getattribute__(item)
            except:
                return {}

        if hasattr(self, '_params_bounded'):
            _params_bounded = self.__dict__['_params_bounded']
            # _params_bounded = super().__getattribute__(
            #     '_params_bounded'
            # )
            if item in _params_bounded.keys():
                info = _params_bounded[item]
                param = super().__getattr__(
                    '_bounded_' + item)
                return self.bounded_param2data(param, **info)

        if hasattr(self, '_params_probability'):
            _params_probability = self.__dict__['_params_probability']
            # _params_probability = super().__getattribute__(
            #     '_params_probability'
            # )
            if item in _params_probability.keys():
                info = _params_probability[item]
                param = super().__getattr__(
                    '_conf_' + item)
                return self.conf2prob(param, **info)

        return super().__getattr__(item)

    def __setattr__(self, item, value):
        if item in ['_params_bounded', '_params_probability']:
            self.__dict__[item] = value

        if hasattr(self, '_params_bounded'):
            _params_bounded = self._params_bounded
            if item in _params_bounded.keys():
                info = _params_bounded[item]
                param = self.bounded_data2param(value, **info)
                super().__setattr__(
                    '_bounded_' + item, nn.Parameter(param)
                )
                return

        if hasattr(self, '_params_probability'):
            _params_probability = self._params_probability
            if item in _params_probability:
                info = _params_probability[item]
                param = self.prob2conf(value, **info)
                super().__setattr__(
                    '_conf_' + item, nn.Parameter(param)
                )
                return

        return super().__setattr__(item, value)

def enforce_float_tensor(v):
    if not torch.is_tensor(v) or not torch.is_floating_point(v):
        return torch.tensor(v, dtype=torch.get_default_dtype())
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

                param = bound.bounded_data2param(data, lb, ub)
                data1 = bound.bounded_param2data(param, lb, ub)
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

    def test_prob_data2param2data(self):
        bound = BoundedModule()
        for p in [0, 0.5, 1]:
            data = torch.cat([
                torch.zeros(1, 3) + p,
                torch.ones(1, 3) - p
                ], dim=0)
            param = bound.prob2conf(data)
            data1 = bound.conf2prob(param)

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

    bound.register_probability_parameter('prob', [0.2, 0.8])
    print(bound.prob)

    bound.prob = [0.7, 0.3]
    print(bound.prob)

    bound.register_bounded_parameter('bounded', [0, 0.5, 1], 0, 1)
    print(bound.bounded)

    bound.bounded = [1, 0.7, 0]
    print(bound.bounded)

    test = TestBoundedModule()
    test.test_bounded_data2param2data()
    test.test_prob_data2param2data()
    test.test_register_bounded_parameter()
    test.test_register_probability_parameter()

    #%%
    res = unittest.defaultTestLoader.loadTestsFromTestCase(TestBoundedModule).run(
        unittest.TestResult())
    print('errors:')
    print(res.errors)
    print('failures:')
    print(res.failures)