#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:22:57 2018

@author: yulkang
"""

import numpy as np
import torch

#%%
class ConstrainedModule(torch.nn.Module):
    def __init__(self):
        self.th = {}
        self.th0 = {}
        self.th_lb = {}
        self.th_ub = {}
        self.th_kind = {}
        
        self._th = {} # Auxiliary variable that gets transformed into th
        
        # Each element of constr is a tensor
        self.constr = []
        # constr_kind: 'A', 'Aeq', 'c', 'ceq'
        self.constr_kind = []
        
    def get_th_names(self):
        return self.th.keys()
    th_names = property(get_th_names)
    
    def __getattr__(self, name):
        if name != 'th' and hasattr(self, 'th') \
                and type(self.th) is dict and name in self.th.keys():
            return self.get_th(name)
        else:
            return super(ConstrainedModule,self).__getattr__(name)
            
    def __setattr__(self, name, value):
        if name != 'th' and hasattr(self, 'th') \
                and type(self.th) is dict and name in self.th.keys():
            self.set_th(name, value)
        else:
            super(ConstrainedModule,self).__setattr__(name, value)
    
    def add_param(self, name, th0, lb=None, ub=None, kind=None):
        assert type(name) is str, 'name must be str!'
        
        self.th[name] = torch.tensor(th0)
        
        if kind is None:
            if lb is None and ub is None:
                kind = 'uncon'
            elif lb is None:
                kind = 'ub'
            elif ub is None:
                kind = 'lb'
            else:
                kind = 'lbub'
                
        self.th_kind[name] = kind
        self._th[name] = torch.tensor(th0, requires_grad=True)
        
        self.th0[name] = torch.tensor(th0)
        self.th_lb[name] = torch.tensor(lb)
        self.th_ub[name] = torch.tensor(ub)
        
    def add_param_prob(self, name, shape, dim=0):
        shape1 = shape[dim] + 1
#        k = 
        
#        self.register_parameter([]
    
    def add_params(self, params):
        for param in params:
            if type(param) is list:
                self.add_param(*tuple(param))
            elif type(param) is dict:
                self.add_param(**param)
            else:
                raise ValueError(
                        'params must be a list of list or dict!')
                
    def get_th(self, name):
        if self.th_kind[name] == 'lbub':
            x = self._th[name]
            p = 1. / (1 + torch.exp(x))
            return p * (self.th_ub[name] - self.th_lb[name]) + self.th_lb[name]
        
    def set_th(self, name):
        if self.th_kind[name] == 'lbub':
            lb = self.th_lb[name]
            ub = self.th_ub[name]
            th = self.th[name]

            p = th.Tensor.new_empty(th.shape)
            incl = lb != ub

            p[~incl] = .5
            p[incl] = (th - lb) / (ub - lb)
            
            self._th[name] = torch.log(p / (1 - p))            
                
#    def add_constraint(self, kind, names, *args):
#        if kind == 'A':
#            self.__add_constr_lin_neq(names, *args)
#        else:
#            raise ValueError('kind=%s is not implemented!'
#                             % kind)
#        self.constr_kind.append(kind)
#            
#    def __add_constr_lin_neq(self, names, gain, offset):
#        constr = torch.tensor(0., requires_grad=True)
#        n = len(names)
#        for ii in range(n):
#            constr += self.th[names[ii]] * torch.tensor(gain[ii])
#            
#        constr -= torch.tensor(offset)
#        constr = constr.sum()
#        
#        self.constr.append(constr)
#        
#    def get_cost(self):
#        for name in self.th_names:
#            torch.clamp(self.th[name], 
#                        min=self.th_lb[name],
#                        max=self.th_ub[name])
          
#%%
            
        
