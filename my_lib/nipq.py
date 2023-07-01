from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np 
from collections import OrderedDict

beta = 1.


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):    
        return input.round()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output



        
@torch.jit.script
def lsq_forward(data, bit, alpha, sym:bool):    
    if sym:
        n_lv = 2 ** (bit.detach() - 1) - 1    
        data_q = F.hardtanh(data / alpha, -1., 1.) * n_lv
    else:
        n_lv = 2 ** bit.detach() - 1
        data_q = F.hardtanh(data / alpha, 0., 1.) * n_lv
        
    out = (data_q.round() + (data_q - data_q.detach())) * (alpha / n_lv)
    #out = RoundQuant.apply(data_q) * (alpha / n_lv)    
    return out



# sampling based 
@torch.jit.script
def noise_quant(data, bit, alpha, is_training, noise, sym:bool, is_stochastic=True, is_discretize=True):
    N_BIN = 256
    bit = 2 + torch.sigmoid(bit)*12

    #Stochastic Rounding
    if is_training and noise and is_stochastic :
        bit += (torch.rand_like(bit) - 0.5)
    
    if not is_training or is_discretize :
        bit = bit.round() + (bit - bit.detach())


    alpha = F.softplus(alpha) 
    lsq = lsq_forward(data, bit.round(), alpha, sym)
    
    if is_training and noise:     
        if sym:
            c1 = data >= alpha
            c2 = data <= -alpha     
            delta = alpha / (2**(bit - 1) - 1)
            
            with torch.no_grad():                
                diff = (lsq - data) / delta 
                sel = diff[torch.logical_not(torch.logical_or(c1, c2))]
                hist = torch.histc(sel, bins=N_BIN, min=-0.5, max=0.5)    
                
                noise = torch.multinomial(hist, data.numel(), True) + torch.rand_like(data.view(-1))               
                noise = (noise / N_BIN - 0.5).view(data.shape)
            return  torch.where(c1, alpha, torch.where(c2, -alpha, data + noise * delta))                                              
        else:
            c1 = data >= alpha
            c2 = data <= 0          
            delta = alpha / (2**bit - 1)                   
                             
            with torch.no_grad():                
                diff = (lsq - data) / delta
                sel = diff[torch.logical_not(torch.logical_or(c1, c2))]                
                hist = torch.histc(sel, bins=N_BIN, min=-0.5, max=0.5)               
                
                noise = torch.multinomial(hist, data.numel(), True) + torch.rand_like(data.view(-1))               
                noise = (noise / N_BIN - 0.5).view(data.shape)
            return  torch.where(c1, alpha, torch.where(c2, 0, data + noise * delta))
    else:
        return lsq   


class Q_ReLU(nn.Module):
    def __init__(self):
        super(Q_ReLU, self).__init__()
        self.quant = False
        self.noise = True
        self.bit = Parameter(torch.Tensor(1).zero_())
        self.alpha = Parameter(torch.Tensor(1).fill_(6)) 

        self.is_stochastic  = True
        self.is_discretize  = True


    def forward(self, x):
        if self.quant is False:
            return x        
        return noise_quant(x, self.bit, self.alpha, self.training, self.noise, False, self.is_stochastic, self.is_discretize)

    
class Q_Sym(nn.Module):
    def __init__(self):
        super(Q_Sym, self).__init__()
        self.quant = False
        self.noise = True
        self.bit = Parameter(torch.Tensor(1).zero_())
        self.alpha = Parameter(torch.Tensor(1).fill_(3)) 


        self.is_stochastic = True
        self.is_discretize = True

    # symmetric & zero-included quant
    # TODO: symmetric & zero-excluded quant 
    def forward(self, x):
        if self.quant is False:
            return x        
        return noise_quant(x, self.bit, self.alpha, self.training, self.noise, True, self.is_stochastic, self.is_discretize)


class Q_HSwish(nn.Module):
    def __init__(self):
        super(Q_HSwish, self).__init__()
        self.quant = False
        self.noise = True
        self.bit = Parameter(torch.Tensor(1).zero_())
        self.alpha = Parameter(torch.Tensor(1).fill_(6))


        self.is_stochastic = True
        self.is_discretize = True

    def forward(self, x):
        if self.quant is False:
            return x
        return noise_quant(x + 3/8, self.bit, self.alpha, self.training, self.noise, True, self.is_stochastic, self.is_discretize) - 3/8





class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, act_func=None, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.quant = False
        self.noise = True
        self.bit = Parameter(torch.Tensor(1).zero_())
        self.alpha = Parameter(torch.Tensor(1).fill_(1)) 
        #self.alpha = Parameter(torch.Tensor(self.out_channels, 1, 1, 1).fill_(1)) 
    

        self.is_stochastic = True
        self.is_discretize = True

    
    def _weight_quant(self):
        if self.quant is False:
            return self.weight
        return noise_quant(self.weight, self.bit, self.alpha, self.training, self.noise, True, self.is_stochastic, self.is_discretize)
        
        
    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)
            
        return F.conv2d(x, self._weight_quant(), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
    

class Q_Linear(nn.Linear):
    def __init__(self, *args, act_func=None, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.quant = False
        self.noise = True
        self.bit = Parameter(torch.Tensor(1).zero_())
        self.alpha = Parameter(torch.Tensor(1).fill_(1)) 
    

        self.is_stochastic = True
        self.is_discretize = True

    
    def _weight_quant(self):
        if self.quant is False:
            return self.weight
        return noise_quant(self.weight, self.bit, self.alpha, self.training, self.noise, True, self.is_stochastic, self.is_discretize)
        
        
    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)
            
        return F.linear(x, self._weight_quant(), self.bias)
    
    
def initialize(model, act=False, weight=False, noise=True, is_stochastic=True, is_discretize=True, fixed_bit=-1):
    for name, module in model.named_modules():
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)) and act:
            module.quant = True
            module.noise = noise

            module.is_stochastic = is_stochastic
            module.is_discretize = is_discretize

            if fixed_bit != -1 :
                bit = ( fixed_bit+0.00001 -2 ) / 12
                bit = np.log(bit/(1-bit))
                module.bit.data.fill_(bit)
                module.bit.requires_grad = False
            
            #module.bit.data.fill_(-2)

        if isinstance(module, (Q_Conv2d, Q_Linear)) and weight:
            module.quant = True
            module.noise = noise
            
            module.is_stochastic = is_stochastic
            module.is_discretize = is_discretize

            #module.bit.data.fill_(-2)

            if fixed_bit != -1 :
                bit = ( fixed_bit -2 ) / 12
                bit = np.log(bit/(1-bit))
                module.bit.data.fill_(bit)
                module.bit.requires_grad = False
            
            if noise:
                module.alpha.data[0] = module.weight.data.max()
                       

class QuantOps(object):
    initialize = initialize
    ReLU = Q_ReLU
    Sym = Q_Sym
    Conv2d = Q_Conv2d
    Linear = Q_Linear
    HSwish = Q_HSwish
    
