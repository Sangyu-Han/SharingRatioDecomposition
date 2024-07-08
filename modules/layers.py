import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import warnings
from collections import OrderedDict, abc as container_abcs
from itertools import chain, islice
import operator
from torch import Tensor
from einops import rearrange
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from util import *
import copy
__all__ = ['Add', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential' , 'Clone']

       
    


def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())


class ReLU(nn.ReLU):
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        self.eps = 1e-20
    
    def save_static_plus_gradients(self, bias_gradients):
        self.static_plus_gradients = bias_gradients

    def save_static_plus(self,bias):
        self.static_plus = bias

    def save_static_minus_gradients(self, bias_gradients):
        self.static_minus_gradients = bias_gradients 

    def save_static_minus(self,bias):
        self.static_minus = bias 
    
    def get_static_cont(self):
        cont = self.static_plus * self.static_plus_gradients + self.static_minus * self.static_minus_gradients
        return cont
    
    def choice_cal(self,input,o,inference_pkg,residual_input=None,total_bias=None):

        if isinstance(o.layer_list[-1],nn.MaxPool2d) and isinstance(o.layer_list[1],nn.BatchNorm2d):
            simmap_total = cal_simmap_mcbr(input,o,inference_pkg,cur_layer_name=self.name)
        elif isinstance(o.layer_list[-1],nn.MaxPool2d) and isinstance(o.layer_list[1],nn.BatchNorm2d) == False:
            simmap_total = cal_simmap_mcr(input,o,inference_pkg,cur_layer_name=self.name)
        elif len(o.layer_list) == 3 and isinstance(o.layer_list[1],nn.BatchNorm2d):
            simmap_total = cal_simmap_cbr(input,o,inference_pkg,cur_layer_name=self.name,residual_input=residual_input,total_bias=total_bias)
        else:
            simmap_total = cal_simmap_cr(input,o,inference_pkg,cur_layer_name=self.name)
        return simmap_total

    
    
    def simprop2(self,o,inference_pkg,residual_input = None,total_bias = None):
        """
        only works for conv-bn-relu seq layer
        """
        if o.target_layer_name == self.name: # if we want to build ERF of specific hidden layer
            o.simmap = o.external_simmap
            o.layer_list = []
            o.layer_list.append(self)
            o.last_layer_name = self.name
            return o
        
        if len(o.layer_list) == 0: # first layer of the block
            o.layer_list.append(self)
            o.last_layer_name = self.name
            return o

        
        
        input = inference_pkg.output_dict[self.name].to(inference_pkg.device)
        if inference_pkg.probe == False:
            simmap_total = self.choice_cal(input,o,inference_pkg,residual_input=residual_input,total_bias = total_bias)
        else:
            simmap_total = self.choice_cal_probe(input,o,inference_pkg)
        o.simmap = simmap_total
        inference_pkg.simmaps[self.name] = simmap_total.cpu()
        o.layer_list = [self] # initialize layer list
        o.last_layer_name = self.name
        #TODO : 다른 레이어도 simprop2 구현. bedofnail 잘 작동하는지 보고 원래 simprop의 디버깅하게 구현.
        return o
        




        
    def forward(self, input: Tensor,inference_pkg) -> Tensor:
        
        input_copy = input.clone()
        output = F.relu(input, inplace=self.inplace)


        if inference_pkg.remember_input_graph:
            inference_pkg.input_dict[self.name] = input_copy
            inference_pkg.output_dict[self.name] = output.clone() # must remember what the block input was
        else:
            inference_pkg.input_dict[self.name] = input_copy.detach().cpu()
            inference_pkg.output_dict[self.name] = output.clone().detach().cpu() 
        return output
    def decomp_forward(self,input,inference_pkg,bias_shrink = None):
        relu_output = inference_pkg.output_dict[self.name]
        binary_output = (relu_output > 0)
        output = input * binary_output.to(inference_pkg.device)

        return output
    
    
        
    

class Dropout(nn.Dropout):
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        
        self.is_decomposition = False

    def forward(self, input: Tensor, inference_pkg) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)
    
    def decomp_forward(self,input,inference_pkg,bias_shrink=None):
        return input

    

class MaxPool2d(nn.MaxPool2d):
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        self.output = None
        self.h_indice = None
        self.w_indice = None
        self.c_indice = None
        
        
        

    def forward(self, input: Tensor,inference_pkg):
        
        
        
        
        output, indices = F.max_pool2d(input, self.kernel_size, self.stride,
                        self.padding, self.dilation, ceil_mode=self.ceil_mode,
                        return_indices=self.return_indices)
        
        if inference_pkg.remember_input_graph:
            inference_pkg.input_dict[self.name] = input.clone()
            inference_pkg.output_dict[self.name] = output.clone() # must remember what the block input was
        else:
            inference_pkg.input_dict[self.name] = input.clone().detach().cpu()
            inference_pkg.output_dict[self.name] = output.clone().detach().cpu() 
        
        inference_pkg.pool_indices_dict[self.name] = indices
        
        return output

    def simprop(self, simmap, inference_pkg):
        map_h = simmap.shape[0]
        map_w = simmap.shape[1]
        result_h = 2 * map_h
        result_w = 2 * map_w
        result_map = torch.zeros((map_h,map_w,result_h,result_w))
        for h in range(map_h):
            for w in range(map_w):
                res_h = 2 * h
                res_w = 2 * w
                
                result_vector = inference_pkg.output_dict[self.name][0,:,h,w]
                
                for i in (-1,0,1):
                    for j in (-1,0,1):
                        if res_h + j < 0 or res_w + i < 0 or res_h + j >= result_h or res_w + i>= result_w:
                            continue

                        target_vector = inference_pkg.output_dict['resnet50.relu'][0,:,res_h + j,res_w + i]
                        bin = (target_vector == result_vector)
                        out = target_vector * bin
                        por_ij = torch.dot(out,result_vector) / torch.norm(result_vector) ** 2
                        
                        result_map[h,w,res_h+j,res_w + i] += por_ij
                
                result_map[h,w] = simmap[h,w] * result_map[h,w]
        
        res = result_map.sum(dim=1)
        res = res.sum(dim=0)
        inference_pkg.simmaps[self.name] = res #for debug
        return res
 
    def simprop2(self,o,inference_pkg):
   
        if len(o.layer_list ) == 0:
       
        


            input = inference_pkg.input_dict[self.name].to(inference_pkg.device)
            simmap_total = torch.zeros((input.shape[0],input.shape[-2],input.shape[-1]),requires_grad=True).to(inference_pkg.device) # simmap : (N,H,W)
            (N,H,W) = simmap_total.shape
            last_out = inference_pkg.output_dict[self.name].to(inference_pkg.device)
            
            kernel_size = self.kernel_size
            padding = self.padding
            stride = self.stride
            scale_factor = (stride,stride)
            padding = (padding,padding,padding,padding)
            for _h in range(kernel_size):
                for _w in range(kernel_size):
                    offset = (_h,_w)
                    mask = torch.ones_like(o.simmap)
                    mask = bed_of_nail_upsample2d(mask,scale_factor=scale_factor,offset=offset,padding=padding) # 여기까진 맞는듯..
                    input_pad = F.pad(input,padding,'constant',0) # 미리 패딩해야함
                    input_pad = mask.unsqueeze(0) * input_pad
                    out = self.decomp_forward(input_pad,self.kernel_size,self.stride,padding=0)
                    cos = nn.CosineSimilarity(dim=1, eps=1e-12)
                    #calculate simmap for given position
                    simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12) # the contribution
                    simmap *= o.simmap
                    simmap = bed_of_nail_upsample2d(simmap,scale_factor=scale_factor,offset=offset,padding=padding) # simmap을 원래대로 돌릴때...
                    simmap_total += simmap[:,self.padding:H+self.padding,self.padding:W + self.padding]
            
            o.simmap = simmap_total
            inference_pkg.simmaps[self.name+'in'] = simmap_total.cpu()
            
            return o
        else:
            o.layer_list.append(self)
            return o


    def decomp_forward(self,input,inference_pkg,hw = None):
        batch_size = input.shape[0]
        h_input = input.shape[2]
        w_input = input.shape[3]
        c_input = input.shape[1]

        indices = inference_pkg.pool_indices_dict[self.name].clone()
        h_indice = indices.shape[-2]
        w_indice = indices.shape[-1]
        c_indice = indices.shape[-3]
        
        indices = torch.broadcast_to(indices,(batch_size,c_indice,h_indice,w_indice))
        input = rearrange(input, 'N C H W -> N C (H W)')
        indices = rearrange(indices, 'N C H W -> N C (H W)')

        output = torch.gather(input,2,indices)
        output = rearrange(output,'N C (H W) -> N C H W',H=h_indice,W=w_indice)
        if inference_pkg.target_layer_name == self.name:
            target = output[:,:,inference_pkg.target_layer_hw[0],inference_pkg.target_layer_hw[1]].clone().detach().cpu()
            inference_pkg.target_tensors.append(target)

        return output
    
    
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        

    def forward(self, input: Tensor,inference_pkg) -> Tensor:
        
        
        output = F.adaptive_avg_pool2d(input, self.output_size)
        inference_pkg.output_dict[self.name] = output.clone().detach().cpu()
        if inference_pkg.remember_input_graph:
            inference_pkg.input_dict[self.name] = input.clone() # must remember what the block input was
        else:
            inference_pkg.input_dict[self.name] = input.clone().detach().cpu() 
        
        return output
    def decomp_forward(self,input,inference_pkg):
    
        output = F.adaptive_avg_pool2d(input, self.output_size)

        if inference_pkg.target_layer_name == self.name:
            target = output[:,:,inference_pkg.target_layer_hw[0],inference_pkg.target_layer_hw[1]].clone().detach().cpu()
            inference_pkg.target_tensors.append(target)

        return output
   
    def simprop2(self,o,inference_pkg,rule = None): # 이거 수정해야함.. classifier니까 classifier로 새로 파야함
        output_size = self.output_size # for now only support outputsize = (1,1)
        # only usable in [avgpool, fc]
        input = inference_pkg.input_dict[self.name].to(inference_pkg.device)
        simmap_total = torch.zeros((input.shape[0],input.shape[-2],input.shape[-1])).to(inference_pkg.device) # simmap : (N,H,W)
        (N,H,W) = simmap_total.shape
        linear = o.layer_list[0]
        C = inference_pkg.output_dict[linear.name].shape[-1]
        if o.model_out == None:
            last_out = torch.zeros(N,C,device=inference_pkg.device)
        else:
            last_out = o.model_out
        
        n = H * W
        input = rearrange(input, 'N C H W -> N H W C ')
        input = input / n
        out = linear.decomp_forward(input,inference_pkg,bias_shrink = n)
        out = rearrange(out, 'N H W C -> N C H W ')
        last_out = last_out.view((N,C,1,1))
        cos = nn.CosineSimilarity(dim=1, eps=1e-12)
            #calculate simmap for given position
        simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12) # the contribution
        inference_pkg.simmaps['true_avgpool'] = simmap.clone().detach().cpu()
        simmap = o.cav_contrastive_rule(simmap)

        inference_pkg.simmaps[self.name] = simmap.clone().detach().cpu()
        
        
        o.simmap = simmap
        o.layer_list = []

        return o



class AvgPool2d(nn.AvgPool2d):
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        

class Add():
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        
        self.is_decomposition = False
    def forward(self, inputs,inference_pkg):
        return torch.add(*inputs)

    def decomp_forward(self,inputs,inference_pkg,bias_shrink=1):
        return torch.add(*inputs)
    
    def simprop2(self,o,inference_pkg):
        
        o1 = SimmapObj()
        o1.external_simmap = o.external_simmap
        o1.last_layer_name = o.last_layer_name
        o1.layer_list = o.layer_list[:] # shallow copy
        o1.target_layer_name = o.target_layer_name
        o1.simmap = o.simmap.clone()
        o2 = SimmapObj()
        o2.external_simmap = o.external_simmap
        o2.last_layer_name = o.last_layer_name
        o2.layer_list = o.layer_list[:] # shallow copy
        o2.target_layer_name = o.target_layer_name
        o2.simmap = o.simmap.clone()
        return (o1,o2)

class Clone():
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        self.output = None
        self.is_decomposition = False

    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

class Sequential(nn.Sequential):
    def __init__(self, *args,name = ''):
        self.name = name
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                setattr(module,'name',self.name +'.'+ key)
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                setattr(module,'name',self.name +'.'+ str(idx))
                self.add_module(str(idx), module)
    
    def simprop2(self,o,inference_pkg):
        for m in reversed(self._modules.values()):
            
            o = m.simprop2(o,inference_pkg)
        return o
    def simprop2_layer1(self,o,inference_pkg,maxpool,relu):
        for m in reversed(self._modules.values()):
            if 'layer1.0' in m.name:
                o = m.simprop2_layer1(o,inference_pkg,maxpool,relu)
            else:
                o = m.simprop2(o,inference_pkg)
        return o

    def forward(self, input,inference_pkg):
        for module in self:
            input = module(input,inference_pkg)
        return input
    def decomp_forward(self, input,inference_pkg,bias_shrink=1):
        for module in self:
            input = module.decomp_forward(input,inference_pkg,bias_shrink)
            
        return input
class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        
        
        

    def forward(self, input: Tensor,inference_pkg) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        true_bias = self.bias - self.weight * self.running_mean / torch.sqrt(self.running_var) 
        inference_pkg.bn_true_bias_dict[self.name] = true_bias
        zoom = self.weight / torch.sqrt(self.running_var + self.eps)
        inference_pkg.bn_zoom_dict[self.name] = zoom

        output = F.batch_norm(
        input,
        # If buffers are not to be tracked, ensure that they won't be updated
        self.running_mean
        if not self.training or self.track_running_stats
        else None,
        self.running_var if not self.training or self.track_running_stats else None,
        self.weight,
        self.bias,
        bn_training,
        exponential_average_factor,
        self.eps,
        )
        inference_pkg.output_dict[self.name] = output.clone().detach().cpu()
        inference_pkg.weight_dict[self.name] = self.weight.clone().detach()
        inference_pkg.bias_dict[self.name] = self.bias.clone().detach()
        inference_pkg.running_mean_dict[self.name] = self.running_mean.clone().detach()
        inference_pkg.running_var_dict[self.name] = self.running_var.clone().detach()

        inference_pkg.sum_run_pls_bias[self.name] = F.batch_norm(
        torch.zeros_like(input),
        # If buffers are not to be tracked, ensure that they won't be updated
        self.running_mean
        if not self.training or self.track_running_stats
        else None,
        self.running_var if not self.training or self.track_running_stats else None,
        self.weight,
        self.bias,
        bn_training,
        exponential_average_factor,
        self.eps,
        ).clone().detach().cpu()
        return output

    def decomp_forward(self, input: Tensor,inference_pkg, bias_shrink = 1) -> Tensor:
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

       
        output = F.batch_norm(
        input,
        # If buffers are not to be tracked, ensure that they won't be updated
        self.running_mean / bias_shrink
        if not self.training or self.track_running_stats
        else None,
        self.running_var if not self.training or self.track_running_stats else None,
        self.weight,
        self.bias / bias_shrink,
        bn_training,
        exponential_average_factor,
        self.eps,
        )
        if inference_pkg.target_layer_name == self.name:
            target = output[:,:,inference_pkg.target_layer_hw[0],inference_pkg.target_layer_hw[1]].clone().detach().cpu()
            inference_pkg.target_tensors.append(target)

        return output    

    
    def simprop2(self,o,inference_pkg):
        if len(o.layer_list) == 1:
            o.last_layer_name = self.name
        o.layer_list.append(self)
        return o


class Linear(nn.Linear):
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        self.output = None

    
    def forward(self, input: Tensor,inference_pkg) -> Tensor:
        
        
        
        output = F.linear(input, self.weight, self.bias)
        inference_pkg.output_dict[self.name] = output.clone().detach().cpu()
        inference_pkg.bias_dict[self.name] = self.bias.clone().detach().cpu()
        return output
        
    def decomp_forward(self, input: Tensor,inference_pkg,bias_shrink = 1) -> Tensor:
      
        
        output = F.linear(input, self.weight, self.bias/bias_shrink)
        
        
        if inference_pkg.target_layer_name == self.name:
            target = output[:,:].clone().detach().cpu()
            inference_pkg.target_tensors.append(target)

            
        return output            

    def simprop2(self,o,inference_pkg):
        o.layer_list.append(self)
        return o

class Conv2d(nn.Conv2d):
    def __init__(self,*args,name='',**kwargs):
        super().__init__(*args,**kwargs)
        self.name = name
        self.output = None
        self.is_decomposition = False

    def forward(self, input: Tensor,inference_pkg) -> Tensor:
        
        inference_pkg.weight_dict[self.name] = self.weight.clone().detach().cpu()
        if self.bias != None:
            inference_pkg.bias_dict[self.name] = self.bias.clone().detach().cpu()
        output = self._conv_forward(input, self.weight, self.bias)
        inference_pkg.output_dict[self.name] = output.clone().detach().cpu()
        return output
    def decomp_forward(self, input: Tensor,inference_pkg,bias_shrink = 1) -> Tensor:
        if self.bias != None:
            output = self._conv_forward(input, self.weight, self.bias / bias_shrink)
        else:
            output = self._conv_forward(input, self.weight, self.bias)
        if inference_pkg.target_layer_name == self.name:
            target = output[:,:,inference_pkg.target_layer_hw[0],inference_pkg.target_layer_hw[1]].clone().detach().cpu()
            inference_pkg.target_tensors.append(target)

        return output
    
    def simprop2(self,o,inference_pkg):
        if len(o.layer_list) == 1:
            o.last_layer_name = self.name
        o.layer_list.append(self)
        return o

    
    