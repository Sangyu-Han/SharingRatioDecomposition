from typing import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from modules.layers import *
import os
#TODO : maxpool에서 오프셋이 전부 h,w가 바뀌어있다... 나중에고치지ㅏㅏ.....
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
class InferencePkg():
    def __init__(self):
        self.output_dict = OrderedDict()
        self.input_dict = OrderedDict()
        self.bias_dict = OrderedDict()
        self.weight_dict = OrderedDict()
        self.running_mean_dict = OrderedDict()
        self.running_var_dict = OrderedDict()
        self.bn_zoom_dict = OrderedDict()
        self.bn_true_bias_dict = OrderedDict()
        self.unpool_dict = OrderedDict()
        self.pool_indices_dict = OrderedDict()
        self.sum_run_pls_bias = OrderedDict()
        self.gamma_X_dict = OrderedDict()
        self.relu_stimulus_dict = OrderedDict()
        self.relu_gamma_dict = OrderedDict()
        self.relu_bias_claim = OrderedDict()
        self.simmaps = OrderedDict()
        self.target_tensors = []
        self.target_tensors_by_image = None
        self.target_tensors_by_bias = None
        self.target_tensors_by_running_mean = None
        self.device = torch.device('cpu')
        self.target_layer_turnon = None
        self.target_layer_name = None
        self.target_layer_hw = None
        self.remember_input_graph = False # used in explanation manipulation
        self.target_tensor_by_image = None 
        self.use_external_gamma = False
        self.probe = False
        self.probe_simmap = OrderedDict()
        self.probe_simmap_effective = OrderedDict()
        
class SimmapObj():
    def __init__(self,device='cpu'):
        self.layer_list = list()
        self.external_simmap = None
        self.target_layer_name = None # target layer name should be...으음..
        self.last_layer_name = None
        self.model_out = None
        self.simmap = None
        self.cav_contrastive_rule = lambda x:x
        self.device = device
        
    
    def simprop2_input(self,block_input,o,inference_pkg,residual_input = None,total_bias = None):
        """
        calculate simmap between block input and last layer
        """
        input = block_input

        if len(o.layer_list) == 3: #conb,bn,relu sequence

            simmap_total = cal_simmap_cbr(input,o,inference_pkg,residual_input=residual_input,total_bias=total_bias)
        else: #at residual

            simmap_total = cal_simmap_x(input,o,inference_pkg,residual_input=residual_input,total_bias=total_bias)

        o.simmap = simmap_total
        
        o.layer_list = [] # initialize layer list
        return o
def cal_simmap_input_cr(input,o,inference_pkg,cur_layer_name=None):
    simmap_total = torch.zeros((input.shape[0],input.shape[1],input.shape[-2],input.shape[-1]),requires_grad=inference_pkg.remember_input_graph).to(inference_pkg.device) # simmap : (N,H,W)
    (N,c_i,H,W) = simmap_total.shape
    last_out = inference_pkg.output_dict[o.layer_list[1].name].to(inference_pkg.device)
    (_,c,_,_) = last_out.shape
    scale_rate = 1
    for m in reversed(o.layer_list):
        if isinstance(m,nn.Conv2d):
            conv = m
            
        if isinstance(m,nn.ReLU):
            relu = m
    w = conv.weight.shape[-1]
    h = conv.weight.shape[-2]
    stride = conv.stride
    
    bias = conv.bias
    if bias == None:
        bias = torch.zeros((c),device=inference_pkg.device)
    padding = conv.padding
    conv_padding = (padding[0],padding[0],padding[1],padding[1])
    total_norm = torch.zeros((last_out.shape[0],1,last_out.shape[-2],last_out.shape[-1]),device=inference_pkg.device)
    for _c in range(c_i):
        i_frame = torch.zeros_like(input)
        i_frame[:,_c,:,:] = input[:,_c,:,:]
        for _w in range(w):
            for _h in range(h):
                weight = torch.zeros_like(conv.weight)
                weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
                if bias == None:
                    out = F.conv2d(i_frame,weight,bias,stride,padding)
                else:
                    out = F.conv2d(i_frame,weight,torch.zeros_like(bias),stride,padding)
                total_norm += torch.norm(out,dim=1,keepdim=True).detach()
    for _c in range(c_i):
        i_frame = torch.zeros_like(input)
        i_frame[:,_c,:,:] = input[:,_c,:,:]
        for _w in range(w):
            for _h in range(h):
                weight = torch.zeros_like(conv.weight)
                weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
                if bias == None:
                    out = F.conv2d(i_frame,weight,bias,stride,padding)
                else:
                    out = F.conv2d(i_frame,weight,torch.zeros_like(bias),stride,padding)
                _,c,H_c,W_c = out.shape
                out_norm = torch.norm(out,dim=1,keepdim=True).detach()
                bias_bunbae = out_norm / (total_norm + 1e-14)
                bias_contribution = bias.reshape(1,c,1,1) * bias_bunbae
                out += bias_contribution
                cos = nn.CosineSimilarity(dim=1, eps=1e-12)
                #calculate simmap for given position
                simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12) # the contribution
                simmap *= o.simmap
                simmap = bed_of_nail_upsample2d(simmap,scale_factor=stride,offset=(_h,_w), target_shape=(N,H,W))
                simmap = simmap[:,padding[0]:H+padding[0],padding[1]:H + padding[1]]

                simmap_total[:,_c,:,:] += simmap
    return simmap_total 
def cal_simmap_x(input,o,inference_pkg,cur_layer_name=None,residual_input = None, total_bias = None):
    last_out = inference_pkg.input_dict[o.last_layer_name].to(inference_pkg.device)
    scale_rate = 1
    cos = nn.CosineSimilarity(dim=1, eps=1e-12)
    true_bias = total_bias if total_bias != None else torch.zeros((1,input.shape[1],1,1),device=inference_pkg.device)
    total_norm = torch.zeros((last_out.shape[0],1,last_out.shape[-2],last_out.shape[-1]),device=inference_pkg.device)
    out_norm = torch.norm(input,dim=1,keepdim=True)
    total_norm += out_norm
    if residual_input != None:
        total_norm += torch.norm(residual_input,dim=1,keepdim=True)
    bias_bunbae = out_norm / total_norm
    bias_contribution = true_bias.reshape(1,input.shape[1],1,1) * bias_bunbae
    out = input
    out += bias_contribution
    #calculate simmap for given position
    simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12) # the contribution
    simmap *= o.simmap
    return simmap
    

def cal_simmap_cbr(input,o,inference_pkg,cur_layer_name=None,residual_input = None,total_bias = None):
    """
    input과 o.last_layer 사이 simmap을 계산
    일단 이건 conv-bn-relu만 가능
    """
    simmap_total = torch.zeros((input.shape[0],input.shape[-2],input.shape[-1]),requires_grad=inference_pkg.remember_input_graph).to(inference_pkg.device) # simmap : (N,H,W)
    (N,H,W) = simmap_total.shape
    last_out = inference_pkg.input_dict[o.layer_list[0].name].to(inference_pkg.device)
    scale_rate = 1
    for m in reversed(o.layer_list):
        if isinstance(m,nn.Conv2d):
            conv = m

        if isinstance(m,nn.BatchNorm2d):
            bn = m
            
        if isinstance(m,nn.ReLU):
            relu = m
    w = conv.weight.shape[-1]
    h = conv.weight.shape[-2]
    stride = conv.stride
    
    bias = conv.bias
    padding = conv.padding
    conv_padding = (padding[0],padding[0],padding[1],padding[1])
    true_bias = inference_pkg.bn_true_bias_dict[bn.name] if total_bias == None else total_bias
    total_norm = torch.zeros((last_out.shape[0],1,last_out.shape[-2],last_out.shape[-1]),device=inference_pkg.device)
    for _w in range(w):
        for _h in range(h):
            weight = torch.zeros_like(conv.weight)
            weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
            if bias == None:
                out = F.conv2d(input,weight,bias,stride,padding)
            else:
                out = F.conv2d(input,weight,torch.zeros_like(bias),stride,padding)
            out = bn.decomp_forward(out,inference_pkg,bias_shrink = torch.inf)
            total_norm += torch.norm(out,dim=1,keepdim=True)
    if residual_input != None:
        total_norm += torch.norm(residual_input,dim=1,keepdim=True)
    #현재는 conv-bn-relu순으로만있을때 가능..
    for _w in range(w):
        for _h in range(h):
            weight = torch.zeros_like(conv.weight)
            weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
            if bias == None:
                out = F.conv2d(input,weight,bias,stride,padding)
            else:
                out = F.conv2d(input,weight,bias / (w*h),stride,padding)
            _,c,H_c,W_c = out.shape
            out = bn.decomp_forward(out,inference_pkg,bias_shrink = torch.inf)
            out_norm = torch.norm(out,dim=1,keepdim=True)

            bias_bunbae = out_norm / total_norm
            bias_contribution = true_bias.reshape(1,c,1,1) * bias_bunbae
            out += bias_contribution
            cos = nn.CosineSimilarity(dim=1, eps=1e-12)
            #calculate simmap for given position
            simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12) # the contribution
            simmap *= o.simmap
            simmap = bed_of_nail_upsample2d(simmap,scale_factor=stride,offset = (_h,_w), padding= conv_padding)
            simmap = simmap[:,padding[0]:H+padding[0],padding[1]:H + padding[1]]

            simmap_total += simmap
    return simmap_total 

def cal_simmap_input_cr_per_neuron(input,o,inference_pkg,cur_layer_name=None):
    simmap_total = torch.zeros((input.shape[0],input.shape[1],input.shape[-2],input.shape[-1]),requires_grad=inference_pkg.remember_input_graph).to(inference_pkg.device) # simmap : (N,H,W)
    (N,c_i,H,W) = simmap_total.shape
    last_out = inference_pkg.output_dict[o.layer_list[1].name].to(inference_pkg.device)
    (_,c,_,_) = last_out.shape
    scale_rate = 1
    for m in reversed(o.layer_list):
        if isinstance(m,nn.Conv2d):
            conv = m
            
        if isinstance(m,nn.ReLU):
            relu = m
    w = conv.weight.shape[-1]
    h = conv.weight.shape[-2]
    stride = conv.stride
    
    bias = conv.bias
    if bias == None:
        bias = torch.zeros((c),device=inference_pkg.device)
    padding = conv.padding
    conv_padding = (padding[0],padding[0],padding[1],padding[1])
    total_norm = torch.zeros((last_out.shape[0],1,last_out.shape[-2],last_out.shape[-1]),device=inference_pkg.device)
    for _c in range(c_i):
        i_frame = torch.zeros_like(input)
        i_frame[:,_c,:,:] = input[:,_c,:,:]
        for _w in range(w):
            for _h in range(h):
                weight = torch.zeros_like(conv.weight)
                weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
                if bias == None:
                    out = F.conv2d(i_frame,weight,bias,stride,padding)
                else:
                    out = F.conv2d(i_frame,weight,torch.zeros_like(bias),stride,padding)
                total_norm += torch.norm(out,dim=1,keepdim=True).detach()
    for _c in range(c_i):
        i_frame = torch.zeros_like(input)
        i_frame[:,_c,:,:] = input[:,_c,:,:]
        for _w in range(w):
            for _h in range(h):
                weight = torch.zeros_like(conv.weight)
                weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
                if bias == None:
                    out = F.conv2d(i_frame,weight,bias,stride,padding)
                else:
                    out = F.conv2d(i_frame,weight,torch.zeros_like(bias),stride,padding)
                _,c,H_c,W_c = out.shape
                out_norm = torch.norm(out,dim=1,keepdim=True).detach()
                bias_bunbae = out_norm / (total_norm + 1e-14)
                bias_contribution = bias.reshape(1,c,1,1) * bias_bunbae
                out += bias_contribution
                cos = nn.CosineSimilarity(dim=1, eps=1e-12)
                #calculate simmap for given position
                simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12) # the contribution
                simmap *= o.simmap
                simmap = bed_of_nail_upsample2d(simmap,scale_factor=stride,offset=(_h,_w),padding=conv_padding)
                simmap = simmap[:,padding[0]:H+padding[0],padding[1]:H + padding[1]]

                simmap_total[:,_c,:,:] += simmap
    return simmap_total 

def cal_simmap_cr(input,o,inference_pkg,cur_layer_name=None):
    """
    input과 o.last_layer 사이 simmap을 계산
    conv-relu만 가능
    """
    simmap_total = torch.zeros((input.shape[0],input.shape[-2],input.shape[-1]),requires_grad=inference_pkg.remember_input_graph).to(inference_pkg.device) # simmap : (N,H,W)
    (N,H,W) = simmap_total.shape
    last_out = inference_pkg.input_dict[o.layer_list[0].name].to(inference_pkg.device)
    
    (N,c,out_H,out_W) = last_out.shape

    scale_rate = 1
    for m in reversed(o.layer_list):
        if isinstance(m,nn.Conv2d):
            conv = m
            
        if isinstance(m,nn.ReLU):
            relu = m
    w = conv.weight.shape[-1]
    h = conv.weight.shape[-2]
    stride = conv.stride
    
    bias = conv.bias
    if bias == None:
        bias = torch.zeros((c),device=inference_pkg.device)
    padding = conv.padding
    probe = False
    conv_padding = (padding[0],padding[1],padding[0],padding[1])
    total_norm = torch.zeros((last_out.shape[0],1,last_out.shape[-2],last_out.shape[-1]),device=inference_pkg.device)
    for _w in range(w):
        for _h in range(h):
            weight = torch.zeros_like(conv.weight)
            weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
            out = F.conv2d(input,weight,torch.zeros_like(bias),stride,padding)
            total_norm += torch.norm(out,dim=1,keepdim=True)
    for _w in range(w):
        for _h in range(h):
            weight = torch.zeros_like(conv.weight)
            weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
            out = F.conv2d(input,weight,torch.zeros_like(bias),stride,padding)
            out_norm = torch.norm(out,dim=1,keepdim=True).detach()
            bias_bunbae = out_norm / (total_norm + 1e-14)
            bias_contribution = bias.reshape(1,out.shape[1],1,1) * bias_bunbae
            out += bias_contribution
            cos = nn.CosineSimilarity(dim=1, eps=1e-12)
            simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12) # the contribution
            simmap *= o.simmap
            simmap = bed_of_nail_upsample2d(simmap,scale_factor=stride,offset = (_h,_w), padding= conv_padding)
            simmap = simmap[:,padding[0]:H+padding[0],padding[1]:H + padding[1]]
            
            
            simmap_total += simmap
    return simmap_total 


def cal_simmap_mcr(input,o,inference_pkg,cur_layer_name=None):
    """
    input과 o.last_layer 사이 simmap을 계산
    maxpool2d-conv-relu만 가능
    """
    simmap_total = torch.zeros((input.shape[0],input.shape[-2],input.shape[-1]),requires_grad=inference_pkg.remember_input_graph).to(inference_pkg.device) # simmap : (N,H,W)
    (N,H,W) = simmap_total.shape
    last_out = inference_pkg.input_dict[o.layer_list[0].name].to(inference_pkg.device)
    scale_rate = 1
    for m in reversed(o.layer_list):
        if isinstance(m,nn.Conv2d):
            conv = m
        if isinstance(m,nn.ReLU):
            relu = m
        if isinstance(m,nn.MaxPool2d):
            maxpool = m
    h_p = maxpool.kernel_size
    w_p = maxpool.kernel_size
    h_p_s = maxpool.stride
    w_p_s = maxpool.stride
    maxpool_padding = (maxpool.padding,maxpool.padding,maxpool.padding,maxpool.padding)
    
    w = conv.weight.shape[-1]
    h = conv.weight.shape[-2]
    stride = conv.stride
    
    bias = conv.bias
    padding = conv.padding
    conv_padding = (conv.padding[0],conv.padding[1],conv.padding[0],conv.padding[1])
    total_norm = torch.zeros((last_out.shape[0],1,last_out.shape[-2],last_out.shape[-1]),device=inference_pkg.device)
    for _w in range(w):
            for _h in range(h):
                for _w_p in range(w_p):
                    for _h_p in range(h_p):
                
                        _input = torch.zeros_like(input)
                        _input[:,:,_w_p::w_p_s,_h_p::h_p_s] = input[:,:,_w_p::w_p_s,_h_p::h_p_s]
                        _input = F.pad(_input,maxpool_padding,'constant',0)
                        _input = maxpool.decomp_forward(_input,inference_pkg)

                        weight = torch.zeros_like(conv.weight)
                        weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
                        if bias == None:
                            out = F.conv2d(_input,weight,bias,stride,padding)
                        else:
                            out = F.conv2d(_input,weight,torch.zeros_like(bias),stride,padding)
                        total_norm += torch.norm(out,dim=1,keepdim=True).detach()


    for _w in range(w):
        for _h in range(h):
            for _w_p in range(w_p):
                for _h_p in range(h_p):
            
                    _input = torch.zeros_like(input)
                    _input[:,:,_w_p::w_p_s,_h_p::h_p_s] = input[:,:,_w_p::w_p_s,_h_p::h_p_s]
                    _input = F.pad(_input,maxpool_padding,'constant',0)
                    _input = maxpool.decomp_forward(_input,inference_pkg)

                    weight = torch.zeros_like(conv.weight)
                    weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
                    if bias == None:
                        out = F.conv2d(_input,weight,bias,stride,padding)
                    else:
                        out = F.conv2d(_input,weight,torch.zeros_like(bias),stride,padding)
                    _,c,H_c,W_c = out.shape
                    out_norm = torch.norm(out,dim=1,keepdim=True).detach()
                    bias_bunbae = out_norm / total_norm
                    bias_contribution = bias.reshape(1,c,1,1) * bias_bunbae
                    out += bias_contribution

                    
                    cos = nn.CosineSimilarity(dim=1, eps=1e-12)
                    #calculate simmap for given position
                    simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12) # the contribution
                    
                    simmap *= o.simmap
                    simmap = bed_of_nail_upsample2d(simmap,scale_factor=stride,offset = (_h,_w), padding= conv_padding)
                    simmap = simmap[:,padding[0]:H_c+padding[0],padding[1]:H_c + padding[1]]
                    #maxpool upsample
                    simmap = bed_of_nail_upsample2d(simmap,scale_factor=(maxpool.stride,maxpool.stride),offset = (_w_p,_h_p),padding= maxpool_padding)
                    simmap = simmap[:,maxpool.padding:H+maxpool.padding,maxpool.padding:W + maxpool.padding]
                    simmap_total += simmap

    return simmap_total 

def cal_simmap_mcbr(input,o,inference_pkg,cur_layer_name=None):
    """
    input과 o.last_layer 사이 simmap을 계산
    maxpool2d-conv-relu만 가능
    """
    simmap_total = torch.zeros((input.shape[0],input.shape[-2],input.shape[-1]),requires_grad=inference_pkg.remember_input_graph).to(inference_pkg.device) # simmap : (N,H,W)
    (N,H,W) = simmap_total.shape
    last_out = inference_pkg.input_dict[o.layer_list[0].name].to(inference_pkg.device)
    scale_rate = 1
    (_,c_i,h_i,w_i) = input.shape
    for m in reversed(o.layer_list):
        if isinstance(m,nn.Conv2d):
            conv = m
            
        if isinstance(m,nn.ReLU):
            relu = m
        if isinstance(m,nn.MaxPool2d):
            maxpool = m
        if isinstance(m,nn.BatchNorm2d):
            bn = m
    h_p = maxpool.kernel_size
    w_p = maxpool.kernel_size
    h_p_s = maxpool.stride
    w_p_s = maxpool.stride
    maxpool_padding = (maxpool.padding,maxpool.padding,maxpool.padding,maxpool.padding)
    
    w = conv.weight.shape[-1]
    h = conv.weight.shape[-2]
    stride = conv.stride
    true_bias = inference_pkg.bn_true_bias_dict[bn.name]
    bias = conv.bias
    padding = conv.padding
    conv_padding = (conv.padding[0],conv.padding[1],conv.padding[0],conv.padding[1])
    total_norm = torch.zeros((last_out.shape[0],1,last_out.shape[-2],last_out.shape[-1]),device=inference_pkg.device)
    for _w in range(w):
        for _h in range(h):
                for _w_p in range(w_p):
                    for _h_p in range(h_p):
                
                        indices = inference_pkg.pool_indices_dict[maxpool.name].clone()
                        unpool = nn.MaxUnpool2d(maxpool.kernel_size,maxpool.stride,maxpool.padding)
                        maxpool_output = inference_pkg.output_dict[maxpool.name].to(inference_pkg.device)
                        unpooled_input = unpool(maxpool_output,indices,output_size = input.size())
                        unpooled_input = F.pad(unpooled_input,maxpool_padding,'constant',0)
                        _input = unpooled_input[:,:,_w_p:_w_p+w_i:w_p_s,_h_p:_h_p+h_i:h_p_s]
                        _input[_input != maxpool_output] = 0

                        weight = torch.zeros_like(conv.weight)
                        weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
                        if bias == None:
                            out = F.conv2d(_input,weight,bias,stride,padding)
                        else:
                            out = F.conv2d(_input,weight,torch.zeros_like(bias),stride,padding)

                        out = bn.decomp_forward(out,inference_pkg,bias_shrink =torch.inf)
                        total_norm += torch.norm(out,dim=1,keepdim=True).detach()
    for _w in range(w):
        for _h in range(h):
            for _w_p in range(w_p):
                for _h_p in range(h_p):
            
                    indices = inference_pkg.pool_indices_dict[maxpool.name].clone()
                    unpool = nn.MaxUnpool2d(maxpool.kernel_size,maxpool.stride,maxpool.padding)
                    maxpool_output = inference_pkg.output_dict[maxpool.name].to(inference_pkg.device)
                    unpooled_input = unpool(maxpool_output,indices,output_size = input.size())
                    unpooled_input = F.pad(unpooled_input,maxpool_padding,'constant',0)
                    _input = unpooled_input[:,:,_w_p:_w_p+w_i:w_p_s,_h_p:_h_p+h_i:h_p_s]
                    _input[_input != maxpool_output] = 0

                    weight = torch.zeros_like(conv.weight)
                    weight[:,:,_h,_w] = conv.weight[:,:,_h,_w]
                    if bias == None:
                        out = F.conv2d(_input,weight,bias,stride,padding)
                    else:
                        out = F.conv2d(_input,weight,bias / (w_p*h_p*w*h),stride,padding)
                    _,c,H_c,W_c = out.shape
                    out = bn.decomp_forward(out,inference_pkg,bias_shrink =torch.inf)
                    out_norm = torch.norm(out,dim=1,keepdim=True).detach()
                    bias_bunbae = out_norm / total_norm
                    bias_contribution = true_bias.reshape(1,c,1,1) * bias_bunbae
                    out += bias_contribution
                    cos = nn.CosineSimilarity(dim=1, eps=1e-12)
                    #calculate simmap for given position
                    simmap = cos(out,last_out) * torch.norm(out,dim=1) / (torch.norm(last_out,dim=1) + 1e-12) # the contribution
                    simmap *= o.simmap
                    #conv upsample
                    simmap = bed_of_nail_upsample2d(simmap,scale_factor=stride,offset = (_h,_w), padding= conv_padding)
                    simmap = simmap[:,padding[0]:H_c+padding[0],padding[1]:H_c + padding[1]]
                    #maxpool upsample
                    simmap = bed_of_nail_upsample2d(simmap,scale_factor=(maxpool.stride,maxpool.stride),offset = (_w_p,_h_p),padding= maxpool_padding)
                    simmap = simmap[:,maxpool.padding:H+maxpool.padding,maxpool.padding:W + maxpool.padding]
                    simmap_total += simmap
    return simmap_total 

def zero_out_minus(tensor):
    tensor = F.relu(tensor)
    return tensor

def zero_out_plus(tensor):
    tensor = F.relu(-tensor)
    return -tensor

def bed_of_nail_upsample2d(tensor,scale_factor,offset=(0,0),padding=(0,0,0,0),target_shape = None):
    """
    bed-of-nail upsample for (N,H,W) tensor
    first padding and then offset
    for example, [[1,2],
                  [3,4]] with scale_factor (2,2), offset = (0,0) ->

                  [[1,0,2,0],
                   [0,0,0,0],make_dir
                   [3,0,4,0],
                   [0,0,0,0]]

             and [[1,2],
                  [3,4]] with scale_factor (2,2), offset = (1,1), ->

                  [[0,0,0,0],
                   [0,1,0,2],
                   [0,0,0,0],
                   [0,3,0,4]]
             and [[1,2],
                  [3,4]] with scale_factor (2,2), offset = (0,0), padding (1,1,1,1) ->

                  [[1,0,2,0,0,0],
                   [0,0,0,0,0,0],
                   [3,0,4,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0]]
    
    

    """


    #make indice
    ones = torch.ones_like(tensor)
    (N,H,W) = tensor.shape
    indices = ones.nonzero(as_tuple=True)
    indices_n = indices[0]
    indices_h = scale_factor[0] * indices[1] + offset[0]
    indices_w = scale_factor[1] * indices[2] + offset[1]
    new_indices = (indices_n,indices_h,indices_w)


    device = tensor.device
    if target_shape ==None:
        upsample = torch.zeros(N,scale_factor[0]*H,scale_factor[1]*W,requires_grad=True).to(device)
        upsample = F.pad(upsample,padding,'constant',0)
    else:
        upsample = torch.zeros(target_shape,device=device) # TODO:수정 필요. 패딩있을때 안됨. 대신 사이즈때문에 컨볼루션 짤리는 경우는 이걸로 가능.
    upsample[new_indices] = tensor[indices]
    
    return upsample

def compute_sharing_ratio(v_x,v_o):
    cos = nn.CosineSimilarity(dim=1)
    sharing_ratio = cos(v_x,v_o) * torch.norm(v_x,dim=1) / (torch.norm(v_o,dim=1) + 1e-12)
    return sharing_ratio

def load_image(data_mean, data_std, device, image_name):
    """
    Helper method to load an image into a torch tensor. Includes preprocessing.
    """
    im = Image.open(image_name)
    x = torchvision.transforms.Normalize(mean=data_mean, std=data_std)(
        torchvision.transforms.ToTensor()(
            torchvision.transforms.CenterCrop(224)(torchvision.transforms.Resize(256)(im))))
    x = x.unsqueeze(0).to(device)
    return x



def make_dir(directory_name):
    if not directory_name:
        return ''
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if directory_name[-1] != '/':
        directory_name = directory_name + '/'

    return directory_name