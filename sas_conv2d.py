# Copyright (C) 2025 Denso IT Laboratory, Inc.
# All Rights Reserved

import math
import torch
from torch import autograd, nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_


    
########################################################################
# SASConv2d
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class SASConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=[0,0], dilation=1, groups=1, bias=True, sparse_m=2, sparse_n=2, is_conv=True):    
        super(SASConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride    
        self.sparse_m = sparse_m
        self.sparse_n = sparse_n
        self.is_conv = is_conv
       
        if not isinstance(stride, (tuple, list)):
            stride = (stride, stride)
        if is_conv:
            self.register_parameter("weight", nn.Parameter(torch.zeros(out_channels, sparse_m*in_channels//groups, kernel_size[0], kernel_size[1])))
        else:
            self.register_parameter("weight", nn.Parameter(torch.zeros(out_channels, sparse_m*in_channels//groups)))

        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None

        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            fan_in = fan_in / self.sparse_m          
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


        
    def SAS_proj(self, X):

        if self.sparse_m==1:
            return X
        
        M = np.abs(self.sparse_m)
        
        S = X.shape
        B,C,H,W = X.shape
        S   = [B, C, M, H, W]

        x_proj = torch.zeros(S, dtype=X.dtype, device=X.device) 
                
        Y = X.abs()

        with torch.no_grad():
            sgn = (X>0)       # sgn = [Flase, True, False, True, ...]
        yp = Y * sgn.float()
        yn = Y * (1.0 - sgn.float())
        x_proj[:,:,0,:,:] = yp    
        x_proj[:,:,1,:,:] = yn  
        x_proj = x_proj.view(B, C*M, H, W)

        return x_proj
    

    def forward(self, x):
        # Sparse Projection
        x_proj = self.SAS_proj(x)
        output = F.conv2d(x_proj, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) 
        return output

