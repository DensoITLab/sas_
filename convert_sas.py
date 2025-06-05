# Copyright (C) 2025 Denso IT Laboratory, Inc.
# All Rights Reserved

import os
import torch
import torch.nn as nn
from sas_conv2d import SASConv2d
from sas_linear import SASLinear


def convert_layers(model, use_sas):

    conversion_count = 0

    for name, module in list(model._modules.items()):
        if module is not None and len(list(module.children())) > 0:
            model._modules[name], num_converted = convert_layers(module, use_sas)
            conversion_count += num_converted
        
        # If use_sas is enabled, perform preprocessing to replace activations
        if use_sas:
            # Replace ReLU with Identity and set a flag _was_relu
            if isinstance(module, nn.ReLU):
                identity_module = nn.Identity()
                identity_module._was_relu = True
                model._modules[name] = identity_module
                conversion_count += 1
                module = identity_module             
            # If module is Identity (with no _was_relu flag), replace it with ReLU
            elif isinstance(module, nn.Identity) and not hasattr(module, '_was_relu'):
                new_relu = nn.ReLU()
                model._modules[name] = new_relu
                conversion_count += 1
                module = new_relu    
            
        if isinstance(module, nn.Conv2d):
            if use_sas and module.in_channels == 3:
                continue

            if use_sas:
                # Replace with SASConv2d
                new_layer = SASConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=(module.bias is not None),
                    sparse_m=2,
                    sparse_n=2,
                    is_conv=True
                )
                with torch.no_grad():
                    # Copy original convolution weights into the new sparse structure:
                    orig_weight = module.weight.data
                    # Expand the weight along the input‐channel dimension
                    expanded_weight = orig_weight.repeat_interleave(new_layer.sparse_m, dim=1)
                    expanded_weight[:, 1::2, :, :] = 0
                    new_layer.weight.data.copy_(expanded_weight)

                    if module.bias is not None:
                        new_layer.bias.data.copy_(module.bias.data)
            else:
                # Replace with dummy Conv2d
                new_layer = dummy_conv(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=(module.bias is not None),
                )
                with torch.no_grad():
                    # Copy the original weight and bias to the dummy_conv submodule
                    new_layer.dummy_conv.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        new_layer.dummy_conv.bias.data.copy_(module.bias.data)

            model._modules[name] = new_layer
            conversion_count += 1
        
        elif isinstance(module, nn.Linear):
            if use_sas:
                # Replace with SASLinear
                new_layer = SASLinear(
                    in_channels=module.in_features,
                    out_channels=module.out_features,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=[0, 0],
                    dilation=1,
                    groups=1,
                    bias=(module.bias is not None),
                    sparse_m=2,
                    sparse_n=2,
                    is_conv=False
                )
                with torch.no_grad():
                    # Copy original linear weights into the new sparse structure:
                    orig_weight = module.weight.data
                    # Expand along the input dimension to match sparse_m
                    expanded_weight = orig_weight.repeat_interleave(new_layer.sparse_m, dim=1)
                    expanded_weight[:, 1::2] = 0
                    new_layer.weight.data.copy_(expanded_weight)

                    if module.bias is not None:
                        new_layer.bias.data.copy_(module.bias.data)
            else:
                # Replace with dummy Linear
                new_layer = dummy_linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=(module.bias is not None),
                )
                with torch.no_grad():
                    # Copy the original weight and bias to the dummy_linear submodule
                    new_layer.dummy_linear.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        new_layer.dummy_linear.bias.data.copy_(module.bias.data)

            model._modules[name] = new_layer
            conversion_count += 1
        
    return model, conversion_count


# Definition of dummy conv/linear layer

class dummy_conv(nn.Module):
    def __init__(self, *args, **kwargs):
        super(dummy_conv, self).__init__()
        self.dummy_conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        return self.dummy_conv(x)
    
    def init_weights(self):
        self.dummy_conv.reset_parameters()

class dummy_linear(nn.Module):
    def __init__(self, *args, **kwargs):
        super(dummy_linear, self).__init__()
        self.dummy_linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return self.dummy_linear(x)
    
    def init_weights(self):
        self.dummy_linear.reset_parameters()
