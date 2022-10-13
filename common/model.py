# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from audioop import bias
import torch.nn as nn
from torch.nn import functional as F
import torch
from einops import rearrange
class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()
        
        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)
        

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames
    
    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames
        
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = self._forward_blocks(x)
        
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        
        return x    

class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x
    
class ConfTemporalModelV1(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.embed_dim   = 32
        self.embed_layer = nn.Sequential(
            nn.Linear(2, self.embed_dim, bias=False),
            nn.ReLU())
        self.expand_conv = nn.Conv1d(self.embed_dim * self.num_joints_in, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
       
        sz = x.shape[:3]
        scr = x[..., -1].contiguous() # get conf. score
        x = x[..., :-1].contiguous()  # get x,y position
        x = self.embed_layer(x)
        # normalize conf score
        scr = torch.softmax(scr, dim=1)
        # scale the input by conf. score
        x = x * scr[..., None]
        x = rearrange(x, 'b f j d -> b (j d) f')
        x = self._forward_blocks(x)
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        return x    

class ConfTemporalModelV2(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.expand_conv = nn.Conv1d(num_joints_in*2, channels, filter_widths[0], bias=False)
        self.filter_widths = filter_widths

        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn   = nn.ModuleList(layers_bn)
         
        
    def _forward_blocks(self, x, scr):
        # c = F.conv1d(scr, torch.ones(1, 1, self.filter_widths[0]).to(scr.device)) + 1e-6
        # x = self.drop(self.relu(self.expand_bn(self.expand_conv(x * scr)/c)))
        # scr = c/self.filter_widths[0]
        scr = F.conv1d(scr, torch.ones(1, 1, self.filter_widths[0]).to(scr.device)/self.filter_widths[0]) + 1e-6
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))) * scr)
        next_dilation = self.filter_widths[0]
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            # c = F.conv1d(scr, torch.ones(1, 1, self.filter_widths[i]).to(scr.device),dilation=next_dilation) + 1e-6
            # x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x * scr)/c)))
            scr = F.conv1d(scr, torch.ones(1, 1, self.filter_widths[i]).to(scr.device)/self.filter_widths[i], dilation=next_dilation) + 1e-6
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))) * scr)
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))     
            # Update conf score for the current layer output
            # scr = c/self.filter_widths[i]
            next_dilation *= self.filter_widths[i]
        x = self.shrink(x)
        return x

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
       
        sz = x.shape[:3]
        scr = x[..., -1].contiguous() # get conf. score
        x = x[..., :-1].contiguous()  # get x,y position
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        scr = scr.mean(dim=-1)[...,None]
        scr = scr.permute(0, 2, 1)
        # compute by frame conf. scr
        # normalize conf score
        x = self._forward_blocks(x, scr)
        
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        return x    

 

class ConfTemporalModelV4(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.embed_dim   = 32
        self.filter_widths = filter_widths
        
        self.embed_layer = nn.Sequential(
            nn.Linear(2, self.embed_dim, bias=False),
            nn.ReLU())
        
        self.l1_conv = nn.Conv2d(self.embed_dim, self.embed_dim, (1, filter_widths[0]), bias=False)
        
        self.l1_bn   = nn.BatchNorm2d(self.embed_dim, momentum=0.1)
        self.shrink = nn.Conv1d(self.embed_dim * num_joints_in, num_joints_out*3, 1)
        
        layers_conv      = []
        layers_bn        = []
        layers_spatial   = []
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            layers_conv.append(nn.Conv2d(self.embed_dim, self.embed_dim,
                                         (1, filter_widths[i]),
                                         dilation= (1, next_dilation),
                                         bias=False))
            layers_spatial.append(nn.Conv2d(num_joints_in, 1,
                                         (1, 1), bias=False))
            layers_bn.append(nn.BatchNorm2d(self.embed_dim, momentum=0.1))
            layers_conv.append(nn.Conv2d(self.embed_dim, self.embed_dim, (1,1), dilation=(1,1), bias=False))
            layers_bn.append(nn.BatchNorm2d(self.embed_dim, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv   = nn.ModuleList(layers_conv)
        self.layers_spatial= nn.ModuleList(layers_spatial)
        self.layers_bn     = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x, scr):
        x   = self.drop(self.relu(self.l1_bn(self.l1_conv(x))))
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[..., pad + shift : x.shape[-1] - pad + shift]
            x   = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x_spatial = self.layers_spatial[i](x.permute(0,2,1,3))
            x = x   + x_spatial.permute(0,2,1,3)
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
            
        x = x.view(x.shape[0], -1, 1)
        x = self.shrink(x)
        return x

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
       
        sz = x.shape[:3]
        scr = x[..., -1].contiguous() # get conf. score
        x   = x[..., :-1].contiguous()  # get x,y position
        x   = self.embed_layer(x)
        x   = x.permute(0, 3, 2, 1)
        x   = self._forward_blocks(x, scr)
        x   = x.permute(0, 2, 1)
        x   = x.view(sz[0], -1, self.num_joints_out, 3)
        return x  

class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.
    
    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0] // 2) if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x


class ConfTemporalModelV3BaseOld(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.expand_conv     = nn.Conv1d(num_joints_in*2, channels, filter_widths[0], bias=False)
        self.expand_conv_conf= nn.Conv1d(num_joints_in,   channels, filter_widths[0], bias=False)
        self.filter_widths = filter_widths
        self.expand_bn_conf = nn.BatchNorm1d(channels, momentum=0.1)
        layers_conv      = []
        layers_bn        = []
        layers_conv_conf = []
        layers_bn_conf   = []
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_conv_conf.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_bn_conf.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn   = nn.ModuleList(layers_bn)
        self.layers_conv_conf = nn.ModuleList(layers_conv_conf)
        self.layers_bn_conf =  nn.ModuleList(layers_bn_conf)
    
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
       
        sz = x.shape[:3]
        scr = x[..., -1].contiguous() # get conf. score
        x   = x[..., :-1].contiguous()  # get x,y position
        x   = x.view(x.shape[0], x.shape[1], -1)
        x   = x.permute(0, 2, 1)
        scr = scr.permute(0, 2, 1)
        x = self._forward_blocks(x, scr)
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        return x   

class ConfTemporalModelV3(ConfTemporalModelV3BaseOld):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.sigmoid     = nn.Sigmoid()
        
    def _forward_blocks(self, x, scr):
        x   = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        scr = self.sigmoid(self.expand_bn_conf(self.expand_conv_conf(scr)))
        x   = x * scr
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x   = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            scr = self.sigmoid(self.layers_bn_conf[i](self.layers_conv_conf[i](scr)))
            assert x.shape == scr.shape
            x = x * scr
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
       
        sz = x.shape[:3]
        scr = x[..., -1].contiguous() # get conf. score
        x   = x[..., :-1].contiguous()  # get x,y position
        x   = x.view(x.shape[0], x.shape[1], -1)
        x   = x.permute(0, 2, 1)
        scr = scr.permute(0, 2, 1)
        
        x = self._forward_blocks(x, scr)
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        return x

class ConfTemporalModelV31(ConfTemporalModelV3BaseOld):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.sigmoid     = nn.Sigmoid()
        
    def _forward_blocks(self, x, scr):
        x   = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        scr = self.drop(self.sigmoid(self.expand_bn_conf(self.expand_conv_conf(scr))))
        x   = x + x * scr
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x   = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            scr = self.drop(self.sigmoid(self.layers_bn_conf[i](self.layers_conv_conf[i](scr))))
            assert x.shape == scr.shape
            x = x + x * scr
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x

class ConfTemporalModelV32(ConfTemporalModelV3BaseOld):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.sigmoid     = nn.Sigmoid()
        
    def _forward_blocks(self, x, scr):
        x   = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        scr = self.drop(self.sigmoid(self.expand_bn_conf(self.expand_conv_conf(scr))))
        x   = x + x * scr
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            x   = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x + x*scr))))
            scr = self.drop(self.sigmoid(self.layers_bn_conf[i](self.layers_conv_conf[i](scr))))
            assert x.shape == scr.shape
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
       
        sz = x.shape[:3]
        scr = x[..., -1].contiguous() # get conf. score
        x   = x[..., :-1].contiguous()  # get x,y position
        x   = x.view(x.shape[0], x.shape[1], -1)
        x   = x.permute(0, 2, 1)
        scr = scr.permute(0, 2, 1)
        x = self._forward_blocks(x, scr)
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        return x