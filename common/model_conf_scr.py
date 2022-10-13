from turtle import forward
from .model import TemporalModelBase
import torch.nn as nn
from torch.nn import functional as F
import torch
from einops import rearrange

class ConfTemporalModelV3Base(TemporalModelBase):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels,  dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.expand_conv     = nn.Conv1d(num_joints_in*2, channels, filter_widths[0], bias=False)
        self.expand_conv_conf= nn.Conv1d(num_joints_in,   channels, filter_widths[0], bias=False)
        self.expand_bn_conf = nn.BatchNorm1d(channels, momentum=0.1)
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        
    def set_bn_momentum(self, momentum):
        for layer in self.modules():
            if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm1d): 
                layer.momentum = momentum

    def _forward_blocks(self, x, cnf):
        x   = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        cnf = self.drop(self.relu(self.expand_bn_conf(self.expand_conv_conf(cnf))))
        
        for i in range(len(self.pad) - 1):
            pad     = self.pad[i+1]
            shift   = self.causal_shift[i+1]
            x, cnf  = self.layers[i](x, cnf, pad, shift)
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

class ConfConvBlockBase(nn.Module):
    def __init__(self,  channels, kernel_size, dilation, dropout, activation='relu') -> None:
        super().__init__()
        if activation == 'relu':
            ActLayer = nn.ReLU
        elif activation == 'sigmoid':
            ActLayer = nn.Sigmoid
        elif activation == 'tanh':
            ActLayer = nn.Tanh()
        else:
            raise NotImplementedError
        self.pos_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels, momentum=0.1),
            ActLayer()
        )
        self.conf_conv= nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels, momentum=0.1),
            ActLayer()
        )
        self.feat_conv= nn.Sequential(
            nn.Conv1d(channels, channels, 1, dilation=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.1),
            ActLayer()
        ) 
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, cnf, pad, shift):
        # Get the residual
        res = x[:, :, pad + shift : x.shape[2] - pad + shift]
        x   = self.drop(self.pos_conv(x))
        cnf = self.drop(self.conf_conv(cnf))
        assert x.shape == cnf.shape
        x   = x * cnf
        x = res + self.drop(self.feat_conv(x))
        return x, cnf

class ConfTemporalModelV33(ConfTemporalModelV3Base):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense)
        layers = []
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            layers.append(ConfConvBlockBase(channels, filter_widths[i], dilation=next_dilation, dropout=dropout))
            next_dilation *= filter_widths[i]
        self.layers = nn.ModuleList(layers)
    
class ConfConvBlockInput(ConfConvBlockBase):
    def __init__(self, channels, kernel_size, dilation, dropout, leak = 0.0, activation='relu') -> None:
        super().__init__(channels, kernel_size, dilation, dropout, activation=activation)
        self.leak = leak 
    def forward(self, x, cnf, pad, shift):
        # Get the residual
        res = x[:, :, pad + shift : x.shape[2] - pad + shift]
        assert x.shape == cnf.shape
        x   = self.drop(self.pos_conv(x * (self.leak + cnf)))
        cnf = self.drop(self.conf_conv(cnf))
        x = res + self.drop(self.feat_conv(x))
        return x, cnf

class ConfTemporalModelV34(ConfTemporalModelV3Base):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense)
        layers = []
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            layers.append(ConfConvBlockInput(channels, filter_widths[i], dilation=next_dilation, dropout=dropout))
            next_dilation *= filter_widths[i]
        self.layers = nn.ModuleList(layers)

class ConfTemporalModelV34sigmoid(ConfTemporalModelV3Base):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense)
        layers = []
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            layers.append(ConfConvBlockInput(channels, filter_widths[i], dilation=next_dilation, dropout=dropout,activation='sigmoid',leak=0.5))
            next_dilation *= filter_widths[i]
        self.layers = nn.ModuleList(layers)

class ConfTemporalModelV34tanh(ConfTemporalModelV3Base):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense)
        layers = []
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            layers.append(ConfConvBlockInput(channels, filter_widths[i], dilation=next_dilation, dropout=dropout,activation='tanh',leak=1.0))
            next_dilation *= filter_widths[i]
        self.layers = nn.ModuleList(layers)

class FeatFusionConv(nn.Module):
    def __init__(self, channels, kernel_size, dilation) -> None:
        super().__init__()
        self.conv_cnf = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=False)
        self.conv_x   = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=False)
        self.bn       = nn.BatchNorm1d(channels, momentum=0.1)
        self.relu     = nn.ReLU()
    def forward(self, x, cnf):
        o  = self.relu(self.bn(self.conv_x(x) + self.conv_cnf(cnf)))
        return o

class ConfConvBlockInputFusion(ConfConvBlockBase):
    def __init__(self, channels, kernel_size, dilation, dropout) -> None:
        super().__init__(channels, kernel_size, dilation, dropout)
        self.conf_conv= FeatFusionConv(channels, kernel_size, dilation)

    def forward(self, x, cnf, pad, shift):
        # Get the residual
        res = x[:, :, pad + shift : x.shape[2] - pad + shift]
        assert x.shape == cnf.shape
        cnf_o = self.drop(self.conf_conv(x, cnf))
        x   = self.drop(self.pos_conv(x * (1 + cnf)))
        x   = res + self.drop(self.feat_conv(x))
        return x, cnf_o

class ConfTemporalModelV35(ConfTemporalModelV3Base):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense)
        layers = []
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            layers.append(ConfConvBlockInputFusion(channels, filter_widths[i], dilation=next_dilation, dropout=dropout))
            next_dilation *= filter_widths[i]
        self.layers = nn.ModuleList(layers)

class ConfConvBlockInputDualFusion(ConfConvBlockBase):
    def __init__(self, channels, kernel_size, dilation, dropout) -> None:
        super().__init__(channels, kernel_size, dilation, dropout)
        self.conf_conv= FeatFusionConv(channels, kernel_size, dilation)
        self.pos_conv= FeatFusionConv(channels, kernel_size, dilation)

    def forward(self, x, cnf, pad, shift):
        # Get the residual
        res = x[:, :, pad + shift : x.shape[2] - pad + shift]
        assert x.shape == cnf.shape
        cnf_o = self.drop(self.conf_conv(x, cnf))
        x   = self.drop(self.pos_conv(x, cnf))
        x   = res + self.drop(self.feat_conv(x))
        return x, cnf_o

class ConfTemporalModelV36(ConfTemporalModelV3Base):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dense)
        layers = []
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            layers.append(ConfConvBlockInputDualFusion(channels, filter_widths[i], dilation=next_dilation, dropout=dropout))
            next_dilation *= filter_widths[i]
        self.layers = nn.ModuleList(layers)