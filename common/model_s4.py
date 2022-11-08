import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.S4.s4 import S4 as S4Layer
from common.S4.s4block import *
import torch.autograd.profiler as profiler

class S4Model(nn.Module):
    def __init__(self, num_joints_in, in_features, num_joins_out, num_layers, 
                    dropout=0.25, channels=128, bidirectional=False):
        super(S4Model, self).__init__()
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joins_out
        self.in_features = in_features
        self.input_dim = in_features*num_joints_in
        self.output_dim = (in_features+1)*num_joins_out
        self.num_layers = num_layers
        self.d_model = channels
        layers = list()
        for _ in range(num_layers):
            layers.append(S4Layer(self.d_model, bidirectional=bidirectional, dropout=dropout))
        self.s4_sequence = nn.ModuleList(layers)
        self.encoder = nn.Linear(self.input_dim, self.d_model)
        self.decoder = nn.Linear(self.d_model, self.output_dim)
    
    def forward(self, x):
        #x=(B,J,D,T)
        x = rearrange(x, 'b t j d -> b t (j d)', j=self.num_joints_in, d=self.in_features)
        with profiler.record_function("encoder"):
            x = self.encoder(x)
        x = rearrange(x, 'b t d -> b d t')
        with profiler.record_function("s4_sequence"):
            for i, layer in enumerate(self.s4_sequence):
                x, next_state = layer(x)
        x = rearrange(x, 'b d t -> b t d')
        with profiler.record_function("decoder"):
            x = self.decoder(x)
        x = rearrange(x, 'b t (j d) -> b t j d', j=self.num_joints_out, d=self.in_features+1)
        return x[:, x.shape[1]//2-1:x.shape[1]//2, :, :]
    
class S4ModelBlock(nn.Module):
    def __init__(self, num_joints_in, in_features, num_joins_out, dropout=0.25, channels=64, bidirectional=False, ff=2, architecture='3,3,3'):
        super(S4ModelBlock, self).__init__()
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joins_out
        self.in_features = in_features
        self.input_dim = in_features*num_joints_in
        self.output_dim = (in_features+1)*num_joins_out
        self.hidden_dim = channels
        self.arc = architecture.split(',')
        self.num_blocks = len(self.arc)

        def s4_block(dim):
            layer = S4Layer(
                d_model=dim,
                d_state=64,
                bidirectional=bidirectional,
                dropout=dropout,
                transposed=True
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
            )

        def ff_block(dim):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=dropout,
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
            )

        self.encoder = nn.Linear(self.input_dim, channels)

        pool = [int(ks) for ks in self.arc]
        d_layers = []
        for p in pool:
            for _ in range(self.num_blocks):
                d_layers.append(s4_block(channels))
                if ff > 0: d_layers.append(ff_block(channels))
            d_layers.append(DownPool(channels, 2, p))
            channels *= 2
        self.s4_sequence = nn.ModuleList(d_layers)

        self.decoder = nn.Linear(channels, self.output_dim)
    
    def forward(self, x):
        x = rearrange(x, 'b t j d -> b t (j d)', j=self.num_joints_in, d=self.in_features)
        x = self.encoder(x)
        x = rearrange(x, 'b t d -> b d t' )
        with profiler.record_function("s4_sequence"):
            for layer in self.s4_sequence:
                x, state = layer(x)
        # import ipdb; ipdb.set_trace()
        x = self.decoder(x.permute(0,2,1))
        x = rearrange(x, 'b t (j d)-> b t j d', j=self.num_joints_out, d=self.in_features+1)
        return x


if __name__ == "__main__":
    # model = S4Model(17, 2, 17, 4).cuda()
    model = S4ModelBlock(17, 2, 17).cuda()
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        x = torch.randn(1024, 27, 17, 2).cuda()
        y = model(x)
    print(y.shape)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))  