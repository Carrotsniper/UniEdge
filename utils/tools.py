import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from einops import rearrange
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch.nn.functional as F
import math
import torch.nn as nn
plt.switch_backend('agg')

def pdd_matrix_edges(original_obs, patch_len, stride, dst=None):
    # original_obs => [N, T, 2]
    decompose = original_obs.unfold(dimension=1, size=patch_len, step=stride) 
    decompose = rearrange(decompose, 'n l d m -> l n m d')  
    edge_list = []
    data_list = []  
    edge_index_for_edge = []
    
    for i in range(decompose.shape[0]): 
        coordinates = decompose[i]  
        coordinates = rearrange(coordinates, 'n t d -> (n t) d')  
        dist_matrix = torch.cdist(coordinates, coordinates, p=2) 

        epsilon = 1e-6
        dist_matrix += epsilon

        adj_matrix = torch.zeros_like(dist_matrix)

        adj_matrix = (dist_matrix < dst).float()
        adj_matrix.fill_diagonal_(0)

        edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        edge_attr = dist_matrix[edge_index[0], edge_index[1]]  
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr=edge_attr)
        
        start_nodes = edge_index[0].unsqueeze(1).expand(-1, edge_index.size(1)).to(edge_index.device)
        end_nodes = edge_index[1].unsqueeze(1).expand(-1, edge_index.size(1)).to(edge_index.device)
        shared_start = start_nodes == start_nodes.t()
        shared_end = end_nodes == end_nodes.t()
        shared_mixed = (start_nodes == end_nodes.t()) | (end_nodes == start_nodes.t()).to(edge_index.device)

        shared_nodes_mask = (shared_start | shared_end | shared_mixed) & ~torch.eye(edge_index.size(1), dtype=torch.bool).to(edge_index.device)

        edge_graph_indices = shared_nodes_mask.nonzero(as_tuple=False)
        edge_graph_index = edge_graph_indices.t().contiguous()


        edge_list.append(edge_index)
        data_list.append(edge_attr)
        edge_index_for_edge.append(edge_graph_index)


    return edge_list, data_list, edge_index_for_edge


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        (f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe



def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)